from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from context_guard import guard_request
from github_source import parse_github_target
from tool_core import ToolContext, ToolRegistry
from github_routing import extract_github_read_scope


MAX_TOOL_CALLS = 3
MAX_PROTOCOL_REPAIRS = 1
DISCIPLINE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(discipline_os/[A-Za-z0-9_./-]+\.(?:json|md|txt))(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
GITHUB_URL_RE = re.compile(r"https?://(?:www\.)?github\.com/[^\s)\]>]+", re.IGNORECASE)


class AgentProtocolError(RuntimeError):
    """Raised when the model does not follow the runtime tool protocol."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class FinalAnswer:
    content: str


def _strip_single_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 3:
        return stripped

    first = lines[0].strip().lower()
    if first not in {"```", "```json"}:
        return stripped
    if lines[-1].strip() != "```":
        return stripped

    return "\n".join(lines[1:-1]).strip()


def parse_agent_response(text: str) -> ToolCall | FinalAnswer:
    raw = _strip_single_json_fence(text)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Plain text is a valid final answer. Only JSON-looking output is treated
        # as a broken tool-protocol attempt.
        if raw.lstrip().startswith("{"):
            raise AgentProtocolError(
                "malformed JSON tool/final response"
            ) from exc
        return FinalAnswer(content=raw.strip())

    if not isinstance(payload, dict):
        raise AgentProtocolError("protocol response must be a JSON object")

    response_type = payload.get("type")

    if response_type == "tool_call":
        allowed = {"type", "name", "arguments"}
        unknown = set(payload) - allowed
        if unknown:
            raise AgentProtocolError(
                "tool_call contains unknown field(s): "
                + ", ".join(sorted(unknown))
            )

        name = payload.get("name")
        arguments = payload.get("arguments")

        if not isinstance(name, str) or not name.strip():
            raise AgentProtocolError("tool_call.name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise AgentProtocolError("tool_call.arguments must be a JSON object")

        return ToolCall(name=name, arguments=arguments)

    if response_type == "final":
        allowed = {"type", "content"}
        unknown = set(payload) - allowed
        if unknown:
            raise AgentProtocolError(
                "final contains unknown field(s): "
                + ", ".join(sorted(unknown))
            )

        content = payload.get("content")
        if not isinstance(content, str):
            raise AgentProtocolError("final.content must be a string")

        return FinalAnswer(content=content.strip())

    raise AgentProtocolError("type must be either 'tool_call' or 'final'")


def build_tool_protocol_prompt(registry: ToolRegistry) -> str:
    specs = json.dumps(
        registry.specs(),
        ensure_ascii=False,
        indent=2,
    )

    return f"""PAE RUNTIME TOOL PROTOCOL

You can either request exactly one runtime tool call or return the final answer.

AVAILABLE TOOLS
{specs}

GITHUB SCOPE
- Always respect the repository and exact file path explicitly requested by the user.
- hssgj/PAE is relevant ONLY to PAE itself, not a default for other repositories.
- Read discipline_os/current_state.json ONLY for explicit PAE priority/focus/state
  questions, never as fallback for an unrelated request or a failed read.
- A GitHub read failure is NOT permission to read a different file.
- Do not claim you checked GitHub unless you actually request github_read.

OUTPUT CONTRACT
To request a tool, return exactly ONE JSON object and nothing else:
{{"type":"tool_call","name":"github_read","arguments":{{"repo":"owner/repository","path":"README.md","ref":"main"}}}}

If no tool is needed, answer the user normally in plain text. A JSON final object
is also accepted:
{{"type":"final","content":"your answer here"}}

Rules:
- Never wrap a tool-call JSON object in explanatory prose.
- Use only registered tool names.
- Tool arguments must follow the declared schema.
- If the user's request requires current external/source data, request the tool
  before answering.
- After receiving a TOOL RESULT, use it to answer or request another tool.
- Never invent a tool result.
- Never say or imply that you checked, opened, refreshed, created, updated, sent,
  or will now perform an external action unless the matching registered tool
  executed successfully in this turn. Do not say "please wait" for work that the
  runtime is not actively executing.
- If no registered tool supports the requested external action, say clearly that
  the capability is not connected.
- gmail_create_draft creates a real Gmail draft but does not send it. Show its
  exact preview and ask the user for explicit confirmation.
- Never request gmail_send in the same user turn as gmail_create_draft. The runtime
  permits gmail_send only for the pending draft after a separate, unambiguous user
  confirmation message.
- A GitHub repository-root URL is a repository, not a file. Use github_list or
  github_search to discover paths; use github_read only with an exact text-file path.
- github_prepare_write only prepares one text-file diff. Never request
  github_apply_write in the same user turn. The runtime permits it only after a
  separate, unambiguous user confirmation.
"""



def build_session_facts_prompt(context: ToolContext) -> str:
    message_count = len(context.session.messages)
    source_count = len(context.session.sources)

    return (
        "SESSION RUNTIME FACTS\n"
        f"Session id: {context.session.session_id}\n"
        f"Persisted chat messages currently loaded: {message_count}\n"
        f"Persisted sources currently loaded: {source_count}\n"
        "The user/assistant messages supplied to this turn are persistent session "
        "history. If the persisted chat message count is greater than 1, do not "
        "claim that this is the first message or that prior chat history is absent."
    )


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", _normalize_text(text))


def _approx_any(tokens: list[str], targets: set[str], cutoff: float = 0.78) -> bool:
    for token in tokens:
        for target in targets:
            if token == target:
                return True
            if len(token) >= 4 and SequenceMatcher(None, token, target).ratio() >= cutoff:
                return True
    return False


def _last_user_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def is_pae_state_request(messages: list[dict[str, str]]) -> bool:
    """Authorize authoritative DOS state only for explicit PAE-state questions."""
    user_text = _last_user_text(messages)
    tokens = _tokens(user_text)
    if "discipline_os/current_state.json" in user_text:
        return True
    is_pae = bool(set(tokens) & {"pae", "dos", "discipline"})
    is_state = _approx_any(tokens, {
        "focus", "stav", "state", "priority", "priorita", "milestone",
        "next", "action", "dalsi", "aktualni", "current",
    })
    return is_pae and is_state


def is_dos_dump(content: str) -> bool:
    return (
        "Cesta: discipline_os/current_state.json" in content
        or (
            '"current_focus"' in content
            and '"open_external_items"' in content
        )
    )


def _last_reported_message_id(messages: list[dict[str, str]]) -> str:
    pattern = re.compile(r"Message ID:\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        match = pattern.search(message.get("content", ""))
        if match:
            return match.group(1)
    return ""


def _github_query_from_request(user_text: str, repo: str) -> str:
    without_url = GITHUB_URL_RE.sub(" ", user_text)
    stop = {
        "github", "repo", "repository", "read", "find", "search", "file", "folder",
        "chapter", "story", "please", "this", "that", "from", "inside", "look",
        "the", "a", "an",
        "najdi", "hledej", "precti", "soubor", "slozku", "kapitolu", "pribeh",
        "tohle", "tomto", "repu", "repozitari", "prosim",
    }
    repo_words = {part.casefold() for part in re.split(r"[/_.-]+", repo)}
    words = [
        word for word in re.findall(r"[A-Za-z0-9_.-]{3,}", _normalize_text(without_url))
        if word not in stop and word not in repo_words
    ]
    return " ".join(words[:4])


def required_tool_call(messages: list[dict[str, str]]) -> ToolCall | None:
    """Return a deterministic minimum tool call when source freshness is mandatory."""
    user_text = _last_user_text(messages)
    if not user_text:
        return None

    explicit_path = DISCIPLINE_PATH_RE.search(user_text)
    if explicit_path:
        return ToolCall(
            name="github_read",
            arguments={
                "repo": "hssgj/PAE",
                "path": explicit_path.group(1),
                "ref": "main",
            },
        )

    tokens = _tokens(user_text)

    read_terms = {"read", "show", "precti", "zobraz", "obsah"}
    reference_terms = {"that", "it", "email", "mail", "zpravu", "zprava", "ho", "ji"}
    referenced_message_id = _last_reported_message_id(messages)
    if (
        referenced_message_id
        and _approx_any(tokens, read_terms)
        and _approx_any(tokens, reference_terms)
    ):
        return ToolCall(
            name="gmail_read", arguments={"message_id": referenced_message_id, "thread_id": ""}
        )

    github_url = GITHUB_URL_RE.search(user_text)
    if github_url:
        try:
            target = parse_github_target(github_url.group(0).rstrip(".,;"))
        except ValueError:
            target = None
        if target is not None:
            if target.kind == "file" and target.path:
                return ToolCall(
                    name="github_read",
                    arguments={"repo": target.repo, "path": target.path, "ref": target.ref},
                )
            if target.kind == "directory":
                return ToolCall(
                    name="github_list",
                    arguments={"repo": target.repo, "path": target.path, "ref": target.ref, "limit": 50},
                )
            query = _github_query_from_request(user_text, target.repo)
            if query:
                return ToolCall(
                    name="github_search",
                    arguments={"repo": target.repo, "query": query, "ref": target.ref, "limit": 10},
                )
            return ToolCall(
                name="github_list",
                arguments={"repo": target.repo, "path": "", "ref": target.ref, "limit": 50},
            )

    mail_terms = {
        "email",
        "e-mail",
        "gmail",
        "mail",
        "posta",
        "postu",
        "posty",
        "zprava",
    }
    latest_terms = {
        "last",
        "latest",
        "newest",
        "posledni",
        "nejnovejsi",
        "aktualni",
    }
    if _approx_any(tokens, mail_terms) and _approx_any(tokens, latest_terms):
        return ToolCall(name="gmail_search", arguments={"query": "in:inbox", "limit": 1})

    freshness_terms = {
        "current",
        "curent",
        "fresh",
        "aktualni",
        "aktual",
        "autoritativni",
        "authoritative",
        "latest",
        "ted",
        "tedka",
    }
    state_terms = {
        "focus",
        "foks",
        "state",
        "stav",
        "priority",
        "priorita",
        "milestone",
        "next",
        "action",
        "akce",
        "dalsi",
    }

    mentions_freshness = _approx_any(tokens, freshness_terms)
    mentions_state = _approx_any(tokens, state_terms)

    if mentions_freshness and mentions_state and is_pae_state_request(messages):
        return ToolCall(
            name="github_read",
            arguments={
                "repo": "hssgj/PAE",
                "path": "discipline_os/current_state.json",
                "ref": "main",
            },
        )

    return None


def _same_required_call(candidate: ToolCall, required: ToolCall) -> bool:
    if candidate.name != required.name:
        return False

    for key, value in required.arguments.items():
        if candidate.arguments.get(key) != value:
            return False

    return True


def user_forbids_tools(messages: list[dict[str, str]]) -> bool:
    """Detect explicit history-only/no-GitHub instructions for this turn."""
    user_text = _last_user_text(messages)
    tokens = _tokens(user_text)
    normalized = _normalize_text(user_text)

    mentions_github = _approx_any(tokens, {"github", "githbu", "git"})
    mentions_tool = _approx_any(tokens, {"tool", "nastroj"})

    if "bez" in tokens and (mentions_github or mentions_tool):
        return True
    if "without" in tokens and (mentions_github or mentions_tool):
        return True
    if "no" in tokens and mentions_tool:
        return True
    if "nepouzivej" in normalized and (mentions_github or mentions_tool):
        return True
    if "dont use" in normalized and (mentions_github or mentions_tool):
        return True

    return False


def _protocol_error_message(error: Exception) -> dict[str, str]:
    return {
        "role": "system",
        "content": (
            "PROTOCOL ERROR\n"
            f"{error}\n"
            "Return exactly one valid JSON object using the PAE RUNTIME TOOL "
            "PROTOCOL. Do not add prose or Markdown."
        ),
    }


def _tool_result_message(
    name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, str]:
    return {
        "role": "system",
        "content": (
            "TOOL RESULT\n"
            f"Tool: {name}\n"
            "Arguments:\n"
            + json.dumps(arguments, ensure_ascii=False, indent=2)
            + "\nResult:\n"
            + json.dumps(result, ensure_ascii=False, indent=2)
            + "\n\nUse the tool result to answer the user normally in plain text, "
            "or request another tool with one JSON tool_call object."
        ),
    }


def _tool_error_message(
    name: str,
    arguments: dict[str, Any],
    error: Exception,
) -> dict[str, str]:
    return {
        "role": "system",
        "content": (
            "TOOL ERROR\n"
            f"Tool: {name}\n"
            "Arguments:\n"
            + json.dumps(arguments, ensure_ascii=False, indent=2)
            + f"\nError: {error}\n"
            "Use this error to answer the user normally in plain text, or correct "
            "the call with one JSON tool_call object if another attempt is justified."
        ),
    }




def _append_tool_attempt(
    working: list[dict[str, str]],
    action: ToolCall,
    *,
    registry: ToolRegistry,
    context: ToolContext,
) -> dict[str, Any] | None:
    logged_arguments = dict(action.arguments)
    for sensitive_body in ("body", "content"):
        value = logged_arguments.get(sensitive_body)
        if isinstance(value, str):
            logged_arguments[sensitive_body] = f"<{len(value)} chars>"
    print(
        "[tool] "
        + action.name
        + " "
        + json.dumps(logged_arguments, ensure_ascii=False)
    )

    working.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "type": "tool_call",
                    "name": action.name,
                    "arguments": action.arguments,
                },
                ensure_ascii=False,
            ),
        }
    )

    try:
        result = registry.execute(
            action.name,
            action.arguments,
            context=context,
        )
    except Exception as exc:
        print(f"[tool error] {action.name}: {exc}")
        working.append(
            _tool_error_message(
                action.name,
                action.arguments,
                exc,
            )
        )
        return None

    working.append(
        _tool_result_message(
            action.name,
            action.arguments,
            result,
        )
    )
    return result


def _format_required_result(action: ToolCall, result: dict[str, Any]) -> str | None:
    if action.name == "gmail_send":
        return (
            "E-mail byl odeslán.\n"
            f"Message ID: {result.get('message_id') or 'UNKNOWN'}\n"
            f"Thread ID: {result.get('thread_id') or 'UNKNOWN'}"
        )
    if action.name == "gmail_read":
        messages = result.get("messages")
        if not isinstance(messages, list) or not messages:
            return "Gmail zpráva neobsahuje žádný čitelný obsah."
        rendered: list[str] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            rendered.append(
                f"Od: {item.get('sender') or 'UNKNOWN'}\n"
                f"Komu: {item.get('to') or 'UNKNOWN'}\n"
                f"Předmět: {item.get('subject') or 'UNKNOWN'}\n"
                f"Čas: {item.get('date_local') or item.get('date') or 'UNKNOWN'}\n"
                f"Message ID: {item.get('message_id') or 'UNKNOWN'}\n\n"
                f"{str(item.get('content') or '')[:8000]}"
            )
        return "\n\n---\n\n".join(rendered)
    if action.name == "gmail_create_draft":
        preview = result.get("preview")
        if not isinstance(preview, dict):
            return None
        return (
            "Draft byl vytvořen, ale nebyl odeslán.\n\n"
            f"Komu: {preview.get('to') or 'UNKNOWN'}\n"
            f"CC: {preview.get('cc') or '-'}\n"
            f"Předmět: {preview.get('subject') or ''}\n\n"
            f"{preview.get('body') or ''}\n\n"
            f"Draft ID: {result.get('draft_id') or 'UNKNOWN'}\n"
            "Pro odeslání napiš v nové zprávě přesně „Odešli to“ nebo „Send it“."
        )
    if action.name == "github_apply_write":
        return (
            "GitHub zápis byl proveden.\n"
            f"Repo: {result.get('repo')}\n"
            f"Cesta: {result.get('path')}\n"
            f"Commit: {result.get('commit_sha') or 'UNKNOWN'}"
        )
    if action.name == "github_prepare_write":
        return (
            "GitHub změna je připravená, ale nebyla zapsána.\n\n"
            f"Repo: {result.get('repo')}\n"
            f"Větev: {result.get('branch')}\n"
            f"Cesta: {result.get('path')}\n"
            f"Operace: {result.get('operation')}\n\n"
            f"{result.get('diff') or '(bez textové změny)'}\n\n"
            "Pro provedení napiš v nové zprávě přesně „Proveď zápis“ nebo „Apply it“."
        )
    if action.name == "github_read":
        return (
            f"Repo: {result.get('repo')}\nCesta: {result.get('path')}\n"
            f"Ref: {result.get('ref')}\nBlob SHA: {result.get('github_blob_sha')}\n\n"
            f"{str(result.get('content') or '')[:12_000]}"
        )
    if action.name == "github_list":
        items = result.get("items")
        if not isinstance(items, list):
            return None
        lines = [
            f"- {item.get('type')}: {item.get('path')}"
            for item in items if isinstance(item, dict)
        ]
        return (
            f"GitHub obsah {result.get('repo')}@{result.get('ref')}:{result.get('path') or '/'}\n"
            + ("\n".join(lines) if lines else "(prázdné)")
        )
    if action.name == "github_search":
        items = result.get("results")
        if not isinstance(items, list):
            return None
        lines = []
        for item in items:
            if isinstance(item, dict):
                suffix = f" — {item.get('snippet')}" if item.get("snippet") else ""
                lines.append(f"- {item.get('path')}{suffix}")
        return (
            f"GitHub hledání v {result.get('repo')} pro „{result.get('query')}“:\n"
            + ("\n".join(lines) if lines else "Nebyly nalezeny žádné výsledky.")
        )
    if action.name != "gmail_search" or action.arguments.get("limit") != 1:
        if action.name != "gmail_search":
            return None
        messages = result.get("messages")
        if not isinstance(messages, list):
            return None
        lines = []
        for message in messages:
            if isinstance(message, dict):
                lines.append(
                    f"- {message.get('date_local') or message.get('date')} | "
                    f"{message.get('sender')} | {message.get('subject')} | "
                    f"Message ID: {message.get('message_id')} | "
                    f"Thread ID: {message.get('thread_id')}\n  {message.get('snippet') or ''}"
                )
        return "Výsledky živého Gmail hledání:\n" + ("\n".join(lines) if lines else "Žádné výsledky.")
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        return "V Gmailu nebyla nalezena žádná odpovídající zpráva."
    message = messages[0]
    if not isinstance(message, dict):
        return None
    sender = message.get("sender_name") or message.get("sender") or "UNKNOWN"
    sender_email = message.get("sender_email")
    if sender_email and sender_email not in str(sender):
        sender = f"{sender} <{sender_email}>"
    return (
        f"Předmět: {message.get('subject') or 'UNKNOWN'}\n"
        f"Odesílatel: {sender}\n"
        f"Čas: {message.get('date_local') or message.get('date') or 'UNKNOWN'}\n"
        f"Message ID: {message.get('message_id') or 'UNKNOWN'}\n"
        f"Thread ID: {message.get('thread_id') or 'UNKNOWN'}"
    )


def _explicit_send_confirmation(messages: list[dict[str, str]]) -> bool:
    normalized = " ".join(_tokens(_last_user_text(messages)))
    return normalized in {
        "send",
        "send it",
        "send the draft",
        "yes send it",
        "confirm send",
        "odesli",
        "odesli to",
        "odesli koncept",
        "ano odesli to",
        "potvrzuji odeslani",
        "odeslat koncept",
    }


def _pending_send_call(
    messages: list[dict[str, str]], context: ToolContext
) -> ToolCall | None:
    pending = context.session.pending_actions.get("gmail_send")
    if not pending or not _explicit_send_confirmation(messages):
        return None
    draft_id = pending.get("draft_id")
    if not isinstance(draft_id, str) or not draft_id:
        return None
    return ToolCall(name="gmail_send", arguments={"draft_id": draft_id})


def _explicit_github_confirmation(messages: list[dict[str, str]]) -> bool:
    normalized = " ".join(_tokens(_last_user_text(messages)))
    return normalized in {
        "apply it",
        "apply the write",
        "write it",
        "confirm write",
        "proved zapis",
        "zapis to",
        "ano proved zapis",
        "potvrzuji zapis",
    }


def _pending_github_write_call(
    messages: list[dict[str, str]], context: ToolContext
) -> ToolCall | None:
    pending = context.session.pending_actions.get("github_write")
    if not pending or not _explicit_github_confirmation(messages):
        return None
    return ToolCall(name="github_apply_write", arguments={})


def _is_deferred_promise(answer: str) -> bool:
    normalized = _normalize_text(answer)
    phrases = (
        "please wait",
        "wait a moment",
        "i will check",
        "i will read",
        "i am checking",
        "i am reading",
        "chvili pockej",
        "prosim pockej",
        "zkontroluji",
        "prectu to",
        "prave ctu",
    )
    return any(phrase in normalized for phrase in phrases)


def _unsupported_success_claim(answer: str, successful_tools: list[str]) -> bool:
    normalized = _normalize_text(answer)
    tool_set = set(successful_tools)
    if any(phrase in normalized for phrase in ("email was sent", "e-mail byl odeslan", "mail byl odeslan")):
        return "gmail_send" not in tool_set
    if any(phrase in normalized for phrase in ("wrote to github", "updated github", "zapsal na github", "gitHub zapis".casefold())):
        return "github_apply_write" not in tool_set
    return False


def _external_action_requested(messages: list[dict[str, str]]) -> bool:
    tokens = _tokens(_last_user_text(messages))
    domains = {
        "gmail", "email", "mail", "calendar", "kalendar", "drive", "contacts",
        "kontakty", "github", "browser", "web",
    }
    actions = {
        "check", "read", "search", "send", "create", "open", "refresh", "update",
        "delete", "zkontroluj", "precti", "najdi", "odesli", "vytvor", "otevri",
        "obnov", "uprav", "smaz",
    }
    return _approx_any(tokens, domains) and _approx_any(tokens, actions)


def _guarded_chat(
    provider,
    messages: list[dict[str, str]],
    *,
    purpose: str = "chat",
) -> str:
    """Single choke point: check the complete request before every provider call.

    Applies to the first call, tool-loop continuations and protocol repairs,
    because they all re-enter this helper. Raises ContextBudgetError before the
    network call when required material cannot fit.
    """
    guarded = guard_request(provider, messages, purpose=purpose)
    if guarded.reduced:
        print(f"[context guard] {guarded.summary()}")
    return provider.chat(guarded.messages)


def _is_honest_limitation(answer: str) -> bool:
    normalized = _normalize_text(answer)
    phrases = (
        "cannot", "can't", "do not have", "don't have", "not connected",
        "unavailable", "failed", "error", "nemohu", "nemuzu", "nemam",
        "neni pripojen", "selhalo", "chyba", "auth_required", "network_unavailable",
    )
    return any(phrase in normalized for phrase in phrases)


def run_agent_turn(
    provider,
    messages: list[dict[str, str]],
    *,
    registry: ToolRegistry,
    context: ToolContext,
) -> str:
    explicit_scope = extract_github_read_scope(_last_user_text(messages))

    # For exact file comparisons, isolate the turn from unrelated cached sources
    # and old assistant answers (including previously pasted DOS dumps).
    if explicit_scope is not None:
        if user_forbids_tools(messages):
            return "Rozpor v zadání: nelze načíst GitHub a současně zakázat nástroje."
        current_user = next(
            message for message in reversed(messages)
            if message.get("role") == "user"
        )
        working = [messages[0], current_user]
    else:
        working = list(messages)
        # GitHub reads are persisted as archive sources. Keep them on disk, but
        # do not inject an unrelated DOS snapshot or old DOS dump into this turn.
        if not is_pae_state_request(messages):
            working = [
                msg for msg in working
                if not (
                    msg.get("role") == "system"
                    and msg.get("content", "").startswith("PERSISTENT SOURCE MATERIAL")
                    and "Path: discipline_os/current_state.json" in msg.get("content", "")
                )
                and not (
                    msg.get("role") == "assistant"
                    and is_dos_dump(msg.get("content", ""))
                )
            ]
    working.insert(
        1,
        {
            "role": "system",
            "content": build_tool_protocol_prompt(registry),
        },
    )
    working.insert(
        2,
        {
            "role": "system",
            "content": build_session_facts_prompt(context),
        },
    )
    working.insert(
        3,
        {
            "role": "system",
            "content": (
                "HISTORY RULES\n"
                "When the user asks what they said, named, defined, or meant earlier, "
                "search the persisted user/assistant history supplied in this request. "
                "Use exact earlier user wording when available. Never substitute the "
                "session id, a source path, or a guessed value for a requested literal. "
                "If the literal is absent, say it is unknown."
            ),
        },
    )

    confirmation_call = None if explicit_scope is not None else (
        _pending_send_call(messages, context)
        or _pending_github_write_call(messages, context)
    )
    required = confirmation_call or (
        None if explicit_scope is not None else required_tool_call(messages)
    )
    forbid_optional_tools = required is None and user_forbids_tools(messages)
    tool_calls = 0
    successful_tools: list[str] = []
    last_tool_action: ToolCall | None = None
    last_tool_result: dict[str, Any] | None = None

    # Small local models struggle to emit exact paths with spaces/quotes.
    # Execute at most three literal file reads from the user's request.
    if explicit_scope is not None:
        failed_paths = []
        for path in explicit_scope.paths:
            call = ToolCall(
                name="github_read",
                arguments={
                    "repo": explicit_scope.repo,
                    "path": path,
                    "ref": explicit_scope.ref,
                },
            )
            tool_calls += 1
            if not _append_tool_attempt(
                working, call, registry=registry, context=context
            ):
                failed_paths.append(path)

        if failed_paths:
            return (
                "Nepodařilo se načíst požadované soubory: "
                + "; ".join(failed_paths)
                + ". Porovnání neprovádím a jiné soubory nehledám."
            )
        working.append({
            "role": "system",
            "content": (
                "EXACT FILE TASK: The requested files have already been fetched. "
                "Do not call tools again. Compare ONLY the fetched source contents. "
                "Never print raw GitHub tool JSON or Discipline OS state. "
                "Answer succinctly in the user's language. If context is missing, "
                "state the limitation rather than inventing facts."
            ),
        })


    # If the runtime can deterministically prove a fresh/source read is required,
    # execute it before asking the model. The model does not get to bypass it.
    if required is not None:
        print(
            "[tool guard] required "
            + required.name
            + " "
            + json.dumps(required.arguments, ensure_ascii=False)
        )
        tool_calls += 1
        if required.name in {"gmail_send", "github_apply_write"}:
            pending_key = "gmail_send" if required.name == "gmail_send" else "github_write"
            pending = context.session.pending_actions.get(pending_key)
            if pending is not None:
                pending["approved"] = True
                context.store.save(context.session)
        required_result = _append_tool_attempt(
            working,
            required,
            registry=registry,
            context=context,
        )
        if required_result is not None:
            successful_tools.append(required.name)
            deterministic_answer = _format_required_result(required, required_result)
            if deterministic_answer is not None:
                return deterministic_answer
        elif required.name == "gmail_send":
            return (
                "E-mail nebyl odeslán. Gmail odeslání selhalo; pending draft zůstává "
                "uložený bez aktivního potvrzení, takže je možné chybu opravit a odeslání "
                "znovu výslovně potvrdit."
            )
        elif required.name == "github_apply_write":
            return (
                "GitHub změna nebyla zapsána. Pending preview zůstává uložený bez "
                "aktivního potvrzení; po opravě připojení nebo oprávnění je nutné zápis "
                "znovu výslovně potvrdit."
            )

    forbidden_attempts = 0
    scoped_reprompts = 0
    unrelated_dos_reprompts = 0

    while True:
        repairs = 0

        while True:
            raw = _guarded_chat(provider, working)

            try:
                action = parse_agent_response(raw)
                break
            except AgentProtocolError as exc:
                if repairs >= MAX_PROTOCOL_REPAIRS:
                    raise AgentProtocolError(
                        f"model failed tool protocol after repair: {exc}; "
                        f"last response={raw!r}"
                    ) from exc

                print(f"[protocol repair] {exc}")
                working.append({"role": "assistant", "content": raw})
                working.append(_protocol_error_message(exc))
                repairs += 1

        if isinstance(action, FinalAnswer):
            if not is_pae_state_request(messages) and is_dos_dump(action.content):
                return (
                    "Odpověď modelu byla zablokována: obsahovala nesouvisející "
                    "stav Discipline OS místo odpovědi na otázku."
                )
            if _is_deferred_promise(action.content):
                if last_tool_action is not None and last_tool_result is not None:
                    formatted = _format_required_result(last_tool_action, last_tool_result)
                    if formatted is not None:
                        return formatted
                return (
                    "Neběží žádná skrytá práce na pozadí. Externí akce musí v tomtéž "
                    "tahu vrátit úspěšný TOOL RESULT; jinak ji nemohu označit za provedenou."
                )
            if _unsupported_success_claim(action.content, successful_tools):
                return (
                    "Externí změna nebyla potvrzena úspěšným runtime nástrojem, takže ji "
                    "nemohu označit za provedenou."
                )
            if (
                _external_action_requested(messages)
                and not successful_tools
                and not _is_honest_limitation(action.content)
            ):
                available = ", ".join(spec["name"] for spec in registry.specs())
                return (
                    "Nemám úspěšný výsledek externího nástroje, takže nemohu tvrdit, "
                    "že jsem tuto akci provedl. Dostupné runtime nástroje: " + available
                )
            return action.content

        if explicit_scope is not None:
            # A model-generated path cannot override the exact user-authorized set.
            print("[tool guard] rejected extra tool call outside exact-file task")
            if scoped_reprompts >= 1:
                return (
                    "Model opakovaně žádal další nástroj místo porovnání "
                    "načtených souborů. Žádné další soubory nebyly čteny."
                )
            scoped_reprompts += 1
            working.append({
                "role": "system",
                "content": (
                    "TOOL REQUEST REJECTED: all explicitly requested files are "
                    "already available above. Return the comparison now with no "
                    "further tool calls; do not access Discipline OS."
                ),
            })
            continue

        # Never use the DOS state file as a fallback for unrelated questions.
        if (
            action.name == "github_read"
            and str(action.arguments.get("path", "")).lstrip("/")
            == "discipline_os/current_state.json"
            and not is_pae_state_request(messages)
        ):
            print("[tool guard] rejected unrelated Discipline OS state read")
            if unrelated_dos_reprompts >= 1:
                return (
                    "Model opakovaně požádal o nesouvisející stav PAE. "
                    "Čtení bylo zablokováno; zkus přesný soubor nebo otázku."
                )
            unrelated_dos_reprompts += 1
            working.append({
                "role": "system",
                "content": (
                    "REJECTED: The user did not request PAE Discipline OS state. "
                    "Do NOT fetch discipline_os/current_state.json. "
                    "Answer from the requested sources only, or clearly say "
                    "what information is missing."
                ),
            })
            continue

        if forbid_optional_tools:
            forbidden_attempts += 1
            print(
                "[tool guard] tool rejected; user requested history-only/no-tool answer: "
                + action.name
            )
            if forbidden_attempts > 1:
                raise AgentProtocolError(
                    "model repeatedly requested a tool despite explicit no-tool instruction"
                )

            working.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "type": "tool_call",
                            "name": action.name,
                            "arguments": action.arguments,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            working.append(
                {
                    "role": "system",
                    "content": (
                        "TOOL CALL REJECTED BY RUNTIME\n"
                        "The user explicitly requested a history-only/no-tool answer. "
                        "Do not call GitHub or any tool for this turn. Search the supplied "
                        "persisted chat history and answer from it. If the requested fact "
                        "is not present there, say UNKNOWN."
                    ),
                }
            )
            continue

        if tool_calls >= MAX_TOOL_CALLS:
            raise AgentProtocolError(
                f"tool call limit exceeded ({MAX_TOOL_CALLS})"
            )

        if action.name == "gmail_send":
            return (
                "Draft nebyl odeslán. Nejdřív musí být vytvořen jako pending draft "
                "v této session a uživatel musí v aktuální zprávě výslovně potvrdit "
                "odeslání, například „Odešli to“ nebo „Send it“."
            )
        if action.name == "github_apply_write":
            return (
                "GitHub změna nebyla zapsána. Nejdřív musí existovat pending preview "
                "v této session a uživatel musí v aktuální zprávě výslovně potvrdit "
                "zápis, například „Proveď zápis“ nebo „Apply it“."
            )

        tool_calls += 1
        tool_result = _append_tool_attempt(
            working,
            action,
            registry=registry,
            context=context,
        )
        if tool_result is not None:
            successful_tools.append(action.name)
            last_tool_action = action
            last_tool_result = tool_result
            if action.name in {
                "gmail_create_draft",
                "gmail_send",
                "gmail_read",
                "github_read",
                "github_prepare_write",
                "github_apply_write",
            }:
                formatted = _format_required_result(action, tool_result)
                if formatted is not None:
                    return formatted
