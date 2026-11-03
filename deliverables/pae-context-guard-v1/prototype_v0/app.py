from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_loop import run_agent_turn
from busy_indicator import BusyProvider
from context_guard import ContextBudgetError, resolve_budget
from provider import build_provider
from model_profiles import load_profiles
from runtime_tools import build_tool_registry
from sessions import SessionStore
from state_extractor import extract_state
from tool_core import ToolContext


DEFAULT_SYSTEM_PROMPT = """You are PAE prototype_v0, a small persistent chat runtime.

Treat loaded source material as authoritative context. If the source is a game
log, canon, campaign notes, project notes, or other continuity material,
preserve it faithfully.

This prototype has no hidden memory system. Everything you must remember comes
from the loaded source material, derived persistent state, and persisted chat
history. Never invent a missing fact when the persistent state says UNKNOWN.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PAE prototype_v0")
    parser.add_argument("--profile", help="Select one model drawer at startup")
    parser.add_argument("--profiles-file", type=Path, help="Use a trusted JSON model catalog")
    parser.add_argument("--list-profiles", action="store_true", help="List model drawers without loading credentials or starting a session")
    parser.add_argument(
        "--session",
        default="default",
        help="Session id to create or resume (default: default)",
    )
    parser.add_argument(
        "--load",
        action="append",
        default=[],
        metavar="PATH",
        help="Load a UTF-8 text/log/canon file into persistent context. Repeatable.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Extract structured state from all loaded sources before chat starts.",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Base system prompt for a newly created session.",
    )
    return parser.parse_args()


def build_messages(session) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": session.system_prompt}
    ]

    for source in session.sources:
        metadata_lines = []
        if source.get("source_type") == "github":
            metadata_lines.extend(
                [
                    f"Repository: {source.get('repo', 'UNKNOWN')}",
                    f"Path: {source.get('path', 'UNKNOWN')}",
                    f"Ref: {source.get('ref', 'UNKNOWN')}",
                    f"GitHub blob SHA: {source.get('github_blob_sha', 'UNKNOWN')}",
                ]
            )

        metadata = ""
        if metadata_lines:
            metadata = "\n" + "\n".join(metadata_lines)

        messages.append(
            {
                "role": "system",
                "content": (
                    "PERSISTENT SOURCE MATERIAL\n"
                    f"Name: {source['name']}\n"
                    f"SHA256: {source['sha256']}"
                    f"{metadata}\n"
                    "--- BEGIN SOURCE ---\n"
                    f"{source['content']}\n"
                    "--- END SOURCE ---"
                ),
            }
        )

    if session.derived_state:
        messages.append(
            {
                "role": "system",
                "content": (
                    "PERSISTENT DERIVED STATE\n"
                    "This is a source-grounded extraction. UNKNOWN means unknown; "
                    "do not fill it by guessing.\n"
                    + json.dumps(
                        session.derived_state,
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
            }
        )

    messages.extend(session.messages)
    return messages


def load_file(store: SessionStore, session, raw_path: str) -> None:
    path = Path(raw_path).expanduser().resolve()
    content = path.read_text(encoding="utf-8")
    added = store.add_source(session, name=str(path), content=content)
    print(f"[{'loaded' if added else 'already loaded'}] {path}")


def analyze_sources(provider, store: SessionStore, session) -> None:
    state = extract_state(provider, session.sources)
    store.set_derived_state(session, state)
    print("[state extracted and saved]")


def print_state(session) -> None:
    if not session.derived_state:
        print("[no derived state; run /analyze]")
        return

    print(json.dumps(session.derived_state, ensure_ascii=False, indent=2))


def compact_tool_result(result: dict[str, object]) -> dict[str, object]:
    compact = dict(result)
    content = compact.pop("content", None)
    if isinstance(content, str):
        compact["content_length"] = len(content)
        compact["content_preview"] = content[:160].replace("\n", " ")
    return compact


def report_budget(provider) -> None:
    """Report the resolved request budget once at startup (no contents, no keys)."""
    try:
        budget = resolve_budget(provider)
    except ContextBudgetError as exc:
        print(f"[context] invalid budget: {exc}")
        return
    print(f"[context] {budget.summary()}")


def print_session_debug(session) -> None:
    print(
        f"[session] id={session.session_id} "
        f"history={len(session.messages)} "
        f"sources={len(session.sources)}"
    )


def print_help() -> None:
    print(
        "Commands:\n"
        "  /load PATH   add a text/log/canon file to persistent session context\n"
        "  /github REPO PATH [REF]   compatibility shortcut for github_read\n"
        "  /tools       list registered runtime tools and their schemas\n"
        "  /tool NAME JSON_ARGS   manually execute one registered tool\n"
        "  /analyze     extract canon/characters/current_scene/important_facts\n"
        "  /state       print the persisted structured state\n"
        "  /sources     list loaded persistent sources\n"
        "  /session     show loaded session/history/source counts\n"
        "  /history [N] show the last N persisted chat messages (default: 10)\n"
        "  /save        force-save the session\n"
        "  /help        show commands\n"
        "  /quit        save and exit"
    )


def main() -> None:
    args = parse_args()
    if args.list_profiles:
        print(json.dumps([p.summary() for p in load_profiles(args.profiles_file).values()], ensure_ascii=False, indent=2))
        return
    selected_provider = build_provider(profile_name=args.profile, profiles_path=args.profiles_file)
    store = SessionStore()
    session = store.open_or_create(args.session, system_prompt=args.system)
    tool_registry = build_tool_registry()
    tool_context = ToolContext(store=store, session=session)

    for path in args.load:
        load_file(store, session, path)

    provider = BusyProvider(selected_provider)
    report_budget(provider)

    if args.analyze:
        try:
            analyze_sources(provider, store, session)
        except Exception as exc:
            print(f"[analysis error] {exc}")

    print(
        f"PAE prototype_v0 | session={session.session_id} | "
        f"provider={provider.name} | tools={len(tool_registry.specs())}"
    )
    print_session_debug(session)
    print_help()

    while True:
        try:
            user_text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            store.save(session)
            break

        if not user_text:
            continue

        if user_text == "/quit":
            store.save(session)
            break

        if user_text == "/help":
            print_help()
            continue

        if user_text == "/save":
            store.save(session)
            print("[saved]")
            continue

        if user_text == "/tools":
            print(json.dumps(tool_registry.specs(), ensure_ascii=False, indent=2))
            continue

        if user_text == "/session":
            print_session_debug(session)
            continue

        if user_text == "/history" or user_text.startswith("/history "):
            raw_limit = user_text[len("/history") :].strip()
            try:
                limit = int(raw_limit) if raw_limit else 10
                if limit < 1:
                    raise ValueError
            except ValueError:
                print("[usage] /history [positive integer]")
                continue

            for index, message in enumerate(session.messages[-limit:], start=max(1, len(session.messages) - limit + 1)):
                role = message.get("role", "UNKNOWN")
                content = message.get("content", "")
                print(f"{index:03d} {role}> {content}")
            continue

        if user_text == "/sources":
            if not session.sources:
                print("[no sources loaded]")
            else:
                for source in session.sources:
                    suffix = ""
                    if source.get("github_blob_sha"):
                        suffix = f" | blob {source['github_blob_sha'][:12]}"
                    print(f"- {source['name']} ({source['sha256'][:12]}){suffix}")
            continue

        if user_text == "/state":
            print_state(session)
            continue

        if user_text == "/analyze":
            try:
                analyze_sources(provider, store, session)
                print_state(session)
            except Exception as exc:
                print(f"[analysis error] {exc}")
            continue

        if user_text.startswith("/load "):
            raw_path = user_text[len("/load ") :].strip()
            try:
                load_file(store, session, raw_path)
            except Exception as exc:
                print(f"[load error] {exc}")
            continue

        if user_text.startswith("/github "):
            raw_args = user_text[len("/github ") :].strip().split()
            if len(raw_args) not in {2, 3}:
                print("[usage] /github REPO PATH [REF]")
                continue

            repo, path = raw_args[0], raw_args[1]
            ref = raw_args[2] if len(raw_args) == 3 else "main"

            try:
                result = tool_registry.execute(
                    "github_read",
                    {"repo": repo, "path": path, "ref": ref},
                    context=tool_context,
                )
                print(
                    f"[{result['status']}] {result['source_id']} "
                    f"(blob {str(result['github_blob_sha'])[:12]})"
                )
            except Exception as exc:
                print(f"[github load error] {exc}")
            continue

        if user_text.startswith("/tool "):
            raw = user_text[len("/tool ") :].strip()
            name, separator, raw_json = raw.partition(" ")
            if not separator or not name or not raw_json.strip():
                print("[usage] /tool NAME JSON_ARGS")
                continue

            try:
                arguments = json.loads(raw_json)
                result = tool_registry.execute(
                    name,
                    arguments,
                    context=tool_context,
                )
                print(
                    json.dumps(
                        compact_tool_result(result),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            except Exception as exc:
                print(f"[tool error] {exc}")
            continue

        session.messages.append({"role": "user", "content": user_text})
        store.save(session)

        try:
            reply = run_agent_turn(
                provider,
                build_messages(session),
                registry=tool_registry,
                context=tool_context,
            )
        except Exception as exc:
            print(f"[agent error] {exc}")
            continue

        print(f"\npae> {reply}")
        session.messages.append({"role": "assistant", "content": reply})
        store.save(session)


if __name__ == "__main__":
    main()
