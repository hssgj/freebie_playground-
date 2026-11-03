"""Context Guard: keep every provider request inside the model's budget.

The runtime can assemble more conversation, source material and tool context
than the selected model accepts. This module resolves a request budget, checks
the complete working request, and reduces *only* the working copy of persisted
conversation history.

Two deliberate limits:

1. Token counts here are an ESTIMATE (characters divided by a constant), not a
   tokenizer. See ``ESTIMATE_KIND`` and the README for the consequences.
2. Only old persisted chat exchanges are reducible. Governing instructions,
   persistent sources, derived state, tool permissions, the current user
   request and the active tool exchange are required material. If those alone
   exceed the budget, the request is refused before the network call.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from threading import Lock
from typing import Any, Mapping, Sequence

#: Divisibility constant of the local estimate. Documented, not tokenizer-exact.
CHARS_PER_TOKEN = 4
#: Per-message framing allowance (role markers, delimiters, JSON overhead).
MESSAGE_OVERHEAD_TOKENS = 4

#: Fallback capacity used only when neither a profile nor the launcher
#: environment declares one. It is not a claim about any device: the shipped
#: runtime preamble (tool protocol + session facts + history rules) is roughly
#: 2.1k estimated tokens, so a smaller fallback would refuse every request. An
#: explicit declaration always wins over this fallback.
DEFAULT_CONTEXT_TOKENS = 4096
DEFAULT_OUTPUT_TOKENS = 512
DEFAULT_MARGIN_TOKENS = 384

ENV_CONTEXT = "PAE_CTX_TOKENS"
ENV_OUTPUT = "PAE_MAX_TOKENS"
ENV_MARGIN = "PAE_CTX_MARGIN"

ESTIMATE_KIND = "estimate:ceil(chars/4)+4/message; not tokenizer-exact"

#: Message prefixes produced by app.build_messages / agent_loop. Used only to
#: label diagnostics by category; message contents are never logged.
_CATEGORY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("persistent_source", "PERSISTENT SOURCE MATERIAL"),
    ("derived_state", "PERSISTENT DERIVED STATE"),
    ("tool_protocol", "PAE RUNTIME TOOL PROTOCOL"),
    ("session_facts", "SESSION RUNTIME FACTS"),
    ("history_rules", "HISTORY RULES"),
    ("tool_exchange", "TOOL RESULT"),
    ("tool_exchange", "TOOL ERROR"),
    ("tool_exchange", "TOOL CALL REJECTED BY RUNTIME"),
    ("protocol_repair", "PROTOCOL ERROR"),
    ("exact_file_task", "EXACT FILE TASK"),
    ("tool_guard_rejection", "REJECTED:"),
)


class ContextBudgetError(RuntimeError):
    """A request was refused before the provider call.

    Raised when required material alone cannot fit the resolved input budget.
    The runtime must not truncate the current request, switch models or retry;
    the caller surfaces this message and stops.
    """

    def __init__(
        self,
        message: str,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def estimate_message_tokens(message: Mapping[str, Any]) -> int:
    """Estimated tokens for one message. Never inspects credentials."""
    content = message.get("content")
    if not isinstance(content, str):
        content = "" if content is None else str(content)
    return MESSAGE_OVERHEAD_TOKENS + math.ceil(len(content) / CHARS_PER_TOKEN)


def estimate_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Estimated tokens for a whole request. This is an ESTIMATE."""
    return sum(estimate_message_tokens(message) for message in messages)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _env_int(environ: Mapping[str, str], key: str) -> int | None:
    raw = (environ.get(key) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw, 10)
    except ValueError:
        raise ContextBudgetError(
            f"{key} must be a positive integer (got a non-numeric value); "
            "refusing to guess a context budget"
        ) from None
    if value <= 0:
        raise ContextBudgetError(f"{key} must be a positive integer; refusing to guess a context budget")
    return value


@dataclass(frozen=True)
class RequestBudget:
    """Resolved request budget for one provider.

    ``context_tokens`` is the model's declared capacity and is never altered
    here. ``input_budget`` is what remains after reserving the configured
    maximum output and the safety margin.
    """

    context_tokens: int
    output_reservation: int
    margin_tokens: int
    capacity_origin: str
    output_origin: str
    margin_origin: str

    @property
    def input_budget(self) -> int:
        return self.context_tokens - self.output_reservation - self.margin_tokens

    @property
    def origin(self) -> str:
        return f"capacity={self.capacity_origin}; output={self.output_origin}; margin={self.margin_origin}"

    def diagnostics(self) -> dict[str, Any]:
        return {
            "capacity_tokens": self.context_tokens,
            "output_reservation_tokens": self.output_reservation,
            "margin_tokens": self.margin_tokens,
            "input_budget_tokens": self.input_budget,
            "capacity_origin": self.capacity_origin,
            "output_origin": self.output_origin,
            "margin_origin": self.margin_origin,
            "estimate": ESTIMATE_KIND,
        }

    def summary(self) -> str:
        return (
            f"capacity={self.context_tokens} "
            f"output={self.output_reservation} "
            f"margin={self.margin_tokens} "
            f"input_budget={self.input_budget} "
            f"({self.origin}; tokens are an estimate)"
        )


def resolve_budget(
    provider: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> RequestBudget:
    """Resolve the request budget.

    Precedence, documented and deliberate:

    1. A selected profile's declaration (attached to the provider as
       ``context_tokens`` / ``max_output_tokens``) wins for capacity and output
       reservation. A profile's declared capacity is never overridden by the
       environment, so it cannot be changed silently.
    2. The existing launcher environment (``PAE_CTX_TOKENS``, ``PAE_MAX_TOKENS``)
       applies on the legacy no-profile path, where launchers export it.
    3. Built-in fallbacks: 4096 context and 512 output. The fallback is a guard
       default for an undeclared capacity, not a device claim.
    4. ``PAE_CTX_MARGIN`` always overrides the margin. The margin is a safety
       knob, not a model capacity, so tightening it from the environment is
       allowed even with a profile selected.
    """
    environ = os.environ if environ is None else environ

    capacity = _positive_int(getattr(provider, "context_tokens", 0))
    if capacity is not None:
        capacity_origin = f"profile:{getattr(provider, 'profile_id', 'unknown')}"
    else:
        capacity = _env_int(environ, ENV_CONTEXT)
        capacity_origin = f"env:{ENV_CONTEXT}" if capacity else "default"
    capacity = capacity or DEFAULT_CONTEXT_TOKENS

    output = _positive_int(getattr(provider, "max_output_tokens", 0))
    if output is not None:
        output_origin = f"profile:{getattr(provider, 'profile_id', 'unknown')}"
    else:
        # The payload's actual max_tokens for the legacy path.
        output = _positive_int(getattr(provider, "max_tokens", None))
        if output is not None:
            output_origin = "provider:max_tokens"
        else:
            output = _env_int(environ, ENV_OUTPUT)
            output_origin = f"env:{ENV_OUTPUT}" if output else "default"
    output = output or DEFAULT_OUTPUT_TOKENS

    margin = _env_int(environ, ENV_MARGIN)
    margin_origin = f"env:{ENV_MARGIN}" if margin else "default"
    margin = margin or DEFAULT_MARGIN_TOKENS

    budget = RequestBudget(
        context_tokens=capacity,
        output_reservation=output,
        margin_tokens=margin,
        capacity_origin=capacity_origin,
        output_origin=output_origin,
        margin_origin=margin_origin,
    )
    if budget.input_budget <= 0:
        raise ContextBudgetError(
            "Reserved output and safety margin leave no room for input: "
            f"capacity={capacity} output={output} margin={margin}. "
            "Lower the output reservation or the margin; the declared capacity "
            "was not changed.",
            diagnostics=budget.diagnostics(),
        )
    return budget


def _current_turn_start(messages: Sequence[Mapping[str, Any]]) -> int:
    """Index of the latest user message; everything from here is protected."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return len(messages)


def _history_start(messages: Sequence[Mapping[str, Any]], turn_start: int) -> int:
    """First persisted chat message (user/assistant) before the current turn."""
    for index in range(turn_start):
        if messages[index].get("role") in {"user", "assistant"}:
            return index
    return turn_start


def _exchanges(
    messages: Sequence[Mapping[str, Any]], start: int, stop: int
) -> list[tuple[int, int]]:
    """Split droppable history into whole exchanges, oldest first.

    A boundary is a user message, so an exchange always ends at the next user
    message. Dropping whole exchanges cannot orphan an assistant tool call or
    its tool result.
    """
    if stop <= start:
        return []
    bounds = [start]
    for index in range(start + 1, stop):
        if messages[index].get("role") == "user":
            bounds.append(index)
    bounds.append(stop)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _category(message: Mapping[str, Any], index: int) -> str:
    if index == 0 and message.get("role") == "system":
        return "governing_system_prompt"
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    for name, prefix in _CATEGORY_PREFIXES:
        if text.startswith(prefix):
            return name
    if message.get("role") == "user":
        return "current_request"
    if message.get("role") == "assistant":
        return "history_assistant"
    return "runtime_system"


def classify_required(messages: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Estimate per category, without exposing message contents."""
    categories: dict[str, dict[str, int]] = {}
    for index, message in enumerate(messages):
        name = _category(message, index)
        bucket = categories.setdefault(name, {"messages": 0, "tokens": 0})
        bucket["messages"] += 1
        bucket["tokens"] += estimate_message_tokens(message)
    return categories


def _breakdown_text(categories: Mapping[str, Mapping[str, int]]) -> str:
    parts = [
        f"{name} {bucket['tokens']} tokens ({bucket['messages']} message(s))"
        for name, bucket in sorted(categories.items(), key=lambda item: -item[1]["tokens"])
    ]
    return ", ".join(parts)


def _oversized_error(
    *,
    purpose: str,
    budget: RequestBudget,
    messages: Sequence[Mapping[str, Any]],
    estimated_input: int,
    omitted_exchanges: int,
) -> ContextBudgetError:
    categories = classify_required(messages)
    diagnostics = {
        **budget.diagnostics(),
        "purpose": purpose,
        "estimated_input_tokens": estimated_input,
        "omitted_exchanges": omitted_exchanges,
        "required_categories": categories,
        "provider_called": False,
    }

    # Name the dominant blocker instead of a generic "narrow the request".
    overhead = sum(
        bucket["tokens"]
        for name, bucket in categories.items()
        if name not in {"persistent_source", "current_request"}
    )
    hints: list[str] = []
    if overhead > budget.input_budget:
        hints.append(
            "the fixed runtime preamble (governing prompt, tool protocol, session "
            f"facts, history rules) alone needs ~{overhead} tokens, more than the whole "
            f"input budget of {budget.input_budget}. This declared capacity cannot host "
            "the registered tool protocol: raise the profile/model capacity to what the "
            "server is really started with, or reduce the registered tool set"
        )
    if categories.get("persistent_source", {}).get("tokens"):
        hints.append(
            "persistent sources are required material and were not truncated; unload or "
            "split them in the session, or start a session without them"
        )
    if categories.get("current_request", {}).get("tokens"):
        hints.append("shorten the current message")
    if not hints:
        hints.append("use a model whose declared capacity fits this request")

    return ContextBudgetError(
        "Request refused before the provider call: required material does not fit.\n"
        f"Purpose: {purpose}\n"
        f"Estimated input: {estimated_input} tokens "
        f"(available input budget {budget.input_budget} = capacity {budget.context_tokens} "
        f"- output {budget.output_reservation} - margin {budget.margin_tokens}).\n"
        f"Required material: {_breakdown_text(categories)}.\n"
        f"Persisted history was already reduced by {omitted_exchanges} exchange(s) "
        "and cannot be reduced further.\n"
        "Nothing was sent, no model was switched and no retry was attempted. "
        "What needs narrowing: " + "; ".join(hints) + ". "
        "Token counts above are an estimate, not tokenizer-exact.",
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class GuardedRequest:
    """The exact message list to send, plus what was reduced to get there."""

    messages: list[dict[str, str]]
    budget: RequestBudget
    purpose: str
    estimated_input: int
    omitted_exchanges: int
    omitted_messages: int

    @property
    def reduced(self) -> bool:
        return self.omitted_exchanges > 0

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.budget.diagnostics(),
            "purpose": self.purpose,
            "estimated_input_tokens": self.estimated_input,
            "omitted_exchanges": self.omitted_exchanges,
            "omitted_messages": self.omitted_messages,
            "provider_called": True,
        }

    def summary(self) -> str:
        return (
            f"{self.purpose}: estimated_input={self.estimated_input} "
            f"<= input_budget={self.budget.input_budget} "
            f"omitted_exchanges={self.omitted_exchanges} "
            f"omitted_messages={self.omitted_messages} "
            f"(capacity={self.budget.context_tokens} "
            f"output={self.budget.output_reservation} "
            f"margin={self.budget.margin_tokens}; estimate, not tokenizer-exact)"
        )


_LAST_LOCK = Lock()
_LAST_DIAGNOSTICS: dict[str, Any] | None = None


def record_observation(diagnostics: dict[str, Any]) -> None:
    """Remember the most recent budget observation for read-only diagnostics."""
    global _LAST_DIAGNOSTICS
    with _LAST_LOCK:
        _LAST_DIAGNOSTICS = diagnostics


def last_diagnostics() -> dict[str, Any] | None:
    """Most recent budget observation, or None before the first request."""
    with _LAST_LOCK:
        return dict(_LAST_DIAGNOSTICS) if _LAST_DIAGNOSTICS is not None else None


def guard_request(
    provider: object,
    messages: Sequence[Mapping[str, Any]],
    *,
    purpose: str = "chat",
    environ: Mapping[str, str] | None = None,
) -> GuardedRequest:
    """Check the complete working request, reducing only old chat history.

    Returns a guarded copy suitable for ``provider.chat``. The input sequence is
    never mutated: persisted sessions, sources, derived state and pending
    actions stay untouched on disk and in memory.
    """
    budget = resolve_budget(provider, environ=environ)
    working = [dict(message) for message in messages]
    estimated = estimate_tokens(working)

    if estimated <= budget.input_budget:
        guarded = GuardedRequest(
            messages=working,
            budget=budget,
            purpose=purpose,
            estimated_input=estimated,
            omitted_exchanges=0,
            omitted_messages=0,
        )
        record_observation(guarded.diagnostics())
        return guarded

    omitted_exchanges = 0
    omitted_messages = 0
    while estimated > budget.input_budget:
        turn_start = _current_turn_start(working)
        start = _history_start(working, turn_start)
        exchanges = _exchanges(working, start, turn_start)
        if not exchanges:
            raise _oversized_error(
                purpose=purpose,
                budget=budget,
                messages=working,
                estimated_input=estimated,
                omitted_exchanges=omitted_exchanges,
            )
        drop_start, drop_stop = exchanges[0]
        omitted_messages += drop_stop - drop_start
        del working[drop_start:drop_stop]
        omitted_exchanges += 1
        estimated = estimate_tokens(working)

    guarded = GuardedRequest(
        messages=working,
        budget=budget,
        purpose=purpose,
        estimated_input=estimated,
        omitted_exchanges=omitted_exchanges,
        omitted_messages=omitted_messages,
    )
    record_observation(guarded.diagnostics())
    return guarded
