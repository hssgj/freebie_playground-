"""Offline tests for the Context Guard.

Everything here is deterministic: scripted in-process providers and a loopback
HTTP fixture standing in for a model server. No Qwen, Groq or Gemini endpoint is
contacted, and no real tokenizer is used (counts are estimates).

Run from ``prototype_v0``:

    python -m pytest test_context_guard.py -q

The tests use only the standard library plus pytest's runner; they do not
require a network connection.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from agent_loop import run_agent_turn
from context_guard import (
    CHARS_PER_TOKEN,
    ContextBudgetError,
    estimate_tokens,
    guard_request,
    last_diagnostics,
    resolve_budget,
)
from provider import Provider
from runtime_tools import build_tool_registry
from sessions import SessionStore
from state_extractor import extract_state
from tool_core import Tool, ToolContext, ToolRegistry

HERE = Path(__file__).resolve().parent
PROTOTYPE = HERE
ROOT = HERE.parent

#: Runtime preamble is roughly 2.2k estimated tokens with the shipped 9-tool
#: registry, so capacities in these tests are chosen around that figure.
ROOMY = 3000
TIGHT = 1500


def make_exchange(index: int) -> list[dict[str, str]]:
    """One persisted user/assistant exchange with unique, greppable markers."""
    filler = "abcdefghij" * 12
    return [
        {"role": "user", "content": f"question-{index:02d} {filler}"},
        {"role": "assistant", "content": f"answer-{index:02d} {filler}"},
    ]


def build_history(exchanges: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for index in range(exchanges):
        messages.extend(make_exchange(index))
    return messages


class ScriptedProvider(Provider):
    """Deterministic provider that records every request it receives."""

    name = "scripted"

    def __init__(
        self,
        *responses: str,
        context_tokens: int = 0,
        max_output_tokens: int = 0,
        profile_id: str = "scripted",
    ) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []
        self.profile_id = profile_id
        if context_tokens:
            self.context_tokens = context_tokens
        if max_output_tokens:
            self.max_output_tokens = max_output_tokens

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append([dict(message) for message in messages])
        if not self._responses:
            return "scripted final answer"
        return self._responses.pop(0)


def tool_registry_with_huge_result(size: int) -> ToolRegistry:
    registry = ToolRegistry()

    def executor(context: ToolContext, arguments: dict[str, object]) -> dict[str, object]:
        return {"content": "z" * size, "status": "loaded"}

    registry.register(
        Tool(
            name="probe",
            description="Return a fixed-size payload for budget tests.",
            argument_schema={
                "type": "object",
                "properties": {"size": {"type": "integer"}},
                "required": ["size"],
            },
            executor=executor,
        )
    )
    return registry


class ContextGuardTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate from a developer's launcher environment.
        patcher = patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("PAE_CTX_TOKENS", "PAE_MAX_TOKENS", "PAE_CTX_MARGIN"):
            os.environ.pop(name, None)
        self._temp = None

    def store(self) -> tuple[SessionStore, object, ToolContext]:
        import tempfile

        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        store = SessionStore(Path(self._temp.name))
        session = store.open_or_create("guard", system_prompt="GOVERNING PROMPT for tests")
        return store, session, ToolContext(store=store, session=session)


class BudgetResolutionTests(ContextGuardTestCase):
    def test_small_request_passes_unchanged(self) -> None:
        provider = ScriptedProvider("ok", context_tokens=8192, max_output_tokens=512)
        messages = [
            {"role": "system", "content": "governing instructions"},
            {"role": "user", "content": "hello"},
        ]

        guarded = guard_request(provider, messages, environ={})

        self.assertFalse(guarded.reduced)
        self.assertEqual(guarded.messages, messages)
        self.assertEqual(guarded.omitted_exchanges, 0)
        self.assertEqual(guarded.estimated_input, estimate_tokens(messages))
        self.assertLessEqual(guarded.estimated_input, guarded.budget.input_budget)

    def test_profile_declaration_wins_over_launcher_environment(self) -> None:
        provider = ScriptedProvider(
            "ok", context_tokens=2048, max_output_tokens=512, profile_id="local-qwen"
        )
        budget = resolve_budget(
            provider, environ={"PAE_CTX_TOKENS": "9999", "PAE_MAX_TOKENS": "4096"}
        )

        # A declared capacity is never silently replaced by the environment.
        self.assertEqual(budget.context_tokens, 2048)
        self.assertEqual(budget.output_reservation, 512)
        self.assertEqual(budget.capacity_origin, "profile:local-qwen")
        self.assertEqual(budget.input_budget, 2048 - 512 - 384)

    def test_launcher_environment_used_without_profile(self) -> None:
        budget = resolve_budget(
            object(),
            environ={"PAE_CTX_TOKENS": "6000", "PAE_MAX_TOKENS": "700", "PAE_CTX_MARGIN": "128"},
        )

        self.assertEqual(budget.context_tokens, 6000)
        self.assertEqual(budget.output_reservation, 700)
        self.assertEqual(budget.margin_tokens, 128)
        self.assertEqual(budget.capacity_origin, "env:PAE_CTX_TOKENS")
        self.assertEqual(budget.input_budget, 6000 - 700 - 128)

    def test_provider_max_tokens_is_reserved_on_legacy_path(self) -> None:
        class Legacy(object):
            name = "legacy"
            max_tokens = 900

        budget = resolve_budget(Legacy(), environ={"PAE_CTX_TOKENS": "4096"})

        # Reserve what the payload will actually ask for.
        self.assertEqual(budget.output_reservation, 900)
        self.assertEqual(budget.output_origin, "provider:max_tokens")

    def test_margin_is_a_safety_knob_not_capacity(self) -> None:
        provider = ScriptedProvider("ok", context_tokens=4096, max_output_tokens=512)
        budget = resolve_budget(provider, environ={"PAE_CTX_MARGIN": "64"})

        self.assertEqual(budget.context_tokens, 4096)
        self.assertEqual(budget.margin_tokens, 64)
        self.assertEqual(budget.margin_origin, "env:PAE_CTX_MARGIN")
        self.assertEqual(budget.input_budget, 4096 - 512 - 64)

    def test_different_capacities_and_output_reservations_change_the_budget(self) -> None:
        cases = [
            (4096, 512, 384),
            (8192, 1024, 384),
            (8192, 4096, 384),
            (2048, 256, 64),
        ]
        budgets = []
        for capacity, output, margin in cases:
            provider = ScriptedProvider(
                "ok", context_tokens=capacity, max_output_tokens=output
            )
            budget = resolve_budget(provider, environ={"PAE_CTX_MARGIN": str(margin)})
            self.assertEqual(budget.context_tokens, capacity)
            self.assertEqual(
                budget.input_budget, capacity - output - margin, (capacity, output)
            )
            budgets.append(budget.input_budget)

        # More reserved output means less room for input at equal capacity.
        self.assertGreater(budgets[1], budgets[2])
        # More declared capacity means more room for input at equal reservation.
        self.assertGreater(budgets[1], budgets[0])

    def test_same_request_fits_bigger_capacity_but_not_smaller(self) -> None:
        messages = [
            {"role": "system", "content": "governing instructions"},
            {"role": "user", "content": "x" * 4000},
        ]
        roomy = ScriptedProvider("ok", context_tokens=8192, max_output_tokens=512)
        small = ScriptedProvider("ok", context_tokens=1200, max_output_tokens=512)

        self.assertFalse(guard_request(roomy, messages, environ={}).reduced)
        with self.assertRaises(ContextBudgetError):
            guard_request(small, messages, environ={})

    def test_invalid_or_impossible_budget_fails_loudly(self) -> None:
        with self.assertRaises(ContextBudgetError) as caught:
            resolve_budget(object(), environ={"PAE_CTX_TOKENS": "lots"})
        self.assertIn("PAE_CTX_TOKENS", str(caught.exception))

        provider = ScriptedProvider("ok", context_tokens=1000, max_output_tokens=700)
        with self.assertRaises(ContextBudgetError) as caught:
            resolve_budget(provider, environ={"PAE_CTX_MARGIN": "400"})
        self.assertIn("input", str(caught.exception))

    def test_estimate_is_labelled_and_derived_from_characters(self) -> None:
        message = {"role": "user", "content": "a" * (CHARS_PER_TOKEN * 10)}
        budget = resolve_budget(ScriptedProvider("ok", context_tokens=4096),
                                environ={})
        diagnostics = budget.diagnostics()

        self.assertIn("estimate", diagnostics["estimate"].lower())
        self.assertEqual(estimate_tokens([message]), 10 + 4)


class HistoryReductionTests(ContextGuardTestCase):
    def test_long_history_reduced_while_current_request_survives(self) -> None:
        provider = ScriptedProvider(
            "final answer", context_tokens=ROOMY, max_output_tokens=100
        )
        registry = build_tool_registry()
        store, session, context = self.store()
        session.sources.append(
            {
                "name": "canon.md",
                "sha256": "a" * 64,
                "content": "CANON-ONE the archive door is locked.",
            }
        )
        session.messages = build_history(12)
        store.save(session)

        from app import build_messages

        messages = build_messages(session)
        messages.append({"role": "user", "content": "CURRENT-REQUEST please summarize"})
        messages = messages[:-1] + [messages[-1]]

        reply = run_agent_turn(provider, messages, registry=registry, context=context)

        self.assertEqual(reply, "final answer")
        self.assertEqual(len(provider.calls), 1)
        sent = provider.calls[0]

        # The current request, governing prompt and tool protocol survive.
        self.assertEqual(sent[-1]["content"], "CURRENT-REQUEST please summarize")
        self.assertIn("GOVERNING PROMPT", sent[0]["content"])
        self.assertTrue(
            any("PAE RUNTIME TOOL PROTOCOL" in message["content"] for message in sent)
        )
        # Persistent sources are required material, not history.
        self.assertTrue(any("CANON-ONE" in message["content"] for message in sent))

        # Oldest exchanges are gone; the newest persisted exchange remains.
        joined = "\n".join(message["content"] for message in sent)
        self.assertNotIn("question-00", joined)
        self.assertNotIn("answer-00", joined)
        self.assertIn("question-11", joined)
        self.assertLessEqual(estimate_tokens(sent), 2516)

    def test_reduction_never_mutates_the_caller_list(self) -> None:
        provider = ScriptedProvider("ok", context_tokens=ROOMY, max_output_tokens=100)
        messages = [
            {"role": "system", "content": "governing instructions"},
            {"role": "system", "content": "P" * 6000},
        ]
        messages.extend(build_history(20))
        messages.append({"role": "user", "content": "CURRENT"})
        snapshot = [dict(message) for message in messages]

        guarded = guard_request(provider, messages, environ={})

        self.assertTrue(guarded.reduced)
        self.assertEqual(messages, snapshot)
        self.assertNotEqual(guarded.messages, messages)
        # Mutating the working copy must not reach the caller's messages.
        guarded.messages[0]["content"] = "changed"
        self.assertEqual(messages[0]["content"], "governing instructions")

    def test_active_tool_exchange_is_preserved(self) -> None:
        provider = ScriptedProvider("ok", context_tokens=ROOMY, max_output_tokens=100)
        messages = [
            {"role": "system", "content": "governing instructions"},
            {"role": "system", "content": "P" * 6000},
        ]
        messages.extend(build_history(20))
        messages.append({"role": "user", "content": "CURRENT"})
        messages.append(
            {
                "role": "assistant",
                "content": '{"type":"tool_call","name":"github_read","arguments":{}}',
            }
        )
        messages.append(
            {
                "role": "system",
                "content": "TOOL RESULT\nACTIVE-TOOL-PAYLOAD " + "q" * 1200,
            }
        )

        guarded = guard_request(provider, messages, environ={})

        self.assertTrue(guarded.reduced)
        self.assertEqual(guarded.messages[-3]["content"], "CURRENT")
        self.assertIn("tool_call", guarded.messages[-2]["content"])
        self.assertTrue(guarded.messages[-1]["content"].startswith("TOOL RESULT"))
        self.assertIn("ACTIVE-TOOL-PAYLOAD", guarded.messages[-1]["content"])
        self.assertNotIn(
            "TOOL RESULT",
            "\n".join(message["content"] for message in guarded.messages[:-1]),
        )

    def test_history_with_tool_exchange_is_dropped_atomically(self) -> None:
        provider = ScriptedProvider("ok", context_tokens=ROOMY, max_output_tokens=100)
        payload = "w" * 2400
        messages = [
            {"role": "system", "content": "governing instructions"},
            {"role": "system", "content": "P" * 6000},
        ]
        for index in (1, 2):
            messages.extend(
                [
                    {"role": "user", "content": f"OLD-QUESTION-{index}"},
                    {
                        "role": "assistant",
                        "content": '{"type":"tool_call","name":"probe","arguments":{}}',
                    },
                    {"role": "system", "content": f"TOOL RESULT\nOLD-TOOL-PAYLOAD-{index}\n{payload}"},
                    {"role": "assistant", "content": f"OLD-ANSWER-{index}"},
                ]
            )
        messages.append({"role": "user", "content": "CURRENT"})

        guarded = guard_request(provider, messages, environ={})

        self.assertTrue(guarded.reduced)
        joined = "\n".join(message["content"] for message in guarded.messages)
        # The oldest whole exchange is gone, including its tool call and result.
        self.assertNotIn("OLD-QUESTION-1", joined)
        self.assertNotIn("OLD-TOOL-PAYLOAD-1", joined)
        self.assertNotIn("OLD-ANSWER-1", joined)
        # A later exchange may survive; if it does, its tool result keeps its call.
        if "OLD-TOOL-PAYLOAD-2" in joined:
            contents = [message["content"] for message in guarded.messages]
            result_index = next(
                index for index, text in enumerate(contents) if "OLD-TOOL-PAYLOAD-2" in text
            )
            self.assertIn("tool_call", contents[result_index - 1])
        self.assertEqual(guarded.messages[-1]["content"], "CURRENT")

    def test_diagnostics_report_counts_without_contents(self) -> None:
        provider = ScriptedProvider("ok", context_tokens=ROOMY, max_output_tokens=100)
        messages = [{"role": "system", "content": "SECRET-GOVERNING-TEXT"}]
        messages.append({"role": "system", "content": "P" * 6000})
        messages.extend(build_history(20))
        messages.append({"role": "user", "content": "CURRENT-SECRET"})

        guarded = guard_request(provider, messages, environ={})
        diagnostics = guarded.diagnostics()

        for key in (
            "capacity_tokens",
            "output_reservation_tokens",
            "margin_tokens",
            "input_budget_tokens",
            "estimated_input_tokens",
            "omitted_exchanges",
            "omitted_messages",
        ):
            self.assertIn(key, diagnostics)
        self.assertGreater(diagnostics["omitted_exchanges"], 0)
        rendered = json.dumps(diagnostics)
        self.assertNotIn("SECRET-GOVERNING-TEXT", rendered)
        self.assertNotIn("CURRENT-SECRET", rendered)
        self.assertEqual(last_diagnostics()["omitted_exchanges"],
                         diagnostics["omitted_exchanges"])


class RequiredMaterialTests(ContextGuardTestCase):
    def test_required_sources_too_large_causes_zero_provider_calls(self) -> None:
        provider = ScriptedProvider("ok", context_tokens=2000, max_output_tokens=200)
        messages = [
            {"role": "system", "content": "governing instructions"},
            {
                "role": "system",
                "content": "PERSISTENT SOURCE MATERIAL\nName: big.txt\n"
                + "z" * 20000,
            },
            {"role": "user", "content": "CURRENT"},
        ]

        with self.assertRaises(ContextBudgetError) as caught:
            guard_request(provider, messages, purpose="chat", environ={})

        self.assertEqual(provider.calls, [])
        error = caught.exception
        text = str(error)
        self.assertIn("estimate", text.lower())
        self.assertIn("persistent_source", text)
        self.assertIn("current_request", text)  # category name only, never the content
        self.assertNotIn("z" * 100, text)
        self.assertFalse(error.diagnostics["provider_called"])
        self.assertEqual(error.diagnostics["omitted_exchanges"], 0)

    def test_preamble_alone_oversized_names_the_real_blocker(self) -> None:
        provider = ScriptedProvider("ok", context_tokens=1200, max_output_tokens=200)
        registry = build_tool_registry()
        from agent_loop import build_tool_protocol_prompt

        messages = [
            {"role": "system", "content": "governing instructions"},
            {"role": "system", "content": build_tool_protocol_prompt(registry)},
            {"role": "user", "content": "hi"},
        ]

        with self.assertRaises(ContextBudgetError) as caught:
            guard_request(provider, messages, environ={})

        self.assertEqual(provider.calls, [])
        self.assertIn("tool protocol", str(caught.exception).lower())
        self.assertIn("declared capacity", str(caught.exception).lower())

    def test_state_extraction_rejects_oversized_source_bundle_clearly(self) -> None:
        provider = ScriptedProvider('{"canon": []}', context_tokens=2000,
                                   max_output_tokens=200)
        sources = [
            {"name": "huge.md", "sha256": "b" * 64, "content": "q" * 20000},
        ]

        with self.assertRaises(ContextBudgetError) as caught:
            extract_state(provider, sources)

        self.assertEqual(provider.calls, [])
        text = str(caught.exception)
        self.assertIn("state extraction", text)
        self.assertIn("Nothing was sent", text)
        self.assertIn("estimate", text.lower())

    def test_state_extraction_still_works_when_bundle_fits(self) -> None:
        provider = ScriptedProvider(
            json.dumps(
                {
                    "canon": ["Door locked"],
                    "characters": [],
                    "current_scene": {"location": "UNKNOWN"},
                    "important_facts": ["Door locked"],
                }
            ),
            context_tokens=8192,
            max_output_tokens=256,
        )
        sources = [{"name": "s.md", "sha256": "c" * 64, "content": "Door is locked."}]

        state = extract_state(provider, sources)

        self.assertEqual(state["canon"], ["Door locked"])
        self.assertEqual(len(provider.calls), 1)


class AgentLoopIntegrationTests(ContextGuardTestCase):
    def test_tool_loop_continuation_is_checked_after_tool_result(self) -> None:
        registry = tool_registry_with_huge_result(200_000)
        provider = ScriptedProvider(
            json.dumps(
                {
                    "type": "tool_call",
                    "name": "probe",
                    "arguments": {"size": 200_000},
                }
            ),
            "unreachable",
            context_tokens=ROOMY,
            max_output_tokens=100,
        )
        store, session, context = self.store()
        session.messages = build_history(6)
        store.save(session)

        from app import build_messages

        messages = build_messages(session)
        messages.append({"role": "user", "content": "CURRENT-REQUEST"})

        with self.assertRaises(ContextBudgetError) as caught:
            run_agent_turn(provider, messages, registry=registry, context=context)

        # The first (small) request ran; the oversized continuation did not.
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(caught.exception.diagnostics["purpose"], "chat")
        self.assertGreater(caught.exception.diagnostics["omitted_exchanges"], 0)
        self.assertIn("tool_exchange", caught.exception.diagnostics["required_categories"])

    def test_protocol_repair_also_passes_through_the_guard(self) -> None:
        import agent_loop

        registry = build_tool_registry()
        provider = ScriptedProvider(
            '{"type":"bogus"}', "repaired final", context_tokens=8192,
            max_output_tokens=256,
        )
        store, session, context = self.store()
        messages = [
            {"role": "system", "content": "governing instructions"},
            {"role": "user", "content": "hello"},
        ]

        purposes: list[str] = []
        real = agent_loop.guard_request

        def spy(provider_, messages_, **kwargs):
            purposes.append(kwargs.get("purpose"))
            return real(provider_, messages_, **kwargs)

        with patch.object(agent_loop, "guard_request", spy):
            reply = run_agent_turn(provider, messages, registry=registry, context=context)

        self.assertEqual(reply, "repaired final")
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(purposes, ["chat", "chat"])

    def test_tool_loop_checks_again_on_second_continuation(self) -> None:
        import agent_loop

        registry = tool_registry_with_huge_result(16)
        provider = ScriptedProvider(
            json.dumps({"type": "tool_call", "name": "probe", "arguments": {"size": 16}}),
            json.dumps({"type": "tool_call", "name": "probe", "arguments": {"size": 16}}),
            "final after tools",
            context_tokens=8192,
            max_output_tokens=256,
        )
        store, session, context = self.store()
        messages = [
            {"role": "system", "content": "governing instructions"},
            {"role": "user", "content": "hello"},
        ]

        real = agent_loop.guard_request
        purposes: list[str] = []

        def spy(provider_, messages_, **kwargs):
            purposes.append(kwargs.get("purpose"))
            return real(provider_, messages_, **kwargs)

        with patch.object(agent_loop, "guard_request", spy):
            reply = run_agent_turn(provider, messages, registry=registry, context=context)

        self.assertEqual(reply, "final after tools")
        self.assertEqual(len(purposes), 3)
        self.assertEqual(purposes, ["chat"] * 3)

    def test_persisted_session_file_unchanged_by_reduction(self) -> None:
        provider = ScriptedProvider("answer", context_tokens=ROOMY, max_output_tokens=100)
        registry = build_tool_registry()
        store, session, context = self.store()
        session.sources = [
            {"name": "canon.md", "sha256": "d" * 64, "content": "Canon text."},
        ]
        session.messages = build_history(10)
        session.derived_state = {"canon": ["Canon text"]}
        session.pending_actions = {"gmail_send": {"draft_id": "draft-1"}}
        store.save(session)
        before = store.path_for("guard").read_bytes()

        from app import build_messages

        messages = build_messages(session)
        messages.append({"role": "user", "content": "CURRENT"})
        run_agent_turn(provider, messages, registry=registry, context=context)

        after = store.path_for("guard").read_bytes()
        self.assertEqual(before, after)
        reloaded = store.load("guard")
        self.assertEqual(len(reloaded.messages), 20)
        self.assertEqual(reloaded.derived_state, {"canon": ["Canon text"]})
        self.assertEqual(reloaded.pending_actions["gmail_send"]["draft_id"], "draft-1")
        self.assertEqual(reloaded.sources[0]["content"], "Canon text.")


class HttpFixture:
    """Loopback stand-in for an OpenAI-compatible endpoint."""

    def __init__(self, reply: str = "[fixture] ok") -> None:
        self.requests: list[dict[str, object]] = []
        self.reply = reply
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture.requests.append(payload)
                data = json.dumps(
                    {
                        "choices": [{"message": {"content": fixture.reply},
                                     "finish_reason": "stop"}],
                        "model": "fixture-server",
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self.server.server_port

    def __enter__(self) -> "HttpFixture":
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


class ProcessIntegrationTests(unittest.TestCase):
    """Real CLI and mobile-server processes against a loopback HTTP fixture."""

    def setUp(self) -> None:
        import shutil
        import tempfile

        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.copy = Path(self._temp.name) / "PAE"
        (self.copy / "prototype_v0").mkdir(parents=True)
        for source in (PROTOTYPE).glob("*.py"):
            if source.name.startswith(("test_", "install_", "apply_")):
                continue
            shutil.copy2(source, self.copy / "prototype_v0" / source.name)
        shutil.copytree(PROTOTYPE / "mobile_web", self.copy / "prototype_v0" / "mobile_web")
        shutil.copytree(ROOT / "config", self.copy / "config")
        self.env = {
            key: os.environ[key]
            for key in ("PATH", "LANG", "LC_ALL", "TMPDIR")
            if key in os.environ
        }
        self.env.update({"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
        for name in ("PAE_CTX_TOKENS", "PAE_MAX_TOKENS", "PAE_CTX_MARGIN", "PAE_PROFILE"):
            self.env.pop(name, None)

    def write_profile(self, port: int, capacity: int, output: int = 100) -> Path:
        path = self.copy / "fixture-profiles.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profiles": {
                        "fixture": {
                            "provider": "openai-compatible",
                            "url": f"http://127.0.0.1:{port}/v1/chat/completions",
                            "model": "fixture-model",
                            "context_tokens": capacity,
                            "max_output_tokens": output,
                            "disable_thinking": True,
                        }
                    },
                }
            )
        )
        return path

    def write_session(self, name: str, exchanges: int) -> Path:
        directory = self.copy / "prototype_v0" / "data" / "sessions"
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": name,
            "system_prompt": "GOVERNING PROMPT for process tests",
            "schema_version": 1,
            "created_at": "2026-11-03T00:00:00+00:00",
            "updated_at": "2026-11-03T00:00:00+00:00",
            "sources": [
                {"name": "canon.md", "sha256": "e" * 64, "content": "CANON-MARKER"}
            ],
            "messages": build_history(exchanges),
            "derived_state": {"canon": ["CANON-MARKER"]},
            "pending_actions": {},
        }
        path = directory / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def run_cli(self, *args: str, user_input: str) -> str:
        result = subprocess.run(
            [sys.executable, "app.py", *args],
            input=user_input,
            cwd=self.copy / "prototype_v0",
            env=self.env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return result.stdout

    def test_cli_reduces_long_history_and_keeps_session(self) -> None:
        with HttpFixture("[fixture] hello") as fixture:
            profiles = self.write_profile(fixture.port, capacity=ROOMY)
            session_path = self.write_session("cli-long", exchanges=12)

            output = self.run_cli(
                "--profile", "fixture",
                "--profiles-file", str(profiles),
                "--session", "cli-long",
                user_input="hello\n/quit\n",
            )

            self.assertEqual(len(fixture.requests), 1)
            self.assertIn("[context guard]", output)
            self.assertIn("[fixture] hello", output)
            sent = fixture.requests[0]["messages"]
            self.assertEqual(sent[-1]["content"], "hello")
            self.assertEqual(fixture.requests[0]["max_tokens"], 100)
            joined = "\n".join(message["content"] for message in sent)
            self.assertNotIn("question-00", joined)
            self.assertIn("CANON-MARKER", joined)

            persisted = json.loads(session_path.read_text())
            self.assertEqual(len(persisted["messages"]), 26)  # 24 + user + reply
            self.assertEqual(persisted["sources"][0]["content"], "CANON-MARKER")
            self.assertEqual(persisted["derived_state"], {"canon": ["CANON-MARKER"]})

    def test_cli_refuses_oversized_preamble_with_zero_requests(self) -> None:
        with HttpFixture() as fixture:
            profiles = self.write_profile(fixture.port, capacity=TIGHT)
            session_path = self.write_session("cli-tight", exchanges=2)
            before = session_path.read_bytes()

            output = self.run_cli(
                "--profile", "fixture",
                "--profiles-file", str(profiles),
                "--session", "cli-tight",
                user_input="hello\n/quit\n",
            )

            self.assertEqual(fixture.requests, [])
            self.assertIn("[agent error]", output)
            self.assertIn("Request refused", output)
            self.assertIn("estimate", output.lower())
            reloaded = json.loads(session_path.read_text())
            # Only the user's new message was appended; nothing was erased.
            self.assertEqual(len(reloaded["messages"]), 5)
            self.assertEqual(reloaded["sources"], json.loads(before)["sources"])

    def mobile(self, profiles: Path):
        ready = self.copy / "mobile-ready.json"
        ready.unlink(missing_ok=True)
        script = (
            "import json; from pathlib import Path; import mobile_server as m; "
            "server=m.ThreadingHTTPServer(('127.0.0.1',0),m.Handler); "
            "Path('../mobile-ready.json').write_text(json.dumps({'port':server.server_port})); "
            "server.serve_forever()"
        )
        env = dict(self.env)
        env.update({"PAE_PROFILE": "fixture", "PAE_PROFILES_FILE": str(profiles)})
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            cwd=self.copy / "prototype_v0",
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(self._stop, process)
        deadline = time.monotonic() + 20
        while not ready.exists():
            if process.poll() is not None:
                raise AssertionError("mobile startup failed: " + process.stderr.read()[-2000:])
            if time.monotonic() > deadline:
                raise AssertionError("mobile startup timed out")
            time.sleep(0.05)
        return json.loads(ready.read_text())["port"]

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stderr:
            process.stderr.close()

    def api(self, port: int, path: str, payload: dict | None = None) -> tuple[int, dict]:
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(
            f"http://127.0.0.1:{port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=20) as response:
                return response.status, json.loads(response.read())
        except Exception as exc:  # HTTPError carries the body
            body = getattr(exc, "read", lambda: b"{}")()
            return getattr(exc, "code", 0), json.loads(body or b"{}")

    def test_mobile_reduces_long_history_and_reports_budget(self) -> None:
        with HttpFixture("[fixture] mobile") as fixture:
            profiles = self.write_profile(fixture.port, capacity=ROOMY)
            session_path = self.write_session("pae-mobile", exchanges=12)
            port = self.mobile(profiles)

            status, info = self.api(port, "/api/runtime")
            self.assertEqual(status, 200)
            budget = info["context_budget"]
            self.assertEqual(budget["capacity_tokens"], ROOMY)
            self.assertEqual(budget["output_reservation_tokens"], 100)
            self.assertEqual(budget["input_budget_tokens"], ROOMY - 100 - 384)

            status, reply = self.api(port, "/api/chat", {"message": "hello"})
            self.assertEqual(status, 200, reply)
            self.assertEqual(reply["reply"], "[fixture] mobile")
            self.assertEqual(len(fixture.requests), 1)

            sent = fixture.requests[0]["messages"]
            self.assertEqual(sent[-1]["content"], "hello")
            joined = "\n".join(message["content"] for message in sent)
            self.assertNotIn("question-00", joined)

            _, info = self.api(port, "/api/runtime")
            last = info["context_budget"]["last_request"]
            self.assertGreater(last["omitted_exchanges"], 0)
            self.assertGreater(last["omitted_messages"], 0)
            self.assertNotIn("question-00", json.dumps(last))

            _, history = self.api(port, "/api/history")
            self.assertEqual(len(history["messages"]), 26)
            persisted = json.loads(session_path.read_text())
            self.assertEqual(len(persisted["messages"]), 26)

    def test_mobile_refuses_required_material_with_zero_requests(self) -> None:
        with HttpFixture() as fixture:
            profiles = self.write_profile(fixture.port, capacity=TIGHT)
            session_path = self.write_session("pae-mobile", exchanges=2)
            before = json.loads(session_path.read_text())
            port = self.mobile(profiles)

            status, payload = self.api(port, "/api/chat", {"message": "hello"})

            self.assertEqual(status, 500)
            self.assertIn("Request refused", payload["error"])
            self.assertIn("estimate", payload["error"].lower())
            self.assertEqual(fixture.requests, [])
            reloaded = json.loads(session_path.read_text())
            # The mobile chat rolls back its own append on failure.
            self.assertEqual(reloaded["messages"], before["messages"])
            self.assertEqual(reloaded["sources"], before["sources"])
            self.assertEqual(reloaded["derived_state"], before["derived_state"])


if __name__ == "__main__":
    unittest.main()
