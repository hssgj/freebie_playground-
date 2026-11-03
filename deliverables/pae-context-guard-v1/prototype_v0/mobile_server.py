from __future__ import annotations

import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from agent_loop import run_agent_turn
from app import DEFAULT_SYSTEM_PROMPT, build_messages
from context_guard import ContextBudgetError, last_diagnostics, resolve_budget
from github_source import github_configured
from gmail_source import refresh_configured
from provider import build_provider
from runtime_tools import build_tool_registry
from sessions import SessionStore
from tool_core import ToolContext


HOST = "127.0.0.1"
PORT = 8765
SESSION_ID = "pae-mobile"
STATIC_DIR = Path(__file__).with_name("mobile_web")
REPO_ROOT = Path(__file__).resolve().parent.parent
STARTED_AT = datetime.now(timezone.utc).isoformat()


def _git_value(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"


GIT_COMMIT_AT_START = _git_value("rev-parse", "HEAD")
GIT_BRANCH_AT_START = _git_value("branch", "--show-current")
GIT_TRACKING_AT_START = _git_value("rev-parse", "--abbrev-ref", "@{upstream}")
GIT_RUNTIME_STATUS_AT_START = _git_value(
    "status", "--porcelain", "--", "prototype_v0", "start-mobile.sh"
)


class MobileRuntime:
    def __init__(self) -> None:
        self.store = SessionStore()
        self.session = self.store.open_or_create(
            SESSION_ID, system_prompt=DEFAULT_SYSTEM_PROMPT
        )
        self.provider = build_provider()
        self.registry = build_tool_registry()
        self.context = ToolContext(store=self.store, session=self.session)
        self.lock = threading.Lock()

    def context_budget(self) -> dict[str, object]:
        """Capacity, reservations and the last request's counts. No contents."""
        try:
            budget = resolve_budget(self.provider)
        except ContextBudgetError as exc:
            return {"error": str(exc)}
        return {**budget.diagnostics(), "last_request": last_diagnostics()}

    def runtime_info(self) -> dict[str, object]:
        return {
            "status": "ready",
            "context_budget": self.context_budget(),
            "profile": getattr(self.provider, "profile_id", None),
            "configured_model": getattr(self.provider, "model", None),
            "git_commit": GIT_COMMIT_AT_START,
            "branch": GIT_BRANCH_AT_START,
            "tracking_branch": GIT_TRACKING_AT_START,
            "runtime_files_dirty": bool(
                GIT_RUNTIME_STATUS_AT_START and GIT_RUNTIME_STATUS_AT_START != "UNKNOWN"
            ),
            "tools": [spec["name"] for spec in self.registry.specs()],
            "gmail_configured": bool(
                os.getenv("PAE_GMAIL_ACCESS_TOKEN", "").strip() or refresh_configured()
            ),
            "github_configured": github_configured(),
            "server_started_at": STARTED_AT,
            "server_pid": os.getpid(),
            "listen_url": f"http://{HOST}:{PORT}",
        }

    def history(self) -> list[dict[str, str]]:
        return [
            {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
            for item in self.session.messages
            if item.get("role") in {"user", "assistant"}
        ]

    def chat(self, message: str) -> str:
        with self.lock:
            if message.strip() == "/runtime":
                return json.dumps(self.runtime_info(), ensure_ascii=False, indent=2)
            self.session.messages.append({"role": "user", "content": message})
            self.store.save(self.session)
            try:
                reply = run_agent_turn(
                    self.provider,
                    build_messages(self.session),
                    registry=self.registry,
                    context=self.context,
                )
            except Exception:
                self.session.messages.pop()
                self.store.save(self.session)
                raise
            self.session.messages.append({"role": "assistant", "content": reply})
            self.store.save(self.session)
            return reply


RUNTIME = MobileRuntime()


class Handler(BaseHTTPRequestHandler):
    server_version = "PAEMobile/0.1"

    def log_message(self, format: str, *args: object) -> None:
        print(f"[mobile] {self.address_string()} {format % args}")

    def send_json(self, status: int, payload: object) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json(200, {"status": "ready", "session": SESSION_ID})
            return
        if path == "/api/runtime":
            self.send_json(200, RUNTIME.runtime_info())
            return
        if path == "/api/history":
            self.send_json(200, {"messages": RUNTIME.history()})
            return
        if path in {"/", "/index.html"}:
            data = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self.send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 64_000:
                raise ValueError("Neplatná délka zprávy")
            payload = json.loads(self.rfile.read(length))
            message = payload.get("message", "")
            if not isinstance(message, str) or not message.strip():
                raise ValueError("Zpráva je prázdná")
            reply = RUNTIME.chat(message.strip())
            self.send_json(200, {"reply": reply})
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            self.send_json(500, {"error": f"PAE selhalo: {exc}"})


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"PAE mobile ready: http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
