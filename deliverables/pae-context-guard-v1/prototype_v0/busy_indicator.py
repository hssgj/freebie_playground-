"""Small terminal-only busy indicator for blocking provider calls."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from itertools import cycle
from threading import Event, Thread
from typing import Iterator, TextIO


FRAMES = ("|", "/", "-", "\\")
LABEL = "[pae] Zpracovávám požadavek"


@contextmanager
def busy_indicator(*, stream: TextIO | None = None, interval: float = 0.12) -> Iterator[None]:
    """Show activity, without claiming that model tokens are streaming.

    Animate only when stdout is a terminal; otherwise print one stable status
    line so redirected output/logs contain no control characters.
    """
    output = stream if stream is not None else sys.stdout
    if not output.isatty():
        print(f"{LABEL}...", file=output, flush=True)
        yield
        return

    stop = Event()

    def draw(frame: str) -> None:
        output.write(f"\r\033[2K{LABEL} {frame}")
        output.flush()

    draw(FRAMES[0])  # Immediate feedback, even for very fast calls.

    def animate() -> None:
        for frame in cycle(FRAMES[1:] + FRAMES[:1]):
            if stop.wait(interval):
                return
            draw(frame)

    worker = Thread(target=animate, name="pae-busy-spinner", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
        output.write("\r\033[2K")
        output.flush()


class BusyProvider:
    """Wrap provider.chat; keep the spinner out of runtime tool logs."""

    def __init__(self, provider: object) -> None:
        self._provider = provider
        self.name = provider.name

    def __getattr__(self, name: str):
        # Keep declared capacity (context_tokens / max_output_tokens) visible to
        # the context guard, which must see the real budget through this wrapper.
        return getattr(self._provider, name)

    def chat(self, messages: list[dict[str, str]]) -> str:
        with busy_indicator():
            return self._provider.chat(messages)
