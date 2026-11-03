from __future__ import annotations

import json
from typing import Any

from context_guard import guard_request
from provider import Provider


EXTRACTION_SYSTEM_PROMPT = """You are the state extractor for PAE prototype_v0.

Your job is to derive a compact structured state ONLY from the persistent source
material supplied in this request.

Hard rules:
- Source material is authoritative.
- Do not invent missing details.
- When a value is not supported by the sources, write exactly "UNKNOWN".
- Keep facts concise and concrete.
- Distinguish explicit canon/facts from uncertainty.
- Return JSON only. No Markdown fences, commentary, or prose outside the JSON.

Return exactly this top-level shape:
{
  "canon": ["fact", "..."],
  "characters": [
    {"name": "name", "facts": ["fact", "..."]}
  ],
  "current_scene": {
    "location": "value or UNKNOWN",
    "situation": "value or UNKNOWN",
    "present_characters": ["name", "..."]
  },
  "important_facts": ["fact", "..."],
  "unknowns": ["missing/uncertain detail", "..."]
}
"""


def _source_bundle(sources: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for source in sources:
        chunks.append(
            "SOURCE\n"
            f"Name: {source['name']}\n"
            f"SHA256: {source['sha256']}\n"
            "--- BEGIN SOURCE ---\n"
            f"{source['content']}\n"
            "--- END SOURCE ---"
        )
    return "\n\n".join(chunks)


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("state extractor did not return a JSON object")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError("state extractor returned invalid JSON") from exc

    if not isinstance(data, dict):
        raise RuntimeError("state extractor JSON must be an object")

    return data


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else "UNKNOWN"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _normalize_state(data: dict[str, Any]) -> dict[str, Any]:
    characters: list[dict[str, Any]] = []
    raw_characters = data.get("characters", [])

    if isinstance(raw_characters, list):
        for item in raw_characters:
            if not isinstance(item, dict):
                continue
            name = _string(item.get("name"))
            facts = _string_list(item.get("facts"))
            characters.append({"name": name, "facts": facts})

    raw_scene = data.get("current_scene")
    if not isinstance(raw_scene, dict):
        raw_scene = {}

    return {
        "canon": _string_list(data.get("canon")),
        "characters": characters,
        "current_scene": {
            "location": _string(raw_scene.get("location")),
            "situation": _string(raw_scene.get("situation")),
            "present_characters": _string_list(
                raw_scene.get("present_characters")
            ),
        },
        "important_facts": _string_list(data.get("important_facts")),
        "unknowns": _string_list(data.get("unknowns")),
    }


def extract_state(
    provider: Provider,
    sources: list[dict[str, str]],
) -> dict[str, Any]:
    if not sources:
        raise RuntimeError("no persistent sources loaded")

    if provider.name == "echo":
        raise RuntimeError(
            "state extraction requires a real model provider; "
            "PAE_PROVIDER=echo can only smoke-test the chat loop"
        )

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Extract the persistent state from these sources. "
                "Remember: unsupported values must stay UNKNOWN.\n\n"
                + _source_bundle(sources)
            ),
        },
    ]

    # The source bundle is required material: it cannot be reduced without
    # answering about sources that were never supplied. An oversized bundle is
    # refused here, before the provider call and before any persisted state
    # changes.
    guarded = guard_request(provider, messages, purpose="state extraction")
    if guarded.reduced:
        print(f"[context guard] {guarded.summary()}")

    raw = provider.chat(guarded.messages)
    return _normalize_state(_extract_json_object(raw))
