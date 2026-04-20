"""Light-LLM reply resolver for pending captures.

Given the user's thread reply and the list of candidates originally
shown, asks a minimal LLM call to pick the right index.
No persona, no history, no context — purely mechanical selection.
"""

from __future__ import annotations

import json
import logging
from typing import Awaitable, Callable

from skynet_capture.sources.base import Candidate

log = logging.getLogger("skynet_capture.resolver")

LLMCaller = Callable[[str, str], Awaitable[str]]

_SYSTEM = (
    "The user chose from options. Return ONLY valid JSON: "
    '{{"index": <0-based int>}} or {{"index": null}} if unclear. '
    "Options:\n{options}"
)


async def resolve_candidate_reply(
    user_text: str,
    candidates: list[Candidate],
    llm_call: LLMCaller,
) -> int | None:
    """Return 0-based candidate index chosen by user, or None if unclear."""
    if not candidates:
        return None

    options = "\n".join(
        f"{i}. {c.title}{' — ' + c.subtitle if c.subtitle else ''}{' (' + str(c.year) + ')' if c.year else ''}"
        for i, c in enumerate(candidates)
    )
    system = _SYSTEM.format(options=options)
    try:
        raw = await llm_call(system, user_text)
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json\n"):
                raw = raw[5:]
        data = json.loads(raw)
        idx = data.get("index")
        if idx is None:
            return None
        idx = int(idx)
        if 0 <= idx < len(candidates):
            return idx
    except Exception as exc:
        log.debug("resolve_candidate_reply failed: %s", exc)
    return None


__all__ = ["resolve_candidate_reply", "LLMCaller"]
