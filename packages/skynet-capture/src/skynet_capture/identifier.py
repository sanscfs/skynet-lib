"""LLM-scored activity identification.

Flow:
  1. ``ActivitySource.search(text)``  → up to 5 Candidates from a public DB.
  2. Light LLM call: user text + formatted candidates → JSON confidence + index.
  3. If confidence >= threshold: return ``confirmed`` with chosen Candidate.
     If candidates exist but confidence low: return ``clarify`` + LLM-written question.
     If no candidates: return ``not_found`` + LLM-written question.

The LLM call uses a two-line prompt so any small model (qwen3:4b, mistral-small)
can handle it without hallucinating. Callers pass ``llm_call`` matching
the ``LLMCaller`` signature from ``skynet_matrix.chat_agent``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from skynet_capture.sources.base import ActivitySource, Candidate

log = logging.getLogger("skynet_capture.identifier")

LLMCaller = Callable[[str, str], Awaitable[str]]

CONFIRMED_THRESHOLD = 0.90

_SCORE_SYSTEM = """\
Identify the best match for the user's message from the search results below.
Return ONLY valid JSON — no markdown, no prose:
{"index": <0-based int or null>, "confidence": <0.0-1.0>}
Use null index when no result fits. Confidence 1.0 = exact unambiguous match.\
"""

_CLARIFY_SYSTEM = """\
The user mentioned a music entity. Search found candidates but confidence is low.
Write a SHORT clarifying question in Ukrainian (1-2 sentences).
Return ONLY valid JSON: {"message": "<question>"}
Include the most likely candidate(s) in the question.\
"""

_NOT_FOUND_SYSTEM = """\
The user mentioned a music entity but nothing was found in the database.
Write a SHORT question in Ukrainian asking them to clarify.
Return ONLY valid JSON: {"message": "<question>"}\
"""


@dataclass
class IdentifyResult:
    status: Literal["confirmed", "clarify", "not_found"]
    candidate: Candidate | None  # set when confirmed
    candidates: list[Candidate] = field(default_factory=list)
    confidence: float = 0.0
    message: str = ""  # what to reply to the user


class ActivityIdentifier:
    """Combine a search source with a light LLM to produce IdentifyResult."""

    def __init__(
        self,
        source: ActivitySource,
        llm_call: LLMCaller,
        *,
        threshold: float = CONFIRMED_THRESHOLD,
        domain: str = "music",
    ) -> None:
        self._source = source
        self._llm = llm_call
        self._threshold = threshold
        self._domain = domain

    async def identify(self, text: str) -> IdentifyResult:
        candidates = await self._source.search(text)

        if not candidates:
            msg = await self._llm_message(_NOT_FOUND_SYSTEM, text, candidates=[])
            return IdentifyResult(
                status="not_found",
                candidate=None,
                candidates=[],
                confidence=0.0,
                message=msg or "Нічого не знайшов, можеш уточнити?",
            )

        idx, confidence = await self._llm_score(text, candidates)

        if confidence >= self._threshold and idx is not None:
            return IdentifyResult(
                status="confirmed",
                candidate=candidates[idx],
                candidates=candidates,
                confidence=confidence,
                message="",
            )

        msg = await self._llm_message(_CLARIFY_SYSTEM, text, candidates=candidates)
        return IdentifyResult(
            status="clarify",
            candidate=None,
            candidates=candidates,
            confidence=confidence,
            message=msg or _default_clarify(candidates),
        )

    async def _llm_score(self, text: str, candidates: list[Candidate]) -> tuple[int | None, float]:
        numbered = _format_candidates(candidates)
        user = f'User wrote: "{text}"\n\nSearch results:\n{numbered}'
        try:
            raw = await self._llm(_SCORE_SYSTEM, user)
            data = _parse_json(raw)
            idx = data.get("index")
            confidence = float(data.get("confidence", 0.0))
            if idx is None:
                return None, confidence
            idx = int(idx)
            if 0 <= idx < len(candidates):
                return idx, confidence
        except Exception as exc:
            log.debug("LLM score failed: %s", exc)
        return None, 0.0

    async def _llm_message(self, system: str, text: str, candidates: list[Candidate]) -> str | None:
        numbered = _format_candidates(candidates) if candidates else ""
        user = f'User wrote: "{text}"\n\nCandidates:\n{numbered}' if numbered else f'User wrote: "{text}"'
        try:
            raw = await self._llm(system, user)
            data = _parse_json(raw)
            msg = data.get("message") or ""
            return msg.strip() or None
        except Exception as exc:
            log.debug("LLM message failed: %s", exc)
        return None


def _format_candidates(candidates: list[Candidate]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        year = f" ({c.year})" if c.year else ""
        subtitle = f" — {c.subtitle}" if c.subtitle else ""
        lines.append(f"{i}. [{c.type}] {c.title}{subtitle}{year}")
    return "\n".join(lines)


def _default_clarify(candidates: list[Candidate]) -> str:
    top = candidates[:3]
    options = "; ".join(
        f"{c.title}{' — ' + c.subtitle if c.subtitle else ''}{' (' + str(c.year) + ')' if c.year else ''}" for c in top
    )
    return f"Знайшов кілька варіантів: {options}. Що саме мав на увазі?"


def _parse_json(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json\n"):
            raw = raw[5:]
    return json.loads(raw)


__all__ = ["ActivityIdentifier", "IdentifyResult", "LLMCaller", "CONFIRMED_THRESHOLD"]
