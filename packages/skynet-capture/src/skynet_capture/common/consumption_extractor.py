"""LLM-backed multi-item extractor for movies / music consumption.

Moved verbatim from ``skynet_profiler.modules._consumption_extractor`` in
Phase 1 of the shared-ingest rollout. The one intentional diff: the LLM
dependency used to be the concrete ``skynet_profiler.llm.LLMClient``;
here it's a ``Protocol`` (``LLMLike``) so the lib doesn't pull in any
LLM SDK and callers can still pass their existing ``LLMClient`` with
zero code change (it already satisfies the shape).

Why this exists: the regex fast path in each consumer only recognises
"!mark-* X" slash commands and a handful of single-item natural-language
openers ("подивився X", "just watched X"). Free-form Matrix messages
routinely mention two or three things in one paragraph plus rich review
text — exactly the signal we don't want to lose.

The LLM gets the full message and returns a JSON list of items. Each
item includes a ``notes`` field holding ONLY the sentences pertinent to
that item, not the whole message repeated. This matches the PG
``watch_log.notes`` / ``listens.notes`` column intent: per-item
user-authored review text.

Design choices:

* **Kind-specific prompts**: movies and music have different failure
  modes (e.g. "слухав" reliably means music, but "дивився" could mean
  "was looking at" in a non-film sense); inlining two prompts is
  cheaper than trying to build a unified schema that confuses both.
* **Watched/listened flag**: the prompt asks the LLM to distinguish
  "finished" from "still watching". Items flagged as in-progress are
  returned with ``watched=False`` / ``listened=False`` so the caller
  can drop them cleanly; dropping inside the helper would hide the
  distinction from module-level metrics.
* **Defensive parse**: every JSON field is coerced + validated so a
  hallucinated year string ("2026 or 2027") doesn't crash the caller.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol

log = logging.getLogger("skynet_capture.common.consumption_extractor")


class LLMLike(Protocol):
    """The narrow subset of an LLM client this extractor needs.

    ``skynet_profiler.llm.LLMClient`` satisfies this directly; so would
    any future async client that implements ``json_completion``. Kept as
    a Protocol so the lib has zero LLM-SDK dependency and tests can
    pass a plain fake.
    """

    async def json_completion(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> dict[str, Any] | None: ...


# Minimum length of a non-regex-match message before the LLM is invoked.
# Below this we assume "hmm, coffee time" style chatter — not worth the
# token budget. The module re-exports this so tests can assert the
# boundary behaviour.
MIN_LLM_TEXT_LEN = 80


_MOVIES_SYSTEM = (
    "You extract movie watch events from a user's chat message. "
    "The user may mention multiple movies in one message with a mixed "
    "review covering all of them. Return STRICT JSON with the shape:\n"
    '{"movies": [{"title": str, "year": int | null, '
    '"notes": str, "watched": bool}]}\n'
    "Rules:\n"
    "- 'title' is the movie title as the user wrote it; keep original "
    "casing and language (Ukrainian/English/mixed). Do NOT translate.\n"
    "- 'year' only if explicitly stated; null otherwise.\n"
    "- 'notes' must contain ONLY the sentences discussing this specific "
    "movie; if the user mixed impressions ('both were ok, but X was "
    "better'), split per movie as best you can; never duplicate the "
    "whole message across movies.\n"
    "- 'watched' is true for completed watches ('watched X', "
    "'подивився X', 'finished X'), false for in-progress "
    "('дивлюсь зараз X', 'watching X now', 'ще не закінчив X', "
    "'halfway through X').\n"
    "- If the message is NOT about watching any movie, return "
    '{"movies": []}.\n'
    "- Return [] not null when empty."
)


_MUSIC_SYSTEM = (
    "You extract music listening events from a user's chat message. "
    "The user may mention multiple artists/tracks/albums in one message "
    "with a mixed review covering all of them. Return STRICT JSON with "
    "the shape:\n"
    '{"tracks": [{"artist": str, "track": str, "year": int | null, '
    '"notes": str, "listened": bool}]}\n'
    "Rules:\n"
    "- 'track' is the album/song title as the user wrote it; keep "
    "original casing and language. Do NOT translate.\n"
    "- 'artist' is empty string '' when the user did not name one.\n"
    "- 'year' only if explicitly stated; null otherwise.\n"
    "- 'notes' must contain ONLY the sentences discussing this specific "
    "item; never duplicate the whole message across items.\n"
    "- 'listened' is true for completed listens ('слухав X', "
    "'listened to X', 'heard X'), false for in-progress / queued "
    "('слухаю зараз X', 'collecting Y', 'about to listen to X').\n"
    "- If the message is NOT about listening to music, return "
    '{"tracks": []}.\n'
    "- Return [] not null when empty."
)


def _clamp_str(v: Any, *, default: str = "", max_len: int = 2000) -> str:
    if v is None:
        return default
    if not isinstance(v, str):
        v = str(v)
    v = v.strip()
    if len(v) > max_len:
        v = v[:max_len]
    return v


def _clamp_year(v: Any) -> int | None:
    if v is None:
        return None
    try:
        year = int(v)
    except (TypeError, ValueError):
        return None
    if year < 1880 or year > 2100:
        return None
    return year


def _clamp_flag(v: Any, *, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return default


def _validate_movies(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    raw = data.get("movies")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = _clamp_str(item.get("title"))
        if not title:
            continue
        out.append(
            {
                "title": title,
                "year": _clamp_year(item.get("year")),
                "notes": _clamp_str(item.get("notes")),
                "watched": _clamp_flag(item.get("watched"), default=True),
            }
        )
    return out


def _validate_tracks(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    raw = data.get("tracks")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        track = _clamp_str(item.get("track"))
        if not track:
            continue
        out.append(
            {
                "artist": _clamp_str(item.get("artist")),
                "track": track,
                "year": _clamp_year(item.get("year")),
                "notes": _clamp_str(item.get("notes")),
                "listened": _clamp_flag(item.get("listened"), default=True),
            }
        )
    return out


async def extract_consumption(
    text: str,
    *,
    kind: Literal["movies", "music"],
    llm: LLMLike,
) -> list[dict[str, Any]]:
    """Ask the LLM to split a free-form message into per-item records.

    Returns an empty list on any failure (LLM unreachable, invalid JSON,
    zero usable items). Callers MUST treat an empty return as "no
    signal"; they must NOT fall back to "treat whole message as one
    item" because that would re-introduce the exact bug this module
    path is supposed to fix (notes-field compression).

    The function does no side effects — pure "text + LLM -> structured
    list". Modules decide deduping, filtering, and PG writes.
    """
    if not text or not text.strip():
        return []
    if kind == "movies":
        system = _MOVIES_SYSTEM
    elif kind == "music":
        system = _MUSIC_SYSTEM
    else:  # pragma: no cover — Literal guarantees the set
        raise ValueError(f"unknown kind: {kind!r}")

    data = await llm.json_completion(
        system=system,
        user=text,
        max_tokens=800,
        temperature=0.0,
    )
    if data is None:
        log.debug("LLM returned None for %s extraction", kind)
        return []
    if kind == "movies":
        return _validate_movies(data)
    return _validate_tracks(data)
