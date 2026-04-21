"""Candidate scoring: mood × preference^β.

``score_candidate``:
    combined = cosine(candidate, mood) × cosine(candidate, pref)^β

β=0 → pure mood, ignores long-term taste.
β=1 → mood and taste weighted equally (geometric mean).
β=0.5 (default) → taste modulates mood without dominating.

``rank_candidates``:
    Batch scorer. Returns sorted (score, candidate) pairs. Candidates
    that have neither "vector" nor "text" field are silently skipped.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from skynet_taste._math import cosine

logger = logging.getLogger("skynet_taste.scorer")

Embedder = Callable[[str], Awaitable[list[float]] | list[float]]


def score_candidate(
    candidate_vec: list[float],
    mood_vec: list[float],
    pref_vec: list[float] | None = None,
    beta: float = 0.5,
) -> float:
    """Combined mood + preference score for one candidate vector.

    Returns 0.0 if either vector is empty. ``pref_vec=None`` or
    ``beta=0`` falls back to pure mood cosine score.
    """
    if not candidate_vec or not mood_vec:
        return 0.0
    mood_score = max(0.0, cosine(candidate_vec, mood_vec))
    if pref_vec is None or beta == 0.0:
        return mood_score
    pref_score = max(0.0, cosine(candidate_vec, pref_vec))
    return mood_score * (pref_score ** beta)


async def rank_candidates(
    candidates: list[dict[str, Any]],
    mood_vec: list[float],
    embedder: Embedder,
    pref_vec: list[float] | None = None,
    beta: float = 0.5,
    top_k: int = 10,
) -> list[tuple[float, dict[str, Any]]]:
    """Score and rank a list of candidates.

    Each candidate dict may carry a pre-computed ``"vector"`` key or a
    ``"text"`` key that gets embedded on-the-fly. Candidates without
    either are skipped. Returns top ``top_k`` sorted descending.
    """
    scored: list[tuple[float, dict[str, Any]]] = []
    for cand in candidates:
        vec = cand.get("vector")
        if not vec and cand.get("text"):
            raw = embedder(cand["text"])
            vec = (await raw) if asyncio.iscoroutine(raw) else raw
        if not vec:
            continue
        s = score_candidate(list(vec), mood_vec, pref_vec, beta)
        scored.append((s, cand))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:top_k]
