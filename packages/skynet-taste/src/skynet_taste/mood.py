"""Decay-weighted mood vector from live Qdrant signals.

``get_mood_vector`` queries the vibe collection, weights each signal by
``exp(−λ × missed_opportunities) × confidence``, and returns a weighted
centroid. The result is stateless — computed fresh on every call so
there's no Redis state to reset on pod restart.

Auto-tuning: new signals (missed_opps=0) dominate. As the engine runs
``suggest()`` and increments missed_opportunities for non-top-5 signals,
old mood signals naturally fade while freshly-reinforced ones stay strong.
User chat corrections land as trait signals (via ``absorb_correction``)
and immediately anchor the centroid.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from skynet_scoring import compute_decay_factor_logical

from skynet_taste._math import cosine_alignment, weighted_centroid
from skynet_taste.models import MoodResult

logger = logging.getLogger("skynet_taste.mood")

Embedder = Callable[[str], Awaitable[list[float]] | list[float]]


async def _embed(embedder: Embedder, text: str) -> list[float]:
    result = embedder(text)
    if asyncio.iscoroutine(result):
        result = await result
    return list(result)


async def get_mood_vector(
    store: Any,
    embedder: Embedder,
    *,
    query: str | None = None,
    domain_hint: str = "current vibe",
    top_k: int = 50,
    source_type: str | None = None,
    decay_lambdas: dict[str, float] | None = None,
) -> MoodResult:
    """Compute a decay-weighted mood centroid from Qdrant vibe signals.

    Parameters
    ----------
    store:
        A ``VibeStore`` instance. Searched via ``store.search()``.
    embedder:
        Async or sync callable ``(text) -> list[float]``.
    query:
        Free-text description of the current moment / what to match.
        If None, falls back to ``domain_hint``.
    domain_hint:
        Fallback seed text when ``query`` is absent. Should be a broad
        domain label like "music" or "cinema" or "current vibe".
    top_k:
        How many nearest signals to pull. More = smoother centroid.
    source_type:
        If set, filter to signals whose ``extra_payload.source_type``
        matches (e.g. ``"music_review"``). None = all domains.
    decay_lambdas:
        Override decay lambdas (passed to ``compute_decay_factor_logical``).
        None = defaults from skynet-scoring.
    """
    seed_text = query or domain_hint
    seed_vec = await _embed(embedder, seed_text)

    filter_: dict | None = None
    if source_type:
        filter_ = {"must": [{"key": "extra_payload.source_type", "match": {"value": source_type}}]}

    try:
        signals = await store.search(seed_vec, top_k=top_k, noise_floor=0.0, filter_=filter_)
    except Exception as exc:
        logger.warning("mood search failed: %s", exc)
        return MoodResult(vector=[], confidence=0.0, age=0.0, signal_count=0)

    if not signals:
        return MoodResult(vector=[], confidence=0.0, age=0.0, signal_count=0)

    weights: list[float] = []
    for sig in signals:
        decay = compute_decay_factor_logical(sig.extra_payload or {}, lambdas=decay_lambdas)
        w = decay * float(sig.confidence)
        weights.append(max(0.0, w))

    vecs = [sig.vectors.content for sig in signals]
    mood_vec = weighted_centroid(vecs, weights)
    if not mood_vec:
        return MoodResult(vector=[], confidence=0.0, age=0.0, signal_count=len(signals))

    confidence = cosine_alignment(vecs, mood_vec, weights)

    total_w = sum(weights)
    if total_w > 0:
        age = sum(
            w * float((sig.extra_payload or {}).get("missed_opportunities", 0))
            for sig, w in zip(signals, weights)
        ) / total_w
    else:
        age = 0.0

    return MoodResult(
        vector=mood_vec,
        confidence=float(confidence),
        age=float(age),
        signal_count=len(signals),
    )
