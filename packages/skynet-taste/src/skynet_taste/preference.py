"""Domain-specific preference vector from trait-class vibe signals.

Preference = decay-weighted centroid of signals whose ``source_type``
matches the domain (e.g. ``"music_review"``, ``"movie_review"``).

Unlike mood, preference is seeded by the domain label embedding (not
the user's current query) so it captures long-term taste regardless
of what the user is asking about right now. Logical-time decay still
applies — preferences that never show up in recommendations gradually
fade, while actively-reinforced ones stay fresh.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from skynet_scoring import compute_decay_factor_logical

from skynet_taste._math import weighted_centroid

logger = logging.getLogger("skynet_taste.preference")

Embedder = Callable[[str], Awaitable[list[float]] | list[float]]


async def _embed(embedder: Embedder, text: str) -> list[float]:
    result = embedder(text)
    if asyncio.iscoroutine(result):
        result = await result
    return list(result)


async def get_preference_vector(
    store: Any,
    embedder: Embedder,
    *,
    source_type: str,
    domain_hint: str | None = None,
    top_k: int = 100,
    decay_lambdas: dict[str, float] | None = None,
) -> list[float] | None:
    """Compute a decay-weighted preference centroid for a domain.

    Parameters
    ----------
    store:
        A ``VibeStore`` instance.
    embedder:
        Async or sync callable ``(text) -> list[float]``.
    source_type:
        Qdrant filter value for ``extra_payload.source_type``.
        E.g. ``"music_review"``, ``"movie_review"``.
    domain_hint:
        Text to embed as the search seed. Falls back to ``source_type``
        if not provided. Should be a broad domain label: "music I like",
        "films I enjoy", etc.
    top_k:
        Max signals to pull. Preference needs more breadth than mood.
    decay_lambdas:
        Override lambdas; None = defaults.

    Returns None if no qualifying signals found.
    """
    seed_text = domain_hint or source_type
    seed_vec = await _embed(embedder, seed_text)

    filter_ = {"must": [{"key": "extra_payload.source_type", "match": {"value": source_type}}]}

    try:
        signals = await store.search(seed_vec, top_k=top_k, noise_floor=0.0, filter_=filter_)
    except Exception as exc:
        logger.warning("preference search failed for %r: %s", source_type, exc)
        return None

    if not signals:
        return None

    weights: list[float] = []
    for sig in signals:
        decay = compute_decay_factor_logical(sig.extra_payload or {}, lambdas=decay_lambdas)
        w = decay * float(sig.confidence)
        weights.append(max(0.0, w))

    result = weighted_centroid([sig.vectors.content for sig in signals], weights)
    return result if result else None
