"""Absorb user corrections as trait-class vibe signals.

Positive corrections ("більше такого", "love this vibe") → high salience,
high confidence, memory_class="trait" → slow decay, long-lasting anchor.

Negative corrections ("не те", "wrong vibe") → low salience, low confidence,
memory_class="trait" (still stored, but decays away once it stops being
seen) → the missing missed_opportunities bump does the rest.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from skynet_vibe.signals import FacetVectors, Source, VibeSignal

logger = logging.getLogger("skynet_taste.absorb")

Embedder = Callable[[str], Awaitable[list[float]] | list[float]]


async def absorb_correction(
    store: Any,
    embedder: Embedder,
    text: str,
    *,
    positive: bool = True,
    source_type: str = "correction",
    domain: str | None = None,
    linked_rec_id: str | None = None,
) -> str:
    """Embed ``text`` and store it as a trait-class correction signal.

    Parameters
    ----------
    store:
        A ``VibeStore`` instance.
    embedder:
        Async or sync callable ``(text) -> list[float]``.
    text:
        The correction phrase ("більше такого", "not this vibe", etc.)
    positive:
        True = high salience/confidence (reinforces direction).
        False = low salience/confidence (pushes away).
    source_type:
        Stored as ``extra_payload.source_type``. Use ``"correction"``
        for generic corrections; domain-specific callers may pass
        ``"music_correction"`` etc.
    domain:
        Optional domain tag stored in extra_payload for diagnostics.
    linked_rec_id:
        Link back to the recommendation that triggered this correction.

    Returns the signal_id of the stored signal.
    """
    raw = embedder(text)
    vec: list[float] = (await raw) if asyncio.iscoroutine(raw) else raw

    salience = 0.9 if positive else 0.15
    confidence = 0.95 if positive else 0.2

    extra: dict[str, Any] = {
        "source_type": source_type,
        "memory_class": "trait",
        "salience": salience,
        "missed_opportunities": 0,
        "positive": positive,
    }
    if domain:
        extra["domain"] = domain

    signal = VibeSignal(
        id=VibeSignal.new_id(),
        text_raw=text,
        vectors=FacetVectors(content=list(vec)),
        source=Source(type="chat"),
        timestamp=datetime.now(timezone.utc),
        confidence=confidence,
        linked_rec_id=linked_rec_id,
        extra_payload=extra,
    )
    await store.put(signal)
    logger.info("absorbed %s correction: %r", "positive" if positive else "negative", text[:80])
    return signal.id
