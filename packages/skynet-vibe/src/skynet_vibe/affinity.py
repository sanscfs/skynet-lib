"""Run-time weighting math.

All functions here are pure and fast -- no I/O, no LLM calls. They implement
the core gradient formula documented in the plan doc::

    signal_weight = confidence
                  * source_trust[source.type]
                  * time_decay(timestamp, now, half_life_days)
                  * cosine(content_vector, prototype_centroid)
                  * (1 + alpha * cosine(content_vector, context_vector))
"""

from __future__ import annotations

import math
from datetime import datetime

from skynet_vibe.signals import VibeSignal

# Source trust table. Source types in the Source schema.
SOURCE_TRUST: dict[str, float] = {
    "chat": 1.0,
    "reaction": 0.7,
    "wiki": 0.9,
    "consumption": 0.8,
    "dag": 0.7,
    "implicit": 0.5,
}


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0.0 on zero-norm or dimension mismatch."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def time_decay(ts: datetime, now: datetime, half_life_days: float) -> float:
    """Exponential decay with a given half-life in days.

    ``time_decay(ts=now - half_life_days, now=now, half_life_days=H) == 0.5``.
    Assumes both datetimes are timezone-aware; if ``ts`` is naive it is
    treated as already aligned with ``now``.
    """
    if half_life_days <= 0:
        return 1.0
    try:
        delta = now - ts
    except TypeError:
        # Mismatched tz-aware vs naive; best-effort by stripping tz.
        delta = now.replace(tzinfo=None) - ts.replace(tzinfo=None)
    age_days = max(0.0, delta.total_seconds() / 86400.0)
    return 0.5 ** (age_days / half_life_days)


def signal_weight(
    signal: VibeSignal,
    prototype_centroid: list[float] | None,
    context_vector: list[float] | None,
    now: datetime,
    half_life_days: float = 45.0,
    context_alpha: float = 0.5,
) -> float:
    """Compute the gradient weight for a single signal.

    Parameters
    ----------
    signal:
        The signal whose weight we want.
    prototype_centroid:
        Domain centroid. If ``None``, the prototype term is elided (treated
        as 1.0) -- used for domain-agnostic retrieval where every signal
        contributes regardless of domain.
    context_vector:
        Optional current-context embedding. If ``None``, the context boost
        term is elided (treated as 1.0).
    now:
        Reference time for decay.
    half_life_days:
        Exponential decay half-life.
    context_alpha:
        Weight of the context-boost multiplier. At 0 the context is ignored;
        at 1 it can double the weight when perfectly aligned.
    """
    trust = SOURCE_TRUST.get(signal.source.type, 0.5)
    decay = time_decay(signal.timestamp, now, half_life_days)
    proto_term = 1.0
    if prototype_centroid is not None:
        proto_term = max(0.0, cosine(signal.vectors.content, prototype_centroid))
    context_term = 1.0
    if context_vector is not None:
        context_term = 1.0 + context_alpha * cosine(signal.vectors.content, context_vector)
    return float(signal.confidence) * trust * decay * proto_term * context_term
