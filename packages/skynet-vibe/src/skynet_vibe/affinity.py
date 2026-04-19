"""Run-time weighting math.

All functions here are pure and fast -- no I/O, no LLM calls. They implement
the core gradient formula::

    signal_weight = confidence
                  * source_trust[source.type]
                  * decay_factor
                  * cosine(content_vector, prototype_centroid)
                  * (1 + alpha * cosine(content_vector, context_vector))

``decay_factor`` is intentionally NOT computed here. Callers pass a
pre-computed value in [0, 1] -- typically from
``skynet_scoring.compute_decay_factor_logical(payload)`` so the whole
Skynet stack shares one logical-time decay rule (missed_opportunities,
silence-safe) instead of each module hand-rolling wall-clock math.
"""

from __future__ import annotations

import math

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


def signal_weight(
    signal: VibeSignal,
    prototype_centroid: list[float] | None,
    context_vector: list[float] | None,
    *,
    decay_factor: float = 1.0,
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
    decay_factor:
        Logical-time decay already computed by the caller (typically via
        ``skynet_scoring.compute_decay_factor_logical(payload)``). 1.0
        means "no decay" -- appropriate for brand-new signals or when
        logical time is disabled. Silence-safe by construction: the
        logical-time clock only ticks when a retrieval had a chance to
        use the signal and chose not to, so a user going quiet does not
        erode vibe strength.
    context_alpha:
        Weight of the context-boost multiplier. At 0 the context is ignored;
        at 1 it can double the weight when perfectly aligned.
    """
    trust = SOURCE_TRUST.get(signal.source.type, 0.5)
    proto_term = 1.0
    if prototype_centroid is not None:
        proto_term = max(0.0, cosine(signal.vectors.content, prototype_centroid))
    context_term = 1.0
    if context_vector is not None:
        context_term = 1.0 + context_alpha * cosine(signal.vectors.content, context_vector)
    return float(signal.confidence) * trust * max(0.0, decay_factor) * proto_term * context_term
