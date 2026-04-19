"""Diagnostic / narrative helpers.

* :func:`describe_current_vibe` -- LLM-rendered prose summary of a weighted
  signal pool. The caller passes in the retrieved signals and their weights;
  we assemble the prompt and hand off to the injected LLM.
* :func:`explain_signal` -- pure-math weight breakdown for a single signal.
  No LLM call. Used by the ``explain_signal`` MCP / diagnostic endpoint.

``decay_factor`` is passed in by the caller (computed upstream via
``skynet_scoring.compute_decay_factor_logical``) so every Skynet component
shares one logical-time decay formula instead of duplicating it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from skynet_vibe.affinity import SOURCE_TRUST, cosine
from skynet_vibe.prototypes import DomainPrototype
from skynet_vibe.signals import VibeSignal

LLMClient = Callable[[str], Awaitable[str] | str]


async def _call_llm(llm: LLMClient, prompt: str) -> str:
    result = llm(prompt)
    if asyncio.iscoroutine(result):
        result = await result  # type: ignore[assignment]
    return str(result).strip()


async def describe_current_vibe(
    signals_with_weights: list[tuple[VibeSignal, float]],
    llm: LLMClient,
    domain: str | None = None,
    window_days: int = 14,
) -> str:
    """Render a short natural-language description of the weighted signal pool.

    ``signals_with_weights`` is the pool already scored by the engine; we
    format the top entries into a prompt for the LLM. Deterministic fallback:
    if no signals are present we return a short canned string without calling
    the LLM. ``window_days`` is a human-readable label for the prompt only --
    the weights already reflect logical-time decay.
    """
    if not signals_with_weights:
        domain_hint = f" for {domain}" if domain else ""
        return f"No vibe signals in the last {window_days} days{domain_hint}."
    sorted_pool = sorted(signals_with_weights, key=lambda sw: sw[1], reverse=True)
    top = sorted_pool[: min(12, len(sorted_pool))]
    lines = []
    for signal, weight in top:
        lines.append(
            f"- [w={weight:.2f} src={signal.source.type} conf={signal.confidence:.2f}] {signal.text_raw[:160]}"
        )
    domain_hint = f" in the '{domain}' domain" if domain else ""
    prompt = (
        f"You are summarising the user's current vibe{domain_hint} from the last {window_days} days. "
        "The following signals are listed from highest to lowest weight; weight already combines "
        "logical-time decay, source trust, and domain alignment. Produce ONE short paragraph (<= 3 "
        "sentences) describing the overall vibe direction -- mood, tempo, texture, what is pulling "
        "strongest. No bullet points, no preamble.\n\n" + "\n".join(lines)
    )
    return await _call_llm(llm, prompt)


def explain_signal(
    signal: VibeSignal,
    *,
    prototype: DomainPrototype | None = None,
    context_vector: list[float] | None = None,
    decay_factor: float = 1.0,
    context_alpha: float = 0.5,
) -> dict[str, Any]:
    """Return a per-term breakdown of the weight formula for ``signal``.

    Useful for diagnostics (``explain_signal`` MCP tool / retrieval debug).
    Does NOT call the LLM. ``decay_factor`` must be pre-computed by the
    caller (typically via ``skynet_scoring.compute_decay_factor_logical``).
    """
    trust = SOURCE_TRUST.get(signal.source.type, 0.5)
    proto_cos = cosine(signal.vectors.content, prototype.centroid) if prototype is not None else None
    proto_term = max(0.0, proto_cos) if proto_cos is not None else 1.0
    if context_vector is not None:
        ctx_cos = cosine(signal.vectors.content, context_vector)
        context_term = 1.0 + context_alpha * ctx_cos
    else:
        ctx_cos = None
        context_term = 1.0
    decay = max(0.0, float(decay_factor))
    final = float(signal.confidence) * trust * decay * proto_term * context_term
    extra = signal.extra_payload or {}
    return {
        "id": signal.id,
        "text_raw": signal.text_raw,
        "source_type": signal.source.type,
        "timestamp": signal.timestamp.isoformat(),
        "confidence": float(signal.confidence),
        "source_trust": trust,
        "decay_factor": decay,
        "missed_opportunities": int(extra.get("missed_opportunities", 0) or 0),
        "memory_class": extra.get("memory_class", "episodic"),
        "domain": prototype.name if prototype is not None else None,
        "prototype_cosine": proto_cos,
        "prototype_term": proto_term,
        "context_cosine": ctx_cos,
        "context_term": context_term,
        "context_alpha": context_alpha,
        "final_weight": final,
    }
