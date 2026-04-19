"""High-level orchestration entry point.

:class:`VibeEngine` ties together the embedder, the store, the prototype
registry, and the LLM client into a small surface:

* :meth:`absorb` -- capture a text signal from any source.
* :meth:`absorb_emoji` -- capture an emoji reaction.
* :meth:`suggest` -- score candidates against the weighted vibe cloud and
  LLM-rerank the top-N.
* :meth:`describe_current_vibe` -- narrative summary (LLM).
* :meth:`explain_signal` -- weight breakdown for a single signal (pure math).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from skynet_scoring import compute_decay_factor_logical

from skynet_vibe.affinity import cosine, signal_weight
from skynet_vibe.emoji import embed_emoji, phrase_for
from skynet_vibe.exceptions import EmbeddingError, PrototypeNotFoundError
from skynet_vibe.explain import describe_current_vibe as _describe_current_vibe
from skynet_vibe.explain import explain_signal as _explain_signal
from skynet_vibe.prototypes import PrototypeRegistry
from skynet_vibe.signals import FacetVectors, Source, VibeSignal
from skynet_vibe.store import VibeStore


def _decay_for(signal: VibeSignal, lambdas: dict[str, float] | None) -> float:
    """Compute logical-time decay for a VibeSignal from its extra_payload.

    Silence-safe by construction: reads ``missed_opportunities`` from
    the payload (store round-trips the root-level field into
    extra_payload), so a signal that never had a chance to be used
    keeps a decay factor of 1.0 no matter how much wall-clock time
    passes. See feedback_decay_logical_time for the rationale.
    """
    extra = signal.extra_payload or {}
    payload_like = {
        "missed_opportunities": int(extra.get("missed_opportunities", 0) or 0),
        "memory_class": extra.get("memory_class", "episodic"),
    }
    if "salience" in extra and extra["salience"] is not None:
        payload_like["salience"] = float(extra["salience"])
    if "compression_level" in extra and extra["compression_level"] is not None:
        payload_like["compression_level"] = int(extra["compression_level"])
    return compute_decay_factor_logical(
        payload_like,
        memory_class=payload_like["memory_class"],
        lambdas=lambdas,
    )


Embedder = Callable[[str], Awaitable[list[float]] | list[float]]
LLMClient = Callable[[str], Awaitable[str] | str]


async def _run(callable_, *args, **kwargs):
    """Await if callable returned a coroutine, otherwise return the value."""
    result = callable_(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result


async def _embed(embedder: Embedder, text: str) -> list[float]:
    result = embedder(text)
    if asyncio.iscoroutine(result):
        result = await result  # type: ignore[assignment]
    if not isinstance(result, list) or not result:
        raise EmbeddingError(f"embedder returned invalid vector for text: {text[:80]!r}")
    return list(result)  # type: ignore[return-value]


@dataclass
class SuggestResult:
    """Output of :meth:`VibeEngine.suggest`.

    ``candidate`` is the caller-supplied dict selected by the engine.
    ``reason`` is the LLM-written "why it fits now" explanation.
    ``rec_id`` is a stable id so downstream feedback can be linked back.
    ``top_contributing_signals`` is a list of ``(signal_id, weight)`` tuples
    for transparency.
    ``vibe_summary`` is a short LLM vibe description of the target.
    """

    candidate: Any
    reason: str
    rec_id: str
    top_contributing_signals: list[tuple[str, float]] = field(default_factory=list)
    vibe_summary: str = ""


class VibeEngine:
    """Top-level coordinator for the vibe signal system."""

    def __init__(
        self,
        store: VibeStore,
        prototypes: PrototypeRegistry,
        embedder: Embedder,
        llm_client: LLMClient,
        context_alpha: float = 0.5,
        decay_lambdas: dict[str, float] | None = None,
    ):
        self.store = store
        self.prototypes = prototypes
        self.embedder = embedder
        self.llm = llm_client
        self.context_alpha = context_alpha
        # Per-memory-class override for skynet_scoring logical decay
        # lambdas. None -> DEFAULT_LAMBDAS from skynet_scoring (identity
        # immortal, trait slow, episodic medium, raw fast).
        self.decay_lambdas = decay_lambdas

    # ------------------------------------------------------------------
    # Absorb

    async def absorb(
        self,
        *,
        text: str,
        source: Source,
        confidence: float = 1.0,
        linked_rec_id: str | None = None,
        context_text: str | None = None,
        user_state_text: str | None = None,
    ) -> VibeSignal:
        """Embed, wrap, and persist a new vibe signal."""
        content_vec = await _embed(self.embedder, text)
        context_vec = await _embed(self.embedder, context_text) if context_text else None
        user_state_vec = await _embed(self.embedder, user_state_text) if user_state_text else None
        signal = VibeSignal(
            id=VibeSignal.new_id(),
            text_raw=text,
            vectors=FacetVectors(content=content_vec, context=context_vec, user_state=user_state_vec),
            source=source,
            timestamp=VibeSignal.now(),
            confidence=float(confidence),
            linked_rec_id=linked_rec_id,
        )
        await self.store.put(signal)
        return signal

    async def absorb_emoji(
        self,
        *,
        emoji: str,
        source: Source,
        linked_rec_id: str | None = None,
    ) -> VibeSignal:
        """Embed an emoji's curated vibe phrase and store it as a reaction signal."""
        content_vec = await embed_emoji(emoji, self.embedder)
        if content_vec is None:
            raise EmbeddingError(f"emoji {emoji!r} not in curated corpus; cannot absorb")
        phrase = phrase_for(emoji) or emoji
        signal = VibeSignal(
            id=VibeSignal.new_id(),
            text_raw=f"{emoji} {phrase}",
            vectors=FacetVectors(content=content_vec),
            source=source,
            timestamp=VibeSignal.now(),
            confidence=1.0,
            linked_rec_id=linked_rec_id,
            extra_payload={"emoji": emoji, "vibe_phrase": phrase},
        )
        await self.store.put(signal)
        return signal

    # ------------------------------------------------------------------
    # Suggest

    async def _signal_pool(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        noise_floor: float = 0.2,
    ) -> list[VibeSignal]:
        return await self.store.search(query_vector=query_vector, top_k=top_k, noise_floor=noise_floor)

    async def _weighted_target(
        self,
        *,
        domain: str | None,
        context_text: str | None,
        pool_size: int,
    ) -> tuple[list[float], list[tuple[VibeSignal, float]]]:
        """Compute the weighted-sum target vector and return the scored pool.

        If ``context_text`` is absent, the pool is seeded from the prototype
        centroid (if ``domain`` provided) or a zero-fallback is used.
        """
        prototype_centroid: list[float] | None = None
        if domain is not None:
            try:
                proto = await self.prototypes.get(domain)
                prototype_centroid = proto.centroid
            except PrototypeNotFoundError:
                prototype_centroid = None

        context_vector: list[float] | None = None
        if context_text:
            context_vector = await _embed(self.embedder, context_text)

        # Seed query vector: prefer context, else prototype centroid.
        seed: list[float] | None = context_vector if context_vector is not None else prototype_centroid
        if seed is None:
            return [], []

        raw_signals = await self._signal_pool(query_vector=seed, top_k=pool_size)
        scored: list[tuple[VibeSignal, float]] = []
        dim = len(seed)
        weighted_sum = [0.0] * dim
        total_weight = 0.0
        for sig in raw_signals:
            decay = _decay_for(sig, self.decay_lambdas)
            w = signal_weight(
                sig,
                prototype_centroid=prototype_centroid,
                context_vector=context_vector,
                decay_factor=decay,
                context_alpha=self.context_alpha,
            )
            if w <= 0 or len(sig.vectors.content) != dim:
                continue
            scored.append((sig, w))
            for i, x in enumerate(sig.vectors.content):
                weighted_sum[i] += w * x
            total_weight += w

        if total_weight == 0:
            return list(seed), scored

        target = [x / total_weight for x in weighted_sum]
        if context_vector is not None:
            # Blend a slice of the context back in so very fresh intent is not drowned.
            blend = self.context_alpha
            target = [(1.0 - blend) * t + blend * c for t, c in zip(target, context_vector)]
        return target, scored

    async def suggest(
        self,
        *,
        candidates: list[dict[str, Any]],
        domain: str | None = None,
        context_text: str | None = None,
        top_k: int = 5,
    ) -> SuggestResult:
        """Rank ``candidates`` against the current weighted vibe cloud.

        Each candidate must provide either a ``vector`` or a ``text`` field.
        The engine scores all candidates by cosine against the weighted target,
        keeps the top ``top_k``, asks the LLM to pick one and narrate why, and
        returns a :class:`SuggestResult`.
        """
        if not candidates:
            raise ValueError("suggest() requires at least one candidate")

        target, pool = await self._weighted_target(domain=domain, context_text=context_text, pool_size=128)
        if not target:
            raise ValueError("cannot build target vector: no prototype centroid and no context_text")

        enriched: list[tuple[float, dict[str, Any]]] = []
        for cand in candidates:
            vec = cand.get("vector")
            if vec is None and cand.get("text"):
                vec = await _embed(self.embedder, cand["text"])
            if not vec:
                continue
            score = cosine(vec, target)
            enriched.append((score, cand))
        if not enriched:
            raise ValueError("no candidates carried a usable vector or text field")

        enriched.sort(key=lambda tup: tup[0], reverse=True)
        top_candidates = enriched[: max(1, top_k)]

        rec_id = VibeSignal.new_id()
        rerank_prompt = _build_rerank_prompt(top_candidates, domain=domain, context_text=context_text)
        llm_out = await _run(self.llm, rerank_prompt)
        choice_idx, reason = _parse_rerank_response(str(llm_out), default_n=len(top_candidates))
        chosen_score, chosen = top_candidates[choice_idx]

        vibe_summary = await _describe_current_vibe(pool, self.llm, domain=domain)
        top_signals_sorted = sorted(pool, key=lambda sw: sw[1], reverse=True)
        top_signals = [(s.id, round(w, 4)) for s, w in top_signals_sorted[:5]]

        # Logical-time decay bookkeeping. The pool (typically ~128 raw
        # signals that matched the seed cos > noise_floor) had a chance
        # to steer the recommendation. The top 5 contributors actually
        # shaped the target vector; treat everything BELOW them as a
        # "missed opportunity" and bump its counter. This is what makes
        # decay advance only when the system observes real traffic --
        # silence freezes the clock, matching feedback_decay_logical_time.
        top_ids = {s.id for s, _w in top_signals_sorted[:5]}
        missed_records = [
            {
                "id": s.id,
                "payload": {
                    "missed_opportunities": int((s.extra_payload or {}).get("missed_opportunities", 0) or 0),
                },
            }
            for s, _w in pool
            if s.id not in top_ids
        ]
        try:
            await self.store.bulk_increment_missed_opportunities(missed_records)
        except Exception:
            pass  # best-effort bookkeeping

        _ = chosen_score  # reserved for future diagnostic use
        return SuggestResult(
            candidate=chosen,
            reason=reason,
            rec_id=rec_id,
            top_contributing_signals=top_signals,
            vibe_summary=vibe_summary,
        )

    # ------------------------------------------------------------------
    # Describe / explain

    async def describe_current_vibe(self, domain: str | None = None, window_days: int = 14) -> str:
        """Render a narrative description of the user's current vibe.

        Uses the domain centroid as the seed query (if domain given) or scrolls
        a reasonable pool otherwise. Applies gradient weighting before handing
        to the LLM. ``window_days`` is passed through to the narrator as a
        human-readable label only -- decay itself is logical-time
        (missed_opportunities), not calendar-day.
        """
        prototype_centroid: list[float] | None = None
        if domain is not None:
            try:
                proto = await self.prototypes.get(domain)
                prototype_centroid = proto.centroid
            except PrototypeNotFoundError:
                prototype_centroid = None

        if prototype_centroid is None:
            # Fallback: describe nothing domain-specific
            return await _describe_current_vibe([], self.llm, domain=domain, window_days=window_days)

        raw_signals = await self._signal_pool(query_vector=prototype_centroid, top_k=128)
        scored: list[tuple[VibeSignal, float]] = []
        for sig in raw_signals:
            decay = _decay_for(sig, self.decay_lambdas)
            w = signal_weight(
                sig,
                prototype_centroid=prototype_centroid,
                context_vector=None,
                decay_factor=decay,
                context_alpha=self.context_alpha,
            )
            if w > 0:
                scored.append((sig, w))
        return await _describe_current_vibe(scored, self.llm, domain=domain, window_days=window_days)

    async def explain_signal(
        self,
        signal_id: str,
        domain: str | None = None,
        context_text: str | None = None,
    ) -> dict[str, Any]:
        """Return the full weight breakdown for ``signal_id``. Pure math."""
        signal = await self.store.get_required(signal_id)
        prototype = None
        if domain is not None:
            try:
                prototype = await self.prototypes.get(domain)
            except PrototypeNotFoundError:
                prototype = None
        context_vector: list[float] | None = None
        if context_text:
            context_vector = await _embed(self.embedder, context_text)
        decay = _decay_for(signal, self.decay_lambdas)
        return _explain_signal(
            signal,
            prototype=prototype,
            context_vector=context_vector,
            decay_factor=decay,
            context_alpha=self.context_alpha,
        )


# ----------------------------------------------------------------------
# Rerank prompt helpers


def _candidate_label(cand: dict[str, Any]) -> str:
    return str(cand.get("title") or cand.get("name") or cand.get("id") or cand.get("text", "")[:80] or "candidate")


def _build_rerank_prompt(
    top_candidates: list[tuple[float, dict[str, Any]]],
    *,
    domain: str | None,
    context_text: str | None,
) -> str:
    lines = []
    for idx, (score, cand) in enumerate(top_candidates):
        label = _candidate_label(cand)
        descr = cand.get("description") or cand.get("text") or ""
        lines.append(f"[{idx}] score={score:.3f} title={label!r} :: {descr[:200]}")
    domain_line = f"Domain: {domain}\n" if domain else ""
    context_line = f"Current context: {context_text}\n" if context_text else ""
    return (
        "You are reranking candidates for a personalised recommendation.\n"
        f"{domain_line}{context_line}"
        "Candidates (higher score = closer to the user's current weighted vibe):\n"
        + "\n".join(lines)
        + "\n\nRespond with JSON of the form: "
        '{"choice": <int index>, "reason": "<one sentence explaining why this fits now>"}. '
        "Keep the reason short, specific, and grounded in the listed descriptions."
    )


def _parse_rerank_response(raw: str, default_n: int) -> tuple[int, str]:
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start : end + 1])
            idx = int(data.get("choice", 0))
            if idx < 0 or idx >= default_n:
                idx = 0
            reason = str(data.get("reason", "")).strip() or "Top-scoring candidate for the current vibe."
            return idx, reason
    except (ValueError, json.JSONDecodeError):
        pass
    return 0, raw.strip() or "Top-scoring candidate for the current vibe."
