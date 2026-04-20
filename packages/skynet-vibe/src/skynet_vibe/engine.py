"""High-level orchestration entry point.

:class:`VibeEngine` ties together the embedder, the store, the prototype
registry, and the LLM client into a small surface:

* :meth:`match` -- entropy-gated prototype classification (v2 API).
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
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from skynet_scoring import compute_decay_factor_logical

from skynet_vibe.affinity import cosine, signal_weight
from skynet_vibe.emoji import embed_emoji, phrase_for
from skynet_vibe.exceptions import EmbeddingError, PrototypeNotFoundError, PrototypeWarmingUpError
from skynet_vibe.explain import describe_current_vibe as _describe_current_vibe
from skynet_vibe.explain import explain_signal as _explain_signal
from skynet_vibe.prototypes import PrototypeRegistry
from skynet_vibe.signals import FacetVectors, Source, VibeSignal
from skynet_vibe.store import VibeStore

# Source types that bypass the novelty gate — intentional records about
# specific works are always worth keeping regardless of similarity to
# existing signals.
_REVIEW_SOURCE_TYPES: frozenset[str] = frozenset({"music_review", "movie_review"})

# novelty_weight below this → too close to an existing signal → skip.
# Simple config value; not adaptive on purpose (feedback_gradient_weights_not_gates:
# the *threshold* is a config knob, the *weight itself* is the adaptive metric).
_MERGE_WEIGHT_THRESHOLD: float = 0.15


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


@dataclass
class ProcessResult:
    """Output of :meth:`VibeEngine.process` — novelty-gated signal capture.

    ``stored`` is True iff the signal was upserted into Qdrant.
    ``novelty_weight`` = novelty_score / rolling_mean: measures how unusual
    this signal is relative to recent insertions. Values > 1.0 are more
    novel than average; < 1.0 are more redundant. This IS the adaptive
    metric — not a threshold (feedback_gradient_weights_not_gates).
    ``nearest_score`` is the max cosine to any existing signal (0 on empty
    collection, 1 on exact duplicate).
    ``signal_id`` is populated iff ``stored=True``.
    """

    stored: bool
    novelty_weight: float
    nearest_score: float
    source_type: str = "emotional"
    signal_id: str | None = None


class RollingMean:
    """In-process rolling mean of recent novelty scores.

    Provides the denominator for ``novelty_weight = novelty_score / mean``.
    Returns ``cold_start_mean`` until at least ``min_samples`` entries
    have been pushed — avoids division-by-near-zero on a fresh pod while
    keeping the cold-start behaviour predictable (everything novel).

    Single-replica pods can keep this in memory; the rolling window
    naturally resets on pod restart which is safe — the collection itself
    is the durable state.
    """

    def __init__(
        self,
        window: int = 200,
        cold_start_mean: float = 0.5,
        min_samples: int = 10,
    ) -> None:
        self._window = window
        self._cold_start = cold_start_mean
        self._min = min_samples
        self._buf: list[float] = []

    def push(self, value: float) -> None:
        self._buf.append(float(value))
        if len(self._buf) > self._window:
            del self._buf[0]

    def mean(self) -> float:
        if len(self._buf) < self._min:
            return self._cold_start
        return sum(self._buf) / len(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class MatchResult:
    """Output of :meth:`VibeEngine.match` -- entropy-gated classification.

    All fields are always populated; callers can keep or discard the
    signal based on ``accepted`` alone, but ``entropy_bits`` /
    ``softmax_probs`` / ``cosines`` are preserved for debug payloads
    (feedback_adaptive_not_hardcoded: we never hardcode a score cutoff,
    the confidence IS the entropy-normalised number).

    * ``winner`` -- name of the prototype with the highest softmax prob.
    * ``confidence`` -- ``1 - H/H_max`` where ``H`` is Shannon entropy
      of the softmax distribution (in bits) and ``H_max = log2(N)``.
      1.0 = completely peaked on one prototype; 0.0 = uniform.
    * ``entropy_bits`` -- raw Shannon entropy ``H`` of the distribution.
    * ``softmax_probs`` -- full per-prototype probability mass.
    * ``cosines`` -- raw pre-softmax cosines (useful for spotting a
      non-semantic input where all cosines are ~0).
    * ``accepted`` -- ``H <= H_max / 2`` gate. Below half of the maximum
      possible entropy the distribution is considered "peaked enough"
      to persist; above, the winner is noise and the caller should drop.
    """

    winner: str
    confidence: float
    entropy_bits: float
    softmax_probs: dict[str, float]
    cosines: dict[str, float]
    accepted: bool


class VibeEngine:
    """Top-level coordinator for the vibe signal system."""

    def __init__(
        self,
        store: VibeStore,
        prototypes: PrototypeRegistry | None,
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
        extra_payload: dict | None = None,
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
            extra_payload=dict(extra_payload) if extra_payload else {},
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
    # Process (v3: novelty-gated, history-based)

    async def process(
        self,
        *,
        text: str,
        source: Source,
        source_type: str = "emotional",
        rolling_mean: RollingMean | None = None,
        extra_payload: dict | None = None,
    ) -> ProcessResult:
        """Embed text, assess novelty against existing signals, conditionally store.

        Acceptance rule (feedback_gradient_weights_not_gates):
        ``novelty_weight = novelty_score / rolling_mean`` measures novelty
        relative to recent insertions, not against an absolute threshold.
        Signals with ``novelty_weight >= _MERGE_WEIGHT_THRESHOLD`` are stored;
        below it, they are too close to existing signals to add information.

        Review signals (``music_review``, ``movie_review``) bypass the gate —
        intentional records about specific works are always worth keeping.

        All signals live in the same Qdrant collection and share the same
        embedding space, so music reviews, movie reviews, and emotional signals
        are all neighbours by semantic proximity. Relations emerge from Qdrant
        nearest-neighbour search at query time — nothing is pre-stored.
        """
        if len(text.strip()) < 10:
            return ProcessResult(
                stored=False, novelty_weight=0.0, nearest_score=1.0, source_type=source_type,
            )

        vec = await _embed(self.embedder, text)

        # True nearest-neighbour: no noise floor so we always get the closest
        # existing signal even if similarity is low (empty collection → 0.0).
        neighbors = await self.store.search(vec, top_k=3, noise_floor=0.0)
        if neighbors:
            nearest_score = max(cosine(vec, n.vectors.content) for n in neighbors)
        else:
            nearest_score = 0.0

        novelty_score = 1.0 - nearest_score
        mean = rolling_mean.mean() if rolling_mean is not None else 0.5
        novelty_weight = novelty_score / mean if mean > 0.0 else novelty_score

        should_store = (source_type in _REVIEW_SOURCE_TYPES) or (novelty_weight >= _MERGE_WEIGHT_THRESHOLD)

        if not should_store:
            return ProcessResult(
                stored=False,
                novelty_weight=float(novelty_weight),
                nearest_score=float(nearest_score),
                source_type=source_type,
            )

        extra = dict(extra_payload or {})
        extra["novelty_weight"] = round(float(novelty_weight), 4)
        extra["source_type"] = source_type

        signal = await self.absorb(
            text=text,
            source=source,
            confidence=min(1.0, max(0.0, float(novelty_weight))),
            extra_payload=extra,
        )

        if rolling_mean is not None:
            rolling_mean.push(novelty_score)

        return ProcessResult(
            stored=True,
            novelty_weight=float(novelty_weight),
            nearest_score=float(nearest_score),
            source_type=source_type,
            signal_id=signal.id,
        )

    # ------------------------------------------------------------------
    # Match (v2: entropy-gated classification)

    async def match(
        self,
        text: str,
        *,
        embedder: Embedder | None = None,
    ) -> MatchResult:
        """Classify ``text`` against every registered prototype.

        Pipeline:

        1. Embed ``text`` (via ``embedder`` if supplied, else ``self.embedder``).
        2. Compute cosine against every prototype centroid.
        3. Softmax over the cosine vector at temperature
           ``self.prototypes.tau`` (calibrated once at warmup).
        4. Shannon entropy of the softmax distribution, in bits.
        5. Accept iff ``H <= H_max / 2`` where ``H_max = log2(N)``.

        The ``τ/H/accepted`` trio replaces the v1 absolute-cosine threshold
        (``min_score``) + top-2 margin + rolling-window dead code --- see
        feedback_adaptive_not_hardcoded. The only tunable "knob" that
        survives is the warmup target mean entropy (``log2(N)/4``), which
        is a ratio of the theoretical maximum, not a magic constant.

        Raises :class:`PrototypeWarmingUpError` if the registry hasn't
        finished embedding its prototypes yet. Raises
        :class:`EmbeddingError` if the embedder returns a bad vector or
        there are zero registered prototypes.
        """
        if self.prototypes is None:
            raise EmbeddingError("match() requires a PrototypeRegistry; use process() for the novelty-gated path")
        if not self.prototypes.ready:
            raise PrototypeWarmingUpError("prototype warmup in progress; match() not available yet")

        protos = self.prototypes.all()
        if not protos:
            raise EmbeddingError("match() called with zero registered prototypes")

        active_embedder: Embedder = embedder if embedder is not None else self.embedder
        vec = await _embed(active_embedder, text)

        tau = float(getattr(self.prototypes, "tau", 1.0) or 1.0)

        # Compute cosines and softmax logits in one pass.
        names: list[str] = []
        cosines: list[float] = []
        for p in protos:
            names.append(p.name)
            cosines.append(cosine(vec, p.centroid))

        # Numerically stable softmax at temperature tau.
        logits = [c / tau for c in cosines]
        max_logit = max(logits)
        exps = [math.exp(lg - max_logit) for lg in logits]
        total = sum(exps)
        if total <= 0.0:
            # Degenerate: treat as uniform.
            probs = [1.0 / len(protos)] * len(protos)
        else:
            probs = [e / total for e in exps]

        # Shannon entropy in bits (base 2).
        h_bits = 0.0
        for p_i in probs:
            if p_i > 1e-12:
                h_bits -= p_i * math.log2(p_i)
        h_max = math.log2(len(protos)) if len(protos) > 1 else 1.0
        confidence = 1.0 - (h_bits / h_max) if h_max > 0 else 0.0
        # Clamp against float drift.
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0

        winner_idx = max(range(len(probs)), key=lambda i: probs[i])
        accepted = h_bits <= (h_max / 2.0)

        return MatchResult(
            winner=names[winner_idx],
            confidence=float(confidence),
            entropy_bits=float(h_bits),
            softmax_probs={name: float(p) for name, p in zip(names, probs, strict=True)},
            cosines={name: float(c) for name, c in zip(names, cosines, strict=True)},
            accepted=bool(accepted),
        )

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

        if not self.prototypes.ready:
            raise PrototypeWarmingUpError(f"prototype warmup in progress; target domain={domain!r} not yet available")

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

    async def vibe_pool_stats(
        self,
        *,
        domain: str | None = None,
        window_days: int = 14,
    ) -> dict[str, Any]:
        """Return vibe pool metadata for diagnostic endpoints.

        Wraps :meth:`VibeStore.pool_stats` (which always applies the
        ``category == sub_category`` filter) so ``/vibe/status`` can
        report the live pool size without poking at the store's private
        client. ``domain`` and ``window_days`` are currently passed
        through as metadata only -- signal filtering by domain happens
        at retrieval time via prototype cosine, not at the payload
        layer, and decay is logical-time (not wall-clock) per project
        convention. See ``feedback_decay_logical_time``.
        """
        stats = await self.store.pool_stats()
        stats.setdefault("available", True)
        stats["domain"] = domain
        stats["window_days"] = window_days
        return stats

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
