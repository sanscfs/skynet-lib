"""Domain prototypes -- the heart of the ``domain-as-prototype`` approach.

A prototype is a named bundle of seed phrases and an L2-normalized centroid
vector (mean of the seed-phrase embeddings). Domain membership of any signal
is a run-time cosine similarity against these centroids; *signals do not
carry a domain label*.

Persistence
-----------
The registry is **in-memory** with optional **YAML file persistence**. This
was chosen over a dedicated Qdrant collection because:

* Prototypes are small (~tens of domains, each = 1 short centroid).
* They change slowly (add/refresh is a deliberate operation, not traffic).
* Loading from YAML at start-up gives reproducible, version-controllable
  domain definitions that travel with the codebase.
* No Qdrant round-trip for what ends up being a hot in-memory cosine loop.

The ``config/default_prototypes.yaml`` shipped with the package defines the
initial 11 domains, and the v2 bank in ``config/prototypes_v2.yaml`` ships
50 prototypes × 4 training phrases on 5 axes (used by ``match()``).
Callers can add/override at runtime via :meth:`add` and :meth:`refresh`.

Temperature calibration
-----------------------
The v2 bank introduces a registry-level ``tau`` attribute used by
``VibeEngine.match()`` to soft-max over prototype cosines. ``tau`` is
calibrated exactly once -- at the tail of ``start_warmup()`` -- by
binary-searching τ ∈ [0.01, 2.0] until the MEAN Shannon entropy across
all training phrases is approximately ``log2(N) / 4`` (i.e. the
distribution is well-peaked but not degenerate). Nothing at runtime
retunes τ. The only legitimate triggers for a recalibration are:

* Pod restart / new warmup
* Prototype set change (``refresh``, ``add`` with ``recalibrate=True``)
* Embedding model change (manually call :meth:`calibrate_tau`)

See feedback_adaptive_not_hardcoded: τ is a ratio (target H / H_max)
rather than a magic constant the operator has to guess.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from skynet_vibe.exceptions import EmbeddingError, PrototypeNotFoundError

logger = logging.getLogger(__name__)

# Embedder protocol: accepts a string, returns a list[float] (sync or awaitable).
Embedder = Callable[[str], Awaitable[list[float]] | list[float]]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _mean_centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise EmbeddingError("cannot compute centroid from zero vectors")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise EmbeddingError("seed-phrase embeddings have inconsistent dimensions")
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    n = len(vectors)
    return _l2_normalize([x / n for x in acc])


async def _embed_one(embedder: Embedder, text: str) -> list[float]:
    result = embedder(text)
    if asyncio.iscoroutine(result):
        result = await result  # type: ignore[assignment]
    if not isinstance(result, list) or not result:
        raise EmbeddingError(f"embedder returned invalid vector for phrase: {text!r}")
    return list(result)  # type: ignore[return-value]


@dataclass
class DomainPrototype:
    """A named domain with seed phrases and a centroid vector."""

    name: str
    seed_phrases: list[str]
    centroid: list[float]
    last_refreshed: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed_phrases": list(self.seed_phrases),
            "centroid": list(self.centroid),
            "last_refreshed": self.last_refreshed.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainPrototype:
        ts = data["last_refreshed"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            name=data["name"],
            seed_phrases=list(data["seed_phrases"]),
            centroid=list(data["centroid"]),
            last_refreshed=ts,
        )


class PrototypeRegistry:
    """In-memory registry of named :class:`DomainPrototype` objects.

    The registry depends on an injected embedder (async or sync callable from
    ``str`` to ``list[float]``) so that seed-phrase embeddings are produced
    by whichever pipeline the caller has configured (e.g. ``skynet-embedding``).
    """

    # Default τ before calibration. 1.0 is a reasonable starting point
    # for cosines in [-1, 1]; after calibration this is typically
    # 0.05 - 0.5 for a well-separated prototype bank.
    DEFAULT_TAU: float = 1.0

    # Target mean-entropy ratio for τ calibration. log2(N)/4 puts
    # training phrases in the "moderately peaked" regime: well-chosen
    # training phrases should resolve to their own prototype without
    # collapsing the distribution to a delta function (which would make
    # the entropy gate useless on real inputs that are never as clean
    # as the training corpus).
    TARGET_ENTROPY_RATIO: float = 1.0 / 4.0

    # Binary-search bounds for τ. 0.01 is tight enough to saturate any
    # reasonable cosine spread; 2.0 is loose enough that the whole set
    # approaches uniform.
    TAU_MIN: float = 0.01
    TAU_MAX: float = 2.0

    def __init__(self, embedder: Embedder):
        self._embedder: Embedder = embedder
        self._prototypes: dict[str, DomainPrototype] = {}
        self._ready: asyncio.Event = asyncio.Event()
        self._warmup_task: asyncio.Task[None] | None = None
        # τ and the training-phrase embeddings are populated by
        # calibrate_tau() (called from _warmup). Until then match()
        # should refuse via the `ready` gate.
        self.tau: float = self.DEFAULT_TAU
        # Per-prototype training-phrase embeddings, kept in memory so
        # calibrate_tau can rerun deterministically without re-embedding.
        self._training_embeddings: dict[str, list[list[float]]] = {}

    # ------------------------------------------------------------------
    # Warmup lifecycle

    def start_warmup(self) -> None:
        """Non-blocking. Schedules background prototype embedding. Idempotent."""
        if self._warmup_task is None or self._warmup_task.done():
            self._warmup_task = asyncio.create_task(self._warmup())

    async def _warmup(self) -> None:
        try:
            await self._load_defaults_impl()
            # τ calibration is part of warmup -- once the prototypes are
            # embedded we binary-search τ on the training phrases so the
            # match() gate is well-conditioned on the first real event.
            # If calibration fails for some reason (empty bank, degenerate
            # embedder) we log and fall back to DEFAULT_TAU; match() will
            # still work, it just won't be optimally peaked.
            try:
                self.calibrate_tau()
            except Exception:
                logger.exception("tau calibration failed; falling back to default")
                self.tau = self.DEFAULT_TAU
            self._ready.set()
        except Exception:
            logger.exception("prototype warmup failed")
            raise

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    async def wait_ready(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _embed_phrases(self, phrases: list[str]) -> list[list[float]]:
        return list(await asyncio.gather(*[_embed_one(self._embedder, p) for p in phrases]))

    async def get(self, name: str) -> DomainPrototype:
        try:
            return self._prototypes[name]
        except KeyError as exc:
            raise PrototypeNotFoundError(f"domain prototype {name!r} not registered") from exc

    def get_sync(self, name: str) -> DomainPrototype:
        """Non-async lookup (no I/O). Raises if missing."""
        try:
            return self._prototypes[name]
        except KeyError as exc:
            raise PrototypeNotFoundError(f"domain prototype {name!r} not registered") from exc

    def all(self) -> list[DomainPrototype]:
        return list(self._prototypes.values())

    def names(self) -> list[str]:
        return list(self._prototypes.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._prototypes

    async def add(self, name: str, seed_phrases: list[str]) -> DomainPrototype:
        if not seed_phrases:
            raise ValueError(f"prototype {name!r} needs at least one seed phrase")
        vectors = await self._embed_phrases(seed_phrases)
        proto = DomainPrototype(
            name=name,
            seed_phrases=list(seed_phrases),
            centroid=_mean_centroid(vectors),
            last_refreshed=datetime.now(timezone.utc),
        )
        self._prototypes[name] = proto
        # Cache per-phrase embeddings for τ calibration. Calibration
        # reads this map directly so it never re-embeds -- important
        # because re-embedding 50×4 = 200 phrases through a real
        # embedder during a binary search would be wasteful.
        self._training_embeddings[name] = [list(v) for v in vectors]
        return proto

    async def refresh(self, name: str, new_seed_phrases: list[str] | None = None) -> DomainPrototype:
        """Recompute the centroid from seed phrases.

        If ``new_seed_phrases`` is supplied, it replaces the existing seeds;
        otherwise the existing seed list is re-embedded (useful after the
        embedding model has been upgraded).
        """
        existing = await self.get(name)
        phrases = list(new_seed_phrases) if new_seed_phrases is not None else list(existing.seed_phrases)
        if not phrases:
            raise ValueError(f"prototype {name!r} refresh requires at least one seed phrase")
        vectors = await self._embed_phrases(phrases)
        refreshed = DomainPrototype(
            name=name,
            seed_phrases=phrases,
            centroid=_mean_centroid(vectors),
            last_refreshed=datetime.now(timezone.utc),
        )
        self._prototypes[name] = refreshed
        self._training_embeddings[name] = [list(v) for v in vectors]
        return refreshed

    async def load_from_config(self, config: dict[str, list[str]]) -> None:
        """Bulk-load prototypes from a ``{name: [seed_phrases, ...]}`` dict."""
        await asyncio.gather(*[self.add(name, list(seeds)) for name, seeds in config.items()])

    async def _load_defaults_impl(self) -> None:
        """Load the packaged default prototype bundle.

        v2.0 ships ``config/prototypes_v2.yaml`` -- 50 prototypes across
        5 axes, used by entropy-gated ``VibeEngine.match()``. If
        the v2 file is present we prefer it; otherwise we fall back to
        the legacy ``default_prototypes.yaml`` (11 domains) so old
        deployments keep working until their image is bumped.

        Requires PyYAML (listed in the package dependencies).
        """
        from importlib import resources

        import yaml  # type: ignore[import-not-found]

        base = resources.files("skynet_vibe")

        # Prefer v2 bank.
        v2_path = base.joinpath("config/prototypes_v2.yaml")
        if v2_path.is_file():
            raw = v2_path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw) or {}
            if not isinstance(data, dict):
                raise ValueError("prototypes_v2.yaml must be a mapping")
            # v2 schema: top-level ``prototypes`` key, value is
            # name->list[phrases]. ``axes`` is metadata for humans/UI and
            # is deliberately ignored at load time.
            protos = data.get("prototypes")
            if not isinstance(protos, dict):
                raise ValueError("prototypes_v2.yaml missing 'prototypes' mapping")
            await self.load_from_config({str(k): [str(x) for x in v] for k, v in protos.items()})
            return

        # Fallback: legacy flat layout.
        raw = base.joinpath("config/default_prototypes.yaml").read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            raise ValueError("default_prototypes.yaml must be a mapping of domain -> seed phrases")
        await self.load_from_config({str(k): [str(x) for x in v] for k, v in data.items()})

    async def load_defaults(self) -> None:
        """Backwards-compat: start warmup + await its completion.

        Prefer :meth:`start_warmup` + :meth:`wait_ready` (or checking
        :attr:`ready`) in new code so service startup stays non-blocking.
        """
        self.start_warmup()
        assert self._warmup_task is not None  # set by start_warmup
        await self._warmup_task

    def remove(self, name: str) -> None:
        self._prototypes.pop(name, None)
        self._training_embeddings.pop(name, None)

    # ------------------------------------------------------------------
    # τ calibration

    def calibrate_tau(
        self,
        *,
        target_ratio: float | None = None,
        tau_min: float | None = None,
        tau_max: float | None = None,
        max_iterations: int = 40,
    ) -> float:
        """Binary-search τ so mean training-phrase entropy ≈ target.

        Pure math, no I/O: uses the cached ``_training_embeddings`` and
        the current ``_prototypes`` centroids, so it can be re-run after
        the prototype set changes (e.g. ``refresh``) without touching
        the embedder again.

        Returns the calibrated τ and stores it on ``self.tau``. Raises
        :class:`EmbeddingError` if there are no training embeddings or
        no prototypes registered.
        """
        target_ratio = target_ratio if target_ratio is not None else self.TARGET_ENTROPY_RATIO
        tau_low = tau_min if tau_min is not None else self.TAU_MIN
        tau_high = tau_max if tau_max is not None else self.TAU_MAX

        protos = list(self._prototypes.values())
        if not protos:
            raise EmbeddingError("calibrate_tau: no prototypes registered")
        if not self._training_embeddings:
            raise EmbeddingError("calibrate_tau: no training embeddings cached")

        n = len(protos)
        h_max = math.log2(n) if n > 1 else 1.0
        target_h = target_ratio * h_max

        # Flatten training phrases across all prototypes -- every cached
        # embedding contributes one entropy observation.
        training_vectors: list[list[float]] = []
        for name in self._prototypes.keys():
            training_vectors.extend(self._training_embeddings.get(name, []))
        if not training_vectors:
            raise EmbeddingError("calibrate_tau: cached training embeddings empty")

        # Precompute the (N × M) cosine matrix once -- centroids fixed,
        # training vectors fixed, τ just rescales the logits.
        # cos_matrix[i][j] = cos(training_vector_i, prototype_j.centroid)
        from skynet_vibe.affinity import cosine as _cos

        cos_matrix: list[list[float]] = [[_cos(tv, p.centroid) for p in protos] for tv in training_vectors]

        def _mean_entropy(tau: float) -> float:
            if tau <= 0.0:
                tau = 1e-6
            total_h = 0.0
            for row in cos_matrix:
                logits = [c / tau for c in row]
                max_l = max(logits)
                exps = [math.exp(lg - max_l) for lg in logits]
                s = sum(exps)
                if s <= 0.0:
                    total_h += h_max
                    continue
                probs = [e / s for e in exps]
                h = 0.0
                for p_i in probs:
                    if p_i > 1e-12:
                        h -= p_i * math.log2(p_i)
                total_h += h
            return total_h / len(cos_matrix)

        # Mean entropy is monotonically increasing in τ: as τ → 0 softmax
        # collapses to a delta (H → 0), as τ → ∞ it flattens (H → H_max).
        # Standard binary search.
        lo, hi = tau_low, tau_high
        for _ in range(max_iterations):
            mid = (lo + hi) / 2.0
            h_mid = _mean_entropy(mid)
            if h_mid < target_h:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-5:
                break
        calibrated = (lo + hi) / 2.0
        self.tau = float(calibrated)
        logger.debug(
            "tau calibrated: tau=%.5f mean_entropy=%.4f target=%.4f H_max=%.4f n_prototypes=%d n_phrases=%d",
            self.tau,
            _mean_entropy(self.tau),
            target_h,
            h_max,
            n,
            len(cos_matrix),
        )
        return self.tau

    # ------------------------------------------------------------------
    # Serialization (optional, for cache/debug)

    def to_dict(self) -> dict[str, Any]:
        return {name: proto.to_dict() for name, proto in self._prototypes.items()}

    def load_precomputed(self, data: dict[str, Any]) -> None:
        """Load already-computed centroids (skip embedding). Useful for tests
        and for reloading from a persisted cache."""
        for name, payload in data.items():
            full_payload = payload if "name" in payload else {**payload, "name": name}
            self._prototypes[name] = DomainPrototype.from_dict(full_payload)
