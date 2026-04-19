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
initial 11 domains; callers can add/override at runtime via :meth:`add` and
:meth:`refresh`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from skynet_vibe.exceptions import EmbeddingError, PrototypeNotFoundError

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

    def __init__(self, embedder: Embedder):
        self._embedder: Embedder = embedder
        self._prototypes: dict[str, DomainPrototype] = {}

    async def _embed_phrases(self, phrases: list[str]) -> list[list[float]]:
        return [await _embed_one(self._embedder, p) for p in phrases]

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
        return refreshed

    async def load_from_config(self, config: dict[str, list[str]]) -> None:
        """Bulk-load prototypes from a ``{name: [seed_phrases, ...]}`` dict."""
        for name, seeds in config.items():
            await self.add(name, list(seeds))

    async def load_defaults(self) -> None:
        """Load the packaged ``config/default_prototypes.yaml`` bundle.

        Convenience wrapper so callers can bootstrap 11 common domains with
        one line. Requires PyYAML (listed in the package dependencies).
        """
        from importlib import resources

        import yaml  # type: ignore[import-not-found]

        raw = resources.files("skynet_vibe").joinpath("config/default_prototypes.yaml").read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            raise ValueError("default_prototypes.yaml must be a mapping of domain -> seed phrases")
        await self.load_from_config({str(k): [str(x) for x in v] for k, v in data.items()})

    def remove(self, name: str) -> None:
        self._prototypes.pop(name, None)

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
