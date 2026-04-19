"""Qdrant-backed persistence for :class:`VibeSignal` objects.

Storage layout
--------------
* Collection: defaults to ``user_profile_raw`` (existing Skynet convention).
  Callers can override via the constructor to keep the package reusable.
* Payload field ``category`` is set to ``vibe_signal`` (configurable) so
  these records coexist with legacy ``cinema_preferences`` /
  ``music_preferences`` / etc. without collision.
* The primary Qdrant ``vector`` is the ``content`` facet -- this is what
  existing search tooling already expects.
* ``context`` and ``user_state`` facet vectors are stored as payload arrays
  (``vector_context``, ``vector_user_state``), preserving the full facet
  bundle without requiring multi-vector Qdrant collections.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from skynet_vibe.exceptions import SignalNotFoundError
from skynet_vibe.signals import FacetVectors, Source, VibeSignal


class _AsyncQdrantLike(Protocol):
    """Duck-typed subset of skynet_qdrant.AsyncQdrantClient used here."""

    async def upsert(self, collection: str, points: list[dict]) -> dict: ...
    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 5,
        *,
        filter: dict | None = None,
        with_payload: bool = True,
        score_threshold: float | None = None,
    ) -> list[dict]: ...
    async def get_point(self, collection: str, point_id, *, with_vector: bool = False) -> dict | None: ...
    async def set_payload(self, collection: str, point_ids: list, payload: dict) -> dict: ...


def _payload_from_signal(signal: VibeSignal, sub_category: str) -> dict[str, Any]:
    extra = dict(signal.extra_payload)
    # Root-level decay-relevant fields. They're promoted out of
    # extra_payload so Qdrant payload indexes (if any) and the shared
    # skynet_scoring.compute_decay_factor_logical(payload) helper can
    # find them without nesting lookups. The mirror in _signal_from_point
    # lifts them back into extra_payload so VibeSignal round-trips.
    missed = int(extra.pop("missed_opportunities", 0) or 0)
    memory_class = extra.pop("memory_class", "episodic")
    salience = extra.pop("salience", None)
    compression_level = extra.pop("compression_level", None)
    payload: dict[str, Any] = {
        "category": sub_category,
        "text_raw": signal.text_raw,
        "source": signal.source.to_dict(),
        "confidence": float(signal.confidence),
        "timestamp": signal.timestamp.isoformat(),
        "linked_rec_id": signal.linked_rec_id,
        "vector_context": list(signal.vectors.context) if signal.vectors.context is not None else None,
        "vector_user_state": list(signal.vectors.user_state) if signal.vectors.user_state is not None else None,
        "extra_payload": extra,
        "missed_opportunities": max(0, missed),
        "memory_class": memory_class,
    }
    if salience is not None:
        payload["salience"] = float(salience)
    if compression_level is not None:
        payload["compression_level"] = int(compression_level)
    return payload


def _signal_from_point(point: dict[str, Any]) -> VibeSignal:
    payload = point.get("payload", {}) or {}
    content_vec = point.get("vector") or payload.get("vector_content") or []
    if isinstance(content_vec, dict):
        # Some Qdrant versions return {"default": [...]}; grab the first one.
        content_vec = next(iter(content_vec.values()))
    vectors = FacetVectors(
        content=list(content_vec),
        context=list(payload["vector_context"]) if payload.get("vector_context") is not None else None,
        user_state=list(payload["vector_user_state"]) if payload.get("vector_user_state") is not None else None,
    )
    ts_raw = payload.get("timestamp")
    if isinstance(ts_raw, datetime):
        ts = ts_raw
    elif isinstance(ts_raw, str):
        ts = datetime.fromisoformat(ts_raw)
    else:
        ts = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    source_raw = payload.get("source") or {"type": "implicit"}
    # Lift root-level decay-relevant fields back into extra_payload so
    # callers (engine, explain) can feed them directly into
    # skynet_scoring.compute_decay_factor_logical(payload_like_dict)
    # without another Qdrant round-trip. Mirror of _payload_from_signal.
    extra = dict(payload.get("extra_payload", {}))
    if "missed_opportunities" in payload:
        extra["missed_opportunities"] = int(payload.get("missed_opportunities") or 0)
    if "memory_class" in payload:
        extra["memory_class"] = payload.get("memory_class")
    if "salience" in payload:
        extra["salience"] = payload.get("salience")
    if "compression_level" in payload:
        extra["compression_level"] = payload.get("compression_level")
    return VibeSignal(
        id=str(point.get("id")),
        text_raw=payload.get("text_raw", ""),
        vectors=vectors,
        source=Source.from_dict(source_raw),
        timestamp=ts,
        confidence=float(payload.get("confidence", 1.0)),
        linked_rec_id=payload.get("linked_rec_id"),
        extra_payload=extra,
    )


class VibeStore:
    """Async Qdrant wrapper for persisting and retrieving :class:`VibeSignal`.

    Parameters
    ----------
    qdrant_client:
        An instance that looks like ``skynet_qdrant.AsyncQdrantClient``
        (duck-typed: we call ``upsert``, ``search``, ``get_point``,
        ``set_payload`` only).
    collection:
        Qdrant collection name. Defaults to ``user_profile_raw``.
    sub_category:
        Value used for the payload ``category`` field to distinguish vibe
        signals from other records in the collection. Defaults to
        ``vibe_signal``.
    """

    def __init__(
        self,
        qdrant_client: _AsyncQdrantLike,
        collection: str = "user_profile_raw",
        sub_category: str = "vibe_signal",
    ):
        self.qdrant = qdrant_client
        self.collection = collection
        self.sub_category = sub_category

    # ------------------------------------------------------------------
    # CRUD-ish operations

    async def put(self, signal: VibeSignal) -> None:
        """Upsert a signal into the Qdrant collection."""
        point = {
            "id": signal.id,
            "vector": list(signal.vectors.content),
            "payload": _payload_from_signal(signal, self.sub_category),
        }
        await self.qdrant.upsert(self.collection, [point])

    async def get(self, signal_id: str) -> VibeSignal | None:
        point = await self.qdrant.get_point(self.collection, signal_id, with_vector=True)
        if not point:
            return None
        return _signal_from_point(point)

    async def get_required(self, signal_id: str) -> VibeSignal:
        signal = await self.get(signal_id)
        if signal is None:
            raise SignalNotFoundError(f"vibe signal {signal_id!r} not found in {self.collection}")
        return signal

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 50,
        noise_floor: float = 0.3,
        filter_: dict | None = None,
    ) -> list[VibeSignal]:
        """Cosine-search the collection for vibe signals near ``query_vector``.

        A ``category == sub_category`` filter is always applied; any additional
        filter in ``filter_`` is merged into it.
        """
        must: list[dict] = [{"key": "category", "match": {"value": self.sub_category}}]
        if filter_:
            # If caller passed a pre-built filter dict, assume it's Qdrant
            # native and merge its ``must`` clauses.
            if "must" in filter_ and isinstance(filter_["must"], list):
                must.extend(filter_["must"])
            elif filter_:
                # Raw {key: value} shortcut
                for key, value in filter_.items():
                    if key in ("must", "should", "must_not"):
                        continue
                    must.append({"key": key, "match": {"value": value}})
        final_filter = {"must": must}
        raw = await self.qdrant.search(
            self.collection,
            vector=list(query_vector),
            limit=top_k,
            filter=final_filter,
            with_payload=True,
            score_threshold=noise_floor,
        )
        out: list[VibeSignal] = []
        for point in raw:
            try:
                out.append(_signal_from_point(point))
            except Exception:
                # Skip malformed records rather than failing the whole retrieval.
                continue
        return out

    async def bulk_increment_missed_opportunities(
        self,
        records: list[dict[str, Any]],
        delta: int = 1,
    ) -> None:
        """Bump ``missed_opportunities`` on matched-but-unused signals.

        Mirrors ``skynet_identity.modules.decay.bulk_update_missed_opportunities``.
        Each record must carry ``id`` and a ``payload`` dict with the
        current ``missed_opportunities`` value (typical shape: a search
        hit from Qdrant). We group ids by the target value so one
        ``set_payload`` call covers every id reaching the same counter --
        usually shrinks O(N) writes to O(distinct-current-values).

        Best-effort: swallows per-bucket exceptions. This is background
        bookkeeping -- losing an increment is acceptable, surfacing an
        exception on the hot recommendation path is not.
        """
        if not records:
            return
        by_new_value: dict[int, list[Any]] = {}
        for r in records:
            pid = r.get("id")
            if pid is None:
                continue
            payload = r.get("payload") or {}
            current = int(payload.get("missed_opportunities", 0) or 0)
            new_value = max(0, current + int(delta))
            by_new_value.setdefault(new_value, []).append(pid)
        for new_value, ids in by_new_value.items():
            try:
                await self.qdrant.set_payload(
                    self.collection,
                    ids,
                    {"missed_opportunities": new_value},
                )
            except Exception:
                continue

    async def patch_vectors(self, signal_id: str, vectors: FacetVectors) -> None:
        """Update the facet vectors of an existing signal in-place.

        Main content vector update requires full upsert with existing payload
        (Qdrant does not expose vector-only patch for single-vector collections
        in a straightforward way via our thin client), so we read-modify-write.
        """
        existing = await self.get_required(signal_id)
        existing.vectors = vectors
        await self.put(existing)
