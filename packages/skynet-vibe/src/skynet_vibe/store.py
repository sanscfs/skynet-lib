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
from skynet_vibe.filters import vibe_signal_filter
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
    async def count(self, collection: str, *, filter: dict | None = None) -> int: ...
    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        *,
        offset: str | int | None = None,
        filter: dict | None = None,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> tuple[list[dict], str | int | None]: ...


def _extract_source_type(payload: dict[str, Any]) -> str:
    """Return a human-readable source bucket from a signal payload.

    Prefers ``source_v2.type`` (post-backfill structured source), falls
    back to ``source.type`` (legacy Source-dict), then the plain
    ``source`` string, then ``"unknown"``. Used by :meth:`VibeStore.pool_stats`.
    """
    src_v2 = payload.get("source_v2")
    if isinstance(src_v2, dict):
        t = src_v2.get("type")
        if t:
            return str(t)
    src = payload.get("source")
    if isinstance(src, dict):
        return str(src.get("type") or "unknown")
    if isinstance(src, str) and src:
        return src
    return str(payload.get("source_type") or "unknown")


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
    # source_v2 mirrors the structured source dict so new writes show up
    # in the ``signal_version == 2`` pool alongside backfilled records.
    # We derive it from ``signal.source`` (a Source dataclass) with a
    # minimal {type, writer} shape; richer fields can be added by callers
    # via ``extra_payload`` if needed. See the 2026-04-20 backfill DAG
    # for the canonical ``source`` -> ``source_v2`` mapping.
    source_dict = signal.source.to_dict()
    source_v2 = {"type": source_dict.get("type", "unknown")}
    if source_dict.get("writer"):
        source_v2["writer"] = source_dict["writer"]
    payload: dict[str, Any] = {
        "category": sub_category,
        "text_raw": signal.text_raw,
        "source": source_dict,
        "source_v2": source_v2,
        "signal_version": 2,
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
        pool_filter: dict | None = None,
    ):
        self.qdrant = qdrant_client
        self.collection = collection
        self.sub_category = sub_category
        # ``pool_filter`` is the filter used by ``pool_stats`` / ``count``
        # to identify vibe-compatible points. Defaults to the disjunction
        # ``signal_version >= 2 OR category == sub_category`` so it
        # includes BOTH:
        #   - new VibeStore.put() writes (tagged category=sub_category), AND
        #   - legacy records retrofitted by backfill DAGs that keep
        #     their original ``category`` (e.g. ``gemini_facts``,
        #     ``phone_telemetry``) but gain ``signal_version=2`` +
        #     ``source_v2``.
        # Keeping BOTH checks as a safety net: if a future writer forgets
        # to stamp ``signal_version`` it still lands in the pool via the
        # ``category`` match, and if a retrofitted record keeps its legacy
        # category it still lands via ``signal_version``. See the
        # 2026-04-20 schema backfill + the filter helper for rationale.
        # Callers that want a strict view can pass an explicit
        # ``pool_filter`` -- typically ``{"must": [{"key": "category",
        # "match": {"value": "vibe_signal"}}]}`` for legacy behaviour.
        self.pool_filter = pool_filter or vibe_signal_filter(
            vibe_category=sub_category,
        )

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
        final_filter = self._vibe_filter(filter_)
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

    async def count(self, filter_: dict | None = None) -> int:
        """Return the exact number of vibe-compatible points.

        Uses ``self.pool_filter`` (default ``signal_version == 2``) so
        this counts the full retrofitted-pool after the 2026-04-20
        backfill, not just records skynet-vibe wrote itself. Pass
        ``filter_`` to merge in additional ``must`` clauses.
        """
        final_filter = self._merge_pool_filter(filter_)
        return int(await self.qdrant.count(self.collection, filter=final_filter))

    async def pool_stats(
        self,
        *,
        limit: int = 2048,
        extra_filter: dict | None = None,
    ) -> dict[str, Any]:
        """Return ``{count, by_source, oldest_ts}`` for the vibe pool.

        Uses a single exact ``count`` call plus a bounded payload scroll
        (for source-bucket breakdown + oldest timestamp). The scroll is
        bounded so on very large pools this stays cheap; callers get the
        exact total via ``count`` regardless.

        Filter = ``self.pool_filter`` merged with ``extra_filter``.
        Default ``pool_filter`` is ``signal_version == 2`` so the
        endpoint reports the full retrofitted pool after the 2026-04-20
        schema backfill -- which left the legacy ``category`` values
        (phone_telemetry, gemini_facts, ...) intact and only added
        ``signal_version`` + ``source_v2`` as new payload fields.
        """
        final_filter = self._merge_pool_filter(extra_filter)
        total = int(await self.qdrant.count(self.collection, filter=final_filter))

        by_source: dict[str, int] = {}
        oldest: str | None = None
        # Scroll a bounded window for source-type breakdown. We do NOT
        # need all points; even 2048 gives a representative distribution
        # on a 13k pool. For exact by_source on very large pools, switch
        # to a Qdrant facet query when the server version supports it.
        offset: Any = None
        scanned = 0
        while scanned < limit:
            batch_limit = min(256, limit - scanned)
            points, next_offset = await self.qdrant.scroll(
                self.collection,
                limit=batch_limit,
                offset=offset,
                filter=final_filter,
                with_payload=True,
                with_vector=False,
            )
            if not points:
                break
            for p in points:
                payload = p.get("payload") or {}
                src = _extract_source_type(payload)
                by_source[src] = by_source.get(src, 0) + 1
                ts = payload.get("timestamp")
                if isinstance(ts, str):
                    if oldest is None or ts < oldest:
                        oldest = ts
            scanned += len(points)
            if not next_offset:
                break
            offset = next_offset

        return {
            "count": total,
            "sampled": scanned,
            "by_source": dict(sorted(by_source.items(), key=lambda kv: -kv[1])),
            "oldest_ts": oldest,
            "collection": self.collection,
            "category": self.sub_category,
            "pool_filter": self.pool_filter,
        }

    def _vibe_filter(self, extra: dict | None) -> dict[str, Any]:
        """Build a Qdrant filter pinned to ``category == sub_category``.

        Used by ``search``: search targets strictly skynet-vibe-managed
        signals, not the broader retrofitted pool. See ``_merge_pool_filter``
        for the counting-path filter that spans the backfilled records.
        """
        must: list[dict] = [{"key": "category", "match": {"value": self.sub_category}}]
        if extra:
            if "must" in extra and isinstance(extra["must"], list):
                must.extend(extra["must"])
            else:
                for key, value in extra.items():
                    if key in ("must", "should", "must_not"):
                        continue
                    must.append({"key": key, "match": {"value": value}})
        return {"must": must}

    def _merge_pool_filter(self, extra: dict | None) -> dict[str, Any]:
        """Build a Qdrant filter = ``self.pool_filter`` AND ``extra``.

        Used by ``count`` and ``pool_stats``. ``self.pool_filter`` is
        the disjunction ``signal_version >= 2 OR category == sub_category``
        by default (see :func:`skynet_vibe.filters.vibe_signal_filter`),
        so the retrofitted 13k pool (post 2026-04-20 backfill) counts as
        "vibe-compatible" alongside explicitly-tagged vibe writes.

        If ``extra`` has no extra clauses, the base disjunction is
        returned verbatim. Otherwise, the disjunction is nested inside
        ``must`` so that ``extra.must`` clauses AND with the OR.
        """
        if not extra:
            # Return a deep-ish copy so callers can't mutate our base.
            return {k: list(v) if isinstance(v, list) else v for k, v in self.pool_filter.items()}

        must_clauses: list[dict[str, Any]] = []
        if "must" in extra and isinstance(extra["must"], list):
            must_clauses.extend(extra["must"])
        for key, value in extra.items():
            if key in ("must", "should", "must_not"):
                continue
            must_clauses.append({"key": key, "match": {"value": value}})

        base = self.pool_filter
        if "should" in base and isinstance(base["should"], list):
            # Disjunction base: nest it inside the must so extras AND it.
            must_clauses.append({"should": list(base["should"])})
        elif "must" in base and isinstance(base["must"], list):
            # Simple conjunction base (back-compat for operator overrides).
            must_clauses = list(base["must"]) + must_clauses
        return {"must": must_clauses}

    async def patch_vectors(self, signal_id: str, vectors: FacetVectors) -> None:
        """Update the facet vectors of an existing signal in-place.

        Main content vector update requires full upsert with existing payload
        (Qdrant does not expose vector-only patch for single-vector collections
        in a straightforward way via our thin client), so we read-modify-write.
        """
        existing = await self.get_required(signal_id)
        existing.vectors = vectors
        await self.put(existing)
