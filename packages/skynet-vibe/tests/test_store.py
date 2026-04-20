"""Tests for VibeStore (with FakeQdrant stub)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from skynet_vibe import FacetVectors, Source, VibeSignal, VibeStore


def _make_signal(**overrides) -> VibeSignal:
    base = dict(
        id=VibeSignal.new_id(),
        text_raw="this album is exactly my evening mood",
        vectors=FacetVectors(content=[0.1, 0.2, 0.3, 0.4], context=[0.5, 0.6, 0.7, 0.8]),
        source=Source(type="chat", room_id="!room"),
        timestamp=datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc),
        confidence=0.9,
    )
    base.update(overrides)
    return VibeSignal(**base)


@pytest.mark.asyncio
async def test_put_and_get_roundtrip(fake_qdrant) -> None:
    store = VibeStore(fake_qdrant, collection="test_coll")
    signal = _make_signal()
    await store.put(signal)
    fetched = await store.get(signal.id)
    assert fetched is not None
    assert fetched.id == signal.id
    assert fetched.text_raw == signal.text_raw
    assert fetched.vectors.content == signal.vectors.content
    assert fetched.vectors.context == signal.vectors.context
    assert fetched.confidence == 0.9
    assert fetched.source.type == "chat"


@pytest.mark.asyncio
async def test_search_filters_by_category(fake_qdrant) -> None:
    store = VibeStore(fake_qdrant, collection="test_coll", sub_category="vibe_signal")
    # A vibe signal
    signal = _make_signal()
    await store.put(signal)
    # A non-vibe payload directly inserted
    await fake_qdrant.upsert(
        "test_coll",
        [
            {
                "id": "other",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {"category": "something_else", "text_raw": "no"},
            }
        ],
    )
    hits = await store.search(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=5, noise_floor=0.0)
    ids = [h.id for h in hits]
    assert signal.id in ids
    assert "other" not in ids


@pytest.mark.asyncio
async def test_search_returns_sorted_by_score(fake_qdrant) -> None:
    store = VibeStore(fake_qdrant, collection="test_coll")
    close = _make_signal()
    far = _make_signal()
    far.vectors = FacetVectors(content=[-0.1, -0.2, -0.3, -0.4])
    await store.put(close)
    await store.put(far)
    hits = await store.search(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=5, noise_floor=-1.0)
    # Closer one should come first (higher cosine)
    assert hits[0].id == close.id


@pytest.mark.asyncio
async def test_patch_vectors(fake_qdrant) -> None:
    store = VibeStore(fake_qdrant, collection="test_coll")
    signal = _make_signal()
    await store.put(signal)
    new_vecs = FacetVectors(content=[0.9, 0.8, 0.7, 0.6])
    await store.patch_vectors(signal.id, new_vecs)
    fetched = await store.get(signal.id)
    assert fetched is not None
    assert fetched.vectors.content == [0.9, 0.8, 0.7, 0.6]
    assert fetched.vectors.context is None


@pytest.mark.asyncio
async def test_get_missing_returns_none(fake_qdrant) -> None:
    store = VibeStore(fake_qdrant)
    assert await store.get("does-not-exist") is None


@pytest.mark.asyncio
async def test_count_inclusive_disjunction_spans_pool(fake_qdrant) -> None:
    """count() must cover BOTH halves of the vibe pool.

    The 2026-04-20 backfill stamped ``signal_version=2`` + ``source_v2``
    on ~13k pre-existing records in ``user_profile_raw`` while
    preserving their legacy ``category`` values (``gemini_facts``,
    ``phone_telemetry``, ``git_history``, ...). New VibeStore writes
    tag ``category=vibe_signal`` (and also stamp signal_version=2).

    The default pool filter is a disjunction —
    ``signal_version >= 2 OR category == vibe_signal`` — so BOTH
    buckets are visible. The ``category=vibe_signal`` branch is kept
    as a safety net for future writers that skip ``signal_version``.

    Also pins the ``store.qdrant`` attribute name (NOT ``store.client``
    -- that was the root cause of the ``/vibe/status store_unavailable``
    regression on a healthy 13k pool).
    """
    store = VibeStore(fake_qdrant, collection="test_coll", sub_category="vibe_signal")
    # Two brand-new vibe writes (category=vibe_signal via put()).
    await store.put(_make_signal())
    await store.put(_make_signal())
    await fake_qdrant.upsert(
        "test_coll",
        [
            # Backfill-retrofitted: legacy category preserved, but
            # signal_version=2 matches the range branch.
            {
                "id": "retrofitted",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {
                    "category": "gemini_facts",
                    "signal_version": 2,
                    "source": "google_takeout_gemini",
                    "source_v2": {"type": "chat", "writer": "google-takeout"},
                },
            },
            # Safety-net branch: no signal_version field at all, but
            # category=vibe_signal matches -- hypothetical writer that
            # forgot to stamp signal_version.
            {
                "id": "category-only",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {
                    "category": "vibe_signal",
                    "source": {"type": "chat"},
                },
            },
            # Neither branch matches -- must not be counted.
            {
                "id": "pre-v2",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {"category": "phone_telemetry", "text_raw": "no"},
            },
        ],
    )
    assert store.qdrant is fake_qdrant
    # Four points match the disjunction:
    # - Two put() writes (signal_version=2 AND category=vibe_signal)
    # - One retrofitted record (signal_version=2 branch)
    # - One category-only record (category=vibe_signal branch)
    # The pre-v2 legacy record matches neither branch and is excluded.
    assert await store.count() == 4


@pytest.mark.asyncio
async def test_pool_stats_backfilled_source_v2(fake_qdrant) -> None:
    """pool_stats() should bucket by ``source_v2.type`` after backfill.

    After the 2026-04-20 backfill every point in ``user_profile_raw``
    carries ``source_v2`` with a structured type (chat/telemetry/dag/etc.).
    Older points retain their legacy ``source`` dict too. pool_stats
    prefers ``source_v2.type`` and falls back gracefully.

    The pool filter (default ``signal_version == 2``) spans both new
    category=vibe_signal writes AND retrofitted records whose
    ``category`` is something else but signal_version==2.
    """
    store = VibeStore(fake_qdrant, collection="test_coll")
    # Two chat signals + one dag signal, mimicking post-backfill payload shape.
    await fake_qdrant.upsert(
        "test_coll",
        [
            {
                "id": "chat-a",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {
                    "category": "vibe_signal",
                    "source": {"type": "chat"},
                    "source_v2": {"type": "chat", "writer": "skynet-chat"},
                    "signal_version": 2,
                    "timestamp": "2026-04-19T12:00:00+00:00",
                },
            },
            {
                "id": "chat-b",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {
                    "category": "vibe_signal",
                    "source": {"type": "chat"},
                    "source_v2": {"type": "chat"},
                    "signal_version": 2,
                    "timestamp": "2026-04-20T09:00:00+00:00",
                },
            },
            {
                "id": "dag-a",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {
                    "category": "vibe_signal",
                    "source": {"type": "dag"},
                    "source_v2": {"type": "dag", "dag_id": "collect_git"},
                    "signal_version": 2,
                    "timestamp": "2026-04-01T00:00:00+00:00",
                },
            },
            {
                "id": "legacy-noise",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {
                    "category": "cinema_preferences",
                    "source": {"type": "movie_watch"},
                },
            },
        ],
    )
    stats = await store.pool_stats()
    assert stats["count"] == 3
    assert stats["by_source"] == {"chat": 2, "dag": 1}
    assert stats["oldest_ts"] == "2026-04-01T00:00:00+00:00"
    assert stats["collection"] == "test_coll"
    assert stats["category"] == "vibe_signal"
