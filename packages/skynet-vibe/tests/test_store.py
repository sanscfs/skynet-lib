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
async def test_count_enforces_category_filter(fake_qdrant) -> None:
    """count() must only see points with category == sub_category.

    Regression test: /vibe/status was returning store_unavailable
    because the fallback looked for a non-existent ``store.client``
    attribute and never reached the count path. Guard both the count
    semantics and the attribute name here.
    """
    store = VibeStore(fake_qdrant, collection="test_coll", sub_category="vibe_signal")
    # Two vibe signals
    await store.put(_make_signal())
    await store.put(_make_signal())
    # One non-vibe point that should NOT be counted
    await fake_qdrant.upsert(
        "test_coll",
        [
            {
                "id": "legacy-pref",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {"category": "cinema_preferences", "text_raw": "no"},
            }
        ],
    )
    # The store attribute for the qdrant client is ``qdrant`` (not ``client``);
    # this check is the attribute-name regression guard.
    assert store.qdrant is fake_qdrant
    assert await store.count() == 2


@pytest.mark.asyncio
async def test_pool_stats_backfilled_source_v2(fake_qdrant) -> None:
    """pool_stats() should bucket by ``source_v2.type`` after backfill.

    After the 2026-04-20 backfill every point in ``user_profile_raw``
    carries ``source_v2`` with a structured type (chat/telemetry/dag/etc.).
    Older points retain their legacy ``source`` dict too. pool_stats
    prefers ``source_v2.type`` and falls back gracefully.
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
