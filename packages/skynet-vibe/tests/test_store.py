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
