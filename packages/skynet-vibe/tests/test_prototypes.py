"""Tests for PrototypeRegistry."""

from __future__ import annotations

import pytest
from skynet_vibe import PrototypeNotFoundError, PrototypeRegistry


@pytest.mark.asyncio
async def test_add_and_get(async_hash_embedder) -> None:
    registry = PrototypeRegistry(async_hash_embedder)
    proto = await registry.add("music", ["albums tracks artists", "rhythm beat production"])
    assert proto.name == "music"
    assert len(proto.centroid) == 16
    # L2-normalized
    norm = sum(x * x for x in proto.centroid) ** 0.5
    assert abs(norm - 1.0) < 1e-6
    assert registry.get_sync("music").name == "music"
    assert "music" in registry
    assert registry.names() == ["music"]


@pytest.mark.asyncio
async def test_refresh_replaces_seeds(async_hash_embedder) -> None:
    registry = PrototypeRegistry(async_hash_embedder)
    await registry.add("music", ["seed one"])
    refreshed = await registry.refresh("music", ["seed two", "seed three"])
    assert refreshed.seed_phrases == ["seed two", "seed three"]


@pytest.mark.asyncio
async def test_refresh_missing_raises(async_hash_embedder) -> None:
    registry = PrototypeRegistry(async_hash_embedder)
    with pytest.raises(PrototypeNotFoundError):
        await registry.refresh("nonexistent")


@pytest.mark.asyncio
async def test_get_missing_raises(async_hash_embedder) -> None:
    registry = PrototypeRegistry(async_hash_embedder)
    with pytest.raises(PrototypeNotFoundError):
        await registry.get("nope")


@pytest.mark.asyncio
async def test_load_from_config(async_hash_embedder) -> None:
    registry = PrototypeRegistry(async_hash_embedder)
    await registry.load_from_config(
        {
            "music": ["albums"],
            "movies": ["films"],
            "books": ["books reading prose"],
        }
    )
    assert set(registry.names()) == {"music", "movies", "books"}


@pytest.mark.asyncio
async def test_sync_embedder_works_too(hash_embedder) -> None:
    registry = PrototypeRegistry(hash_embedder)
    proto = await registry.add("photo", ["images light composition"])
    assert proto.name == "photo"


@pytest.mark.asyncio
async def test_add_requires_seeds(async_hash_embedder) -> None:
    registry = PrototypeRegistry(async_hash_embedder)
    with pytest.raises(ValueError):
        await registry.add("empty", [])
