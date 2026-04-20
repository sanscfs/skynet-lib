"""Tests for PrototypeRegistry."""

from __future__ import annotations

import asyncio

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


# ----------------------------------------------------------------------
# Background warmup lifecycle


@pytest.mark.asyncio
async def test_start_warmup_is_non_blocking(async_hash_embedder) -> None:
    """start_warmup must return synchronously; no await required."""

    slow_calls = 0

    async def slow_embedder(text: str) -> list[float]:
        nonlocal slow_calls
        slow_calls += 1
        await asyncio.sleep(0.05)
        return await async_hash_embedder(text)

    registry = PrototypeRegistry(slow_embedder)
    # start_warmup is a plain `def`, not `async def` -- calling it should
    # return None immediately without doing the embeddings.
    before = slow_calls
    registry.start_warmup()
    after = slow_calls
    # Haven't yielded the loop yet; no embed call should have fired.
    assert after == before
    assert registry.ready is False
    # Clean up the scheduled task so pytest doesn't complain.
    assert await registry.wait_ready(timeout=10.0) is True


@pytest.mark.asyncio
async def test_ready_false_immediately_after_start(async_hash_embedder) -> None:
    registry = PrototypeRegistry(async_hash_embedder)
    registry.start_warmup()
    assert registry.ready is False
    # Give the task a chance to finish so pytest-asyncio shuts down cleanly.
    await registry.wait_ready(timeout=10.0)


@pytest.mark.asyncio
async def test_ready_true_after_warmup_completes(async_hash_embedder) -> None:
    registry = PrototypeRegistry(async_hash_embedder)
    registry.start_warmup()
    assert registry.ready is False
    ok = await registry.wait_ready(timeout=10.0)
    assert ok is True
    assert registry.ready is True
    # Default bundle should have populated some prototypes.
    assert len(registry.names()) > 0


@pytest.mark.asyncio
async def test_start_warmup_is_idempotent(async_hash_embedder) -> None:
    registry = PrototypeRegistry(async_hash_embedder)
    registry.start_warmup()
    task1 = registry._warmup_task
    # Calling again while still running must not spawn a second task.
    registry.start_warmup()
    task2 = registry._warmup_task
    assert task1 is task2
    await registry.wait_ready(timeout=10.0)


@pytest.mark.asyncio
async def test_wait_ready_times_out(async_hash_embedder) -> None:
    """wait_ready returns False if the deadline is hit before set()."""

    async def blocking_embedder(text: str) -> list[float]:
        await asyncio.sleep(10.0)
        return await async_hash_embedder(text)

    registry = PrototypeRegistry(blocking_embedder)
    registry.start_warmup()
    ok = await registry.wait_ready(timeout=0.05)
    assert ok is False
    assert registry.ready is False
    # Cancel the dangling task so the event loop shuts down cleanly.
    if registry._warmup_task is not None:
        registry._warmup_task.cancel()
        try:
            await registry._warmup_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_load_defaults_backcompat(async_hash_embedder) -> None:
    """load_defaults still works and leaves the registry ready."""
    registry = PrototypeRegistry(async_hash_embedder)
    await registry.load_defaults()
    assert registry.ready is True
    assert len(registry.names()) > 0
