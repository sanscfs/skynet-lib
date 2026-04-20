"""Lazy-singleton Redis client factory.

Provides both sync and async clients with the same lazy-init pattern
used across all Skynet components.
"""

from __future__ import annotations

import logging
from typing import Optional

import redis as redis_sync

logger = logging.getLogger(__name__)

_sync_client: Optional[redis_sync.Redis] = None
_async_client = None  # Optional[redis.asyncio.Redis]


def get_redis(
    host: str = "redis.redis.svc",
    port: int = 6379,
    db: int = 0,
    decode_responses: bool = True,
    *,
    force_new: bool = False,
) -> redis_sync.Redis:
    """Return a lazy-initialized sync Redis client.

    The client is cached globally. Pass force_new=True to create a fresh one
    (useful in tests or after connection loss).
    """
    global _sync_client
    if _sync_client is not None and not force_new:
        return _sync_client
    _sync_client = redis_sync.Redis(
        host=host,
        port=port,
        db=db,
        decode_responses=decode_responses,
        socket_timeout=5,
        socket_connect_timeout=5,
    )
    try:
        _sync_client.ping()
    except Exception as e:
        logger.warning("Redis ping failed (host=%s:%s): %s", host, port, e)
    return _sync_client


def get_async_redis(
    host: str = "redis.redis.svc",
    port: int = 6379,
    db: int = 0,
    decode_responses: bool = True,
    *,
    force_new: bool = False,
    socket_timeout: float | None = 5,
    socket_connect_timeout: float | None = 5,
):
    """Return a lazy-initialized async Redis client.

    Requires `redis[hiredis]` or `redis` with asyncio support.

    ``socket_timeout`` defaults to 5s, which is right for short-RTT
    calls. Long-blocking consumers (e.g. the impulse engine's
    ``XREADGROUP BLOCK 3600000`` signal-driven loop) must override this
    via ``force_new=True`` plus ``socket_timeout=None`` (infinite) or a
    value greater than their longest expected block. Otherwise the
    socket timeout fires mid-block and drain returns a spurious error.
    """
    global _async_client
    if _async_client is not None and not force_new:
        return _async_client
    import redis.asyncio as aioredis

    client = aioredis.Redis(
        host=host,
        port=port,
        db=db,
        decode_responses=decode_responses,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_connect_timeout,
    )
    if not force_new:
        _async_client = client
    return client


def reset() -> None:
    """Close and discard cached clients. Mainly for tests."""
    global _sync_client, _async_client
    if _sync_client is not None:
        try:
            _sync_client.close()
        except Exception:
            pass
        _sync_client = None
    if _async_client is not None:
        # async close needs event loop; best-effort
        try:
            import asyncio

            asyncio.get_event_loop().run_until_complete(_async_client.close())
        except Exception:
            pass
        _async_client = None
