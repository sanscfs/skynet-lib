"""Lazy-singleton Redis client factory.

Provides both sync and async clients with the same lazy-init pattern
used across all Skynet components.
"""

from __future__ import annotations

import logging
from typing import Optional

import redis as redis_sync
from redis.backoff import ExponentialBackoff
from redis.retry import Retry

logger = logging.getLogger(__name__)

_sync_client: Optional[redis_sync.Redis] = None
_async_client = None  # Optional[redis.asyncio.Redis]

# Retry up to 6 times with exponential backoff capped at 30 s.
# Covers transient outages (pod restarts, network blips) without
# hot-spinning. Applied to both sync and async clients.
_RETRY = Retry(ExponentialBackoff(cap=30, base=1), retries=6)
_RETRY_ERRORS = [ConnectionError, TimeoutError, OSError]


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
        retry=_RETRY,
        retry_on_error=_RETRY_ERRORS,
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
        retry=_RETRY,
        retry_on_error=_RETRY_ERRORS,
    )
    if not force_new:
        _async_client = client
    return client


_RELEASE_LUA = (
    "if redis.call('get',KEYS[1])==ARGV[1] then "
    "return redis.call('del',KEYS[1]) else return 0 end"
)


def leader_lock(redis_client: redis_sync.Redis, key: str, ttl: int, holder: str) -> bool:
    """Acquire a distributed leader lock.  Returns True if this pod is leader.

    Use before periodic background jobs so only one replica runs per interval.
    TTL should equal the job interval: a crashed pod's lock auto-expires before
    the next run so another replica can take over without manual cleanup.

    Release early (after job completes) with :func:`release_leader_lock` so
    the next run starts on time instead of waiting for the full TTL.
    """
    return bool(redis_client.set(key, holder, nx=True, ex=ttl))


async def async_leader_lock(async_redis, key: str, ttl: int, holder: str) -> bool:
    """Async variant of :func:`leader_lock`."""
    return bool(await async_redis.set(key, holder, nx=True, ex=ttl))


def release_leader_lock(redis_client: redis_sync.Redis, key: str, holder: str) -> None:
    """Release a leader lock only if ``holder`` still owns it.

    Uses a Lua compare-and-delete so we never accidentally release a lock
    that was re-acquired by a different replica after TTL expiry.
    """
    try:
        redis_client.eval(_RELEASE_LUA, 1, key, holder)
    except Exception:
        pass


async def async_release_leader_lock(async_redis, key: str, holder: str) -> None:
    """Async variant of :func:`release_leader_lock`."""
    try:
        await async_redis.eval(_RELEASE_LUA, 1, key, holder)
    except Exception:
        pass


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
