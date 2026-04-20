"""Shared Postgres helpers for capture-module writers.

Moved verbatim from ``skynet_profiler.modules._pg`` in Phase 1 of the
shared-ingest rollout. Imports were rewritten to drop the
profiler-specific logger name.

Two concerns live here:

1. **Lazy pool construction** — the first call to ``get_pool(dsn, ...)``
   builds an asyncpg pool and caches it under the given cache key; later
   calls reuse the same pool. This lets consumers boot without PG
   configured (dev / dry-run) and lazily light up each module's
   connection only when the first relevant event arrives.

2. **Schema bootstrap contract** — each writer module owns a small
   "ensure this column + unique index exist" idempotent SQL snippet that
   runs on the first use of the pool. The unique index guards against
   duplicate inserts when XCLAIM redelivers a Matrix event (edit, bot
   restart, etc.). The helper itself stays schema-agnostic.

The module deliberately does NOT implement Vault credential rotation.
The design target is: Vault K8s auth is configured on the pod via
``VAULT_ADDR`` + ``VAULT_ROLE``, and the writer module receives a fully
resolved DSN (``postgresql://user:pw@host/db``) built once at startup
from a dynamic Vault role. If creds rotate, the pool eventually fails
auth and the operator triggers a pod restart — for the low-throughput
write paths this is the cheapest correct answer; upgrading to
``skynet_postgres.AsyncPool`` remains on the table once the Vault role
bootstrap lands.

Modules that do NOT want the DSN-shortcut path can pass an already-built
pool directly; this helper never makes the module's write path depend
on one specific authentication flow.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional, Protocol

log = logging.getLogger("skynet_capture.common.pg")

# One asyncpg Pool per DSN-cache-key (typically "movies" / "music").
# asyncpg.Pool is thread-safe across asyncio tasks within one event
# loop; we assume each consumer runs in a single asyncio loop so one
# pool per cache key is fine.
_pools: dict[str, Any] = {}
_pool_locks: dict[str, asyncio.Lock] = {}


class PoolLike(Protocol):
    """The narrow subset of ``asyncpg.Pool`` callers actually use.

    Declared as a ``Protocol`` so tests can pass a plain duck-typed fake
    (an object with ``execute`` + ``fetchrow`` coroutines) instead of
    spinning a real asyncpg pool against a test Postgres.
    """

    async def execute(self, query: str, *args: Any) -> Any: ...
    async def fetchrow(self, query: str, *args: Any) -> Any: ...


def _lock_for(key: str) -> asyncio.Lock:
    lock = _pool_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _pool_locks[key] = lock
    return lock


async def get_pool(dsn: str, *, cache_key: str) -> Any:
    """Return (lazily creating) an asyncpg pool for ``dsn``.

    ``cache_key`` is the process-local cache slot; pass something stable
    per-module like ``"movies"`` or ``"music"``. Re-calling with a
    different DSN under the same key replaces the old pool (useful in
    tests; in production DSNs don't change at runtime).

    Raises ``RuntimeError`` if asyncpg isn't installed — callers should
    check env/DSN readiness first (see ``resolve_dsn``) and/or install
    the ``skynet-capture[postgres]`` extra.
    """
    try:  # deferred: keep asyncpg import optional in dev / dry-run
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — runtime-only guard
        raise RuntimeError("asyncpg is required for PG writes; install skynet-capture[postgres] to pull it in") from exc

    async with _lock_for(cache_key):
        existing = _pools.get(cache_key)
        if existing is not None:
            return existing
        pool = await asyncpg.create_pool(
            dsn,
            min_size=1,
            max_size=4,
            command_timeout=10.0,
        )
        _pools[cache_key] = pool
        log.info("opened asyncpg pool for %s", cache_key)
        return pool


async def close_pool(cache_key: str) -> None:
    """Close and forget the pool under ``cache_key``. Safe to call twice."""
    async with _lock_for(cache_key):
        pool = _pools.pop(cache_key, None)
        if pool is None:
            return
        try:
            await pool.close()
        except Exception as exc:  # pragma: no cover — best-effort close
            log.warning("pool close failed for %s: %s", cache_key, exc)


async def close_all_pools() -> None:
    """Close every cached pool. Used by test teardown and graceful shutdown."""
    keys = list(_pools.keys())
    for k in keys:
        await close_pool(k)


def resolve_dsn(env_var: str) -> Optional[str]:
    """Return the DSN from ``env_var`` or ``None`` if unset/empty.

    Callers gate their ``write()`` on this: when ``None``, the module
    logs the signal but skips the DB insert — same pattern as
    ``PROFILER_DRY_RUN`` but scoped per-target so movies can be live
    while music is still dry-running.
    """
    dsn = os.getenv(env_var, "").strip()
    return dsn or None
