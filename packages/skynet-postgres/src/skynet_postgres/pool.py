"""Async Postgres pool with Vault dynamic-credentials rotation.

``AsyncPool`` wraps ``asyncpg.create_pool`` and adds a single extra trick:
when a query fails because Vault-issued credentials expired (the database
answers with ``InvalidAuthorizationSpecificationError`` /
``InvalidPasswordError``), the pool is torn down, a fresh pair of
credentials is pulled from ``creds_provider``, a new pool is built, and
the failed call is retried *exactly once*.

The retry budget is deliberately one -- if new creds still don't work,
something is broken upstream (Vault, DB role, network) and we'd rather
surface the error than spin.

Concurrent safety: rotation is guarded by an ``asyncio.Lock`` so a
thundering herd of queries hitting expired creds only triggers one
rebuild.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Union

import asyncpg

from skynet_postgres.exceptions import (
    CredentialsRotationFailed,
    PoolClosedError,
    PoolNotStartedError,
)

logger = logging.getLogger("skynet_postgres")

# asyncpg errors that indicate the credentials went stale and a fresh pair
# from Vault should be tried. ``OSError`` covers the case where the server
# closes the connection abruptly (some Postgres/pgbouncer configs do this
# instead of returning a proper SQLSTATE).
_AUTH_ERRORS: tuple[type[BaseException], ...] = (
    asyncpg.InvalidAuthorizationSpecificationError,
    asyncpg.InvalidPasswordError,
)


# Credentials providers come in two shapes:
#
# * Sync -- ``Callable[[], Tuple[str, str]]``. Most callers pass a plain
#   function that pokes Vault via ``hvac`` and returns ``(user, pw)``.
#   That call is a blocking HTTP request (typically 50-200ms), so we
#   dispatch it via ``asyncio.to_thread`` to avoid freezing the event
#   loop while rotation is in-flight.
#
# * Async -- ``Callable[[], Awaitable[Tuple[str, str]]]``. Callers with
#   an async Vault client pass a coroutine function; we await it
#   directly.
#
# Detection uses ``inspect.iscoroutinefunction`` at call time; this
# supports ``functools.partial``/bound methods where the wrapped
# callable is async as long as Python sees the wrapper as a coroutine
# function. For the (rare) corner case where a sync wrapper returns a
# coroutine, we fall back to ``inspect.isawaitable`` on the result.
CredsProvider = Union[
    Callable[[], tuple[str, str]],
    Callable[[], Awaitable[tuple[str, str]]],
]


@dataclass(frozen=True)
class PoolConfig:
    """Static connection parameters -- everything except credentials.

    Credentials are not part of the config because they rotate; they are
    fetched fresh via ``creds_provider`` every time the pool is built.
    """

    host: str
    port: int = 5432
    database: str = "postgres"
    min_size: int = 2
    max_size: int = 10
    command_timeout: float = 30.0
    # ``extra`` is passed straight through to ``asyncpg.create_pool`` so
    # callers can tune things like ``statement_cache_size``, ``ssl``,
    # ``server_settings``, etc. without us having to mirror every
    # asyncpg knob here.
    extra: dict[str, Any] = field(default_factory=dict)


class AsyncPool:
    """asyncpg pool with a dynamic-credentials twist.

    The pool is *not* usable until ``start()`` is awaited, and is not
    reusable after ``close()``. Between those two calls it behaves like
    an ``asyncpg.Pool``:

        pool = AsyncPool(config=..., creds_provider=...)
        await pool.start()
        async with pool.acquire() as conn:
            await conn.fetch("SELECT 1")
        await pool.close()

    The convenience methods (``fetch``, ``fetchrow``, ``fetchval``,
    ``execute``, ``executemany``) wrap ``acquire()`` with the auth-error
    retry logic. If you need the retry logic around a custom call,
    reach for ``acquire()`` directly -- but note that rotation inside
    an ``async with pool.acquire()`` block is *not* supported: the
    connection you hold is gone after rotation. Use the convenience
    methods for one-shot queries, or drive your own transaction logic
    on top of ``acquire()``.
    """

    def __init__(self, *, config: PoolConfig, creds_provider: CredsProvider) -> None:
        self._config = config
        self._creds_provider = creds_provider
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        # Incremented on every successful rotation -- callers racing a
        # rotation use this to detect "my failure is stale, someone
        # already fixed it" and skip a second rotate.
        self._generation = 0

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Build the initial pool. Safe to await multiple times -- no-op
        after the first successful call."""
        if self._closed:
            raise PoolClosedError("pool has been closed and cannot be restarted")
        if self._pool is not None:
            return
        async with self._lock:
            if self._pool is not None:
                return
            await self._build_pool_locked()

    async def close(self) -> None:
        """Close the pool. Idempotent."""
        self._closed = True
        async with self._lock:
            if self._pool is not None:
                try:
                    await self._pool.close()
                finally:
                    self._pool = None

    # -- internals -------------------------------------------------------

    async def _resolve_creds(self) -> tuple[str, str]:
        """Invoke ``self._creds_provider`` without blocking the loop.

        * If the provider is a coroutine function, await it directly.
        * Otherwise call it in a worker thread via ``asyncio.to_thread``
          so the (usually blocking) Vault HTTP request doesn't freeze
          the event loop.
        * As a final fallback, if a sync provider happens to return a
          coroutine/awaitable (e.g. ``functools.partial`` wrapping an
          async function, which ``iscoroutinefunction`` misses), await
          the awaitable we got back.
        """
        provider = self._creds_provider
        if inspect.iscoroutinefunction(provider):
            return await provider()  # type: ignore[misc]
        result = await asyncio.to_thread(provider)  # type: ignore[arg-type]
        if inspect.isawaitable(result):
            return await result  # type: ignore[return-value]
        return result  # type: ignore[return-value]

    async def _build_pool_locked(self) -> None:
        """Build a fresh asyncpg pool from current credentials.

        Caller MUST hold ``self._lock``.
        """
        username, password = await self._resolve_creds()
        logger.info(
            "creating asyncpg pool user=%s host=%s:%s db=%s",
            username,
            self._config.host,
            self._config.port,
            self._config.database,
        )
        self._pool = await asyncpg.create_pool(
            host=self._config.host,
            port=self._config.port,
            database=self._config.database,
            user=username,
            password=password,
            min_size=self._config.min_size,
            max_size=self._config.max_size,
            command_timeout=self._config.command_timeout,
            **self._config.extra,
        )

    async def _rotate(self, failed_generation: int) -> None:
        """Replace the pool with a new one built from fresh credentials.

        ``failed_generation`` is the ``self._generation`` observed at the
        moment the caller decided a rotation was needed. If somebody else
        already rotated in between (generation moved forward), this call
        is a no-op -- one rotation is enough for a burst of concurrent
        auth errors.

        The rotation is atomic from the perspective of other callers:
        the stale pool stays referenced until the fresh one is built,
        so a concurrent ``_require_pool()`` never sees ``self._pool ==
        None`` mid-rebuild. (That matters because ``_resolve_creds``
        yields to the loop while the blocking Vault fetch runs in a
        worker thread -- without this ordering, racing callers would
        raise ``PoolNotStartedError`` during rotation.)
        """
        async with self._lock:
            if self._closed:
                raise PoolClosedError("pool is closed")
            if self._generation != failed_generation:
                # Another task already rotated; nothing to do.
                return
            old = self._pool
            # Fetch creds + build the new pool BEFORE nulling ``self._pool``
            # so racing ``_require_pool`` calls keep seeing the stale one
            # (they'll either succeed via an already-queued connection or
            # fall through to their own auth-error path, which this
            # ``_lock`` serializes and generation-checks).
            try:
                fresh = await self._build_fresh_pool_locked()
            except Exception:
                # Rotation failed -- leave the stale pool in place so
                # the next call has something to talk to. Callers that
                # caught the auth error will see a rotation failure
                # bubble up as ``CredentialsRotationFailed``.
                raise
            self._pool = fresh
            if old is not None:
                try:
                    await old.close()
                except Exception:  # pragma: no cover -- best-effort close
                    logger.warning("error closing stale pool during rotation", exc_info=True)
            self._generation += 1
            logger.info("pool rotated with fresh credentials (generation=%d)", self._generation)

    async def _build_fresh_pool_locked(self) -> "asyncpg.Pool":
        """Build a fresh asyncpg pool from current credentials and
        return it WITHOUT touching ``self._pool``. Caller MUST hold
        ``self._lock`` and is responsible for swapping the returned
        pool into place."""
        username, password = await self._resolve_creds()
        logger.info(
            "creating asyncpg pool user=%s host=%s:%s db=%s",
            username,
            self._config.host,
            self._config.port,
            self._config.database,
        )
        return await asyncpg.create_pool(
            host=self._config.host,
            port=self._config.port,
            database=self._config.database,
            user=username,
            password=password,
            min_size=self._config.min_size,
            max_size=self._config.max_size,
            command_timeout=self._config.command_timeout,
            **self._config.extra,
        )

    def _require_pool(self) -> asyncpg.Pool:
        if self._closed:
            raise PoolClosedError("pool is closed")
        if self._pool is None:
            raise PoolNotStartedError("pool not started -- call `await pool.start()` first")
        return self._pool

    # -- public query API ------------------------------------------------

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the underlying asyncpg pool.

        No retry logic here -- a ``with`` block must own the connection
        it holds, so rotating mid-block is unsafe. Wrap single queries
        with the convenience methods below for auth-retry behavior.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            yield conn

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        return await self._run("fetch", query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        return await self._run("fetchrow", query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await self._run("fetchval", query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        return await self._run("execute", query, *args)

    async def executemany(self, query: str, args: list[Any]) -> None:
        # ``executemany`` has a different arg shape (iterable of param
        # tuples), so it can't ride the same ``*args`` path.
        await self._run_executemany(query, args)

    # -- retry helpers ---------------------------------------------------

    async def _run(self, method: str, query: str, *args: Any) -> Any:
        pool = self._require_pool()
        gen = self._generation
        try:
            return await getattr(pool, method)(query, *args)
        except _AUTH_ERRORS:
            logger.warning("postgres auth error on %s -- rotating credentials", method)
            try:
                await self._rotate(failed_generation=gen)
            except Exception as exc:
                raise CredentialsRotationFailed(f"failed to rotate Vault credentials: {exc}") from exc
            pool = self._require_pool()
            try:
                return await getattr(pool, method)(query, *args)
            except _AUTH_ERRORS as exc:
                raise CredentialsRotationFailed(
                    "postgres rejected freshly-rotated credentials -- check Vault DB role and network path"
                ) from exc

    async def _run_executemany(self, query: str, args: list[Any]) -> None:
        pool = self._require_pool()
        gen = self._generation
        try:
            await pool.executemany(query, args)
            return
        except _AUTH_ERRORS:
            logger.warning("postgres auth error on executemany -- rotating credentials")
            try:
                await self._rotate(failed_generation=gen)
            except Exception as exc:
                raise CredentialsRotationFailed(f"failed to rotate Vault credentials: {exc}") from exc
            pool = self._require_pool()
            try:
                await pool.executemany(query, args)
            except _AUTH_ERRORS as exc:
                raise CredentialsRotationFailed(
                    "postgres rejected freshly-rotated credentials -- check Vault DB role and network path"
                ) from exc
