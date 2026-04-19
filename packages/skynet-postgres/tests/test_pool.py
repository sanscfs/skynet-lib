"""Tests for AsyncPool with mocked asyncpg.

All tests patch ``asyncpg.create_pool`` so nothing hits a real database.
The fake pool records query calls and lets tests arm specific calls to
raise auth errors, exercising the rotation-on-auth-error logic.
"""

from __future__ import annotations

import asyncio

import asyncpg
import pytest
from skynet_postgres import AsyncPool, PoolConfig
from skynet_postgres.exceptions import CredentialsRotationFailed

pytestmark = pytest.mark.asyncio


# --- test doubles -----------------------------------------------------


class FakePool:
    """Minimal asyncpg.Pool stand-in.

    Each query method reads a scripted response off ``self._scripts`` and
    returns or raises accordingly. If the script runs out, the default
    "OK" response is returned. ``closed`` flips to True after ``close()``.
    """

    def __init__(self, owner: "FakeAsyncpg", creds: tuple[str, str]):
        self.owner = owner
        self.creds = creds  # what credentials were used to build this pool
        self.closed = False
        self.calls: list[tuple[str, tuple]] = []

    async def _run(self, name: str, *args):
        self.calls.append((name, args))
        self.owner.all_calls.append((self.creds, name, args))
        if self.owner.scripts:
            action = self.owner.scripts.pop(0)
            if isinstance(action, BaseException):
                raise action
            return action
        # default benign return
        return {"fetch": [], "fetchrow": None, "fetchval": 1, "execute": "OK"}.get(name, None)

    async def fetch(self, query, *args):
        return await self._run("fetch", query, *args)

    async def fetchrow(self, query, *args):
        return await self._run("fetchrow", query, *args)

    async def fetchval(self, query, *args):
        return await self._run("fetchval", query, *args)

    async def execute(self, query, *args):
        return await self._run("execute", query, *args)

    async def executemany(self, query, args):
        self.calls.append(("executemany", (query, args)))
        self.owner.all_calls.append((self.creds, "executemany", (query, args)))
        if self.owner.scripts:
            action = self.owner.scripts.pop(0)
            if isinstance(action, BaseException):
                raise action
        return None

    async def close(self):
        self.closed = True


class FakeAsyncpg:
    """Factory that replaces ``asyncpg.create_pool`` during a test."""

    def __init__(self):
        self.scripts: list = []  # consumed left-to-right by query calls
        self.created_pools: list[FakePool] = []
        self.all_calls: list[tuple] = []
        # Raise this on the Nth create_pool call if set.
        self.create_pool_errors: list[BaseException | None] = []

    async def create_pool(self, **kwargs):
        if self.create_pool_errors:
            err = self.create_pool_errors.pop(0)
            if err is not None:
                raise err
        creds = (kwargs["user"], kwargs["password"])
        pool = FakePool(self, creds)
        self.created_pools.append(pool)
        return pool


@pytest.fixture
def fake_asyncpg(monkeypatch):
    fake = FakeAsyncpg()
    monkeypatch.setattr(
        "skynet_postgres.pool.asyncpg.create_pool",
        fake.create_pool,
    )
    return fake


def _config() -> PoolConfig:
    return PoolConfig(
        host="pg.example",
        port=5432,
        database="test",
        min_size=1,
        max_size=2,
        command_timeout=5,
    )


def _auth_error() -> asyncpg.InvalidAuthorizationSpecificationError:
    # asyncpg exceptions take a message positional arg.
    return asyncpg.InvalidAuthorizationSpecificationError("creds expired")


# --- start / initial creds -------------------------------------------


async def test_start_builds_pool_with_initial_creds(fake_asyncpg):
    # The production code calls ``creds_provider`` synchronously, so a
    # plain callable (not AsyncMock) is the correct double.
    calls: list[None] = []

    def creds():
        calls.append(None)
        return ("u1", "p1")

    pool = AsyncPool(config=_config(), creds_provider=creds)
    await pool.start()

    assert len(fake_asyncpg.created_pools) == 1
    assert fake_asyncpg.created_pools[0].creds == ("u1", "p1")
    assert len(calls) == 1

    # start() is idempotent.
    await pool.start()
    assert len(fake_asyncpg.created_pools) == 1

    await pool.close()


async def test_fetch_returns_pool_result_when_no_error(fake_asyncpg):
    fake_asyncpg.scripts = [[{"id": 1}]]
    pool = AsyncPool(config=_config(), creds_provider=lambda: ("u", "p"))
    await pool.start()
    rows = await pool.fetch("SELECT 1")
    assert rows == [{"id": 1}]
    await pool.close()


# --- rotation on auth error ------------------------------------------


async def test_fetch_rotates_on_auth_error_and_retries_once(fake_asyncpg, caplog):
    # First call fails with auth error; after rotation, second call succeeds.
    fake_asyncpg.scripts = [_auth_error(), [{"ok": True}]]
    seq = iter([("u1", "p1"), ("u2", "p2")])
    pool = AsyncPool(config=_config(), creds_provider=lambda: next(seq))

    with caplog.at_level("INFO", logger="skynet_postgres"):
        await pool.start()
        rows = await pool.fetch("SELECT 1")

    assert rows == [{"ok": True}]
    # Two pools were built -- the original and the rotated one.
    assert len(fake_asyncpg.created_pools) == 2
    assert fake_asyncpg.created_pools[0].creds == ("u1", "p1")
    assert fake_asyncpg.created_pools[1].creds == ("u2", "p2")
    # The stale pool was closed.
    assert fake_asyncpg.created_pools[0].closed is True
    # Rotation was logged; password was not.
    log_text = "\n".join(r.message for r in caplog.records)
    assert "rotating" in log_text.lower() or "rotated" in log_text.lower()
    assert "p1" not in log_text and "p2" not in log_text

    await pool.close()


async def test_rotation_does_not_loop_forever(fake_asyncpg):
    # Both the original call and the rotated retry fail auth -- we must
    # raise CredentialsRotationFailed, not loop.
    fake_asyncpg.scripts = [_auth_error(), _auth_error()]
    seq = iter([("u1", "p1"), ("u2", "p2")])
    pool = AsyncPool(config=_config(), creds_provider=lambda: next(seq))
    await pool.start()

    with pytest.raises(CredentialsRotationFailed):
        await pool.fetch("SELECT 1")

    # We rotated exactly once (two pools total: initial + one rotation).
    assert len(fake_asyncpg.created_pools) == 2

    await pool.close()


async def test_rotation_failure_during_rebuild_is_wrapped(fake_asyncpg):
    # Initial create_pool succeeds; rotation's create_pool raises
    # (e.g. Vault unreachable). The caller sees CredentialsRotationFailed.
    fake_asyncpg.scripts = [_auth_error()]
    fake_asyncpg.create_pool_errors = [None, RuntimeError("vault down")]
    pool = AsyncPool(
        config=_config(),
        creds_provider=lambda: ("u", "p"),
    )
    await pool.start()

    with pytest.raises(CredentialsRotationFailed):
        await pool.fetch("SELECT 1")

    await pool.close()


# --- concurrency ------------------------------------------------------


async def test_concurrent_callers_share_single_rotation(fake_asyncpg):
    # Three concurrent callers all hit an auth error on the *original*
    # pool. They should collectively trigger ONE rotation, not three.
    #
    # The scripts queue is shared across all pool instances, so we can't
    # just arm "3 auth errors then 3 OKs" -- ordering across tasks is
    # non-deterministic. Instead we patch FakePool._run to raise an
    # auth error iff the call lands on pool #0 (the stale one) and
    # return a benign result otherwise.
    original_run = FakePool._run
    call_count = {"n": 0}

    async def scripted_run(self, name, *args):
        call_count["n"] += 1
        if self is fake_asyncpg.created_pools[0]:
            # Stale pool -- always auth-fails.
            raise _auth_error()
        # Fresh pool -- benign result.
        return {"fetch": [{"ok": True}]}.get(name, None)

    FakePool._run = scripted_run  # type: ignore[assignment]
    try:
        creds_calls: list[int] = []

        def creds():
            n = len(creds_calls)
            creds_calls.append(n)
            return (f"u{n}", f"p{n}")

        pool = AsyncPool(config=_config(), creds_provider=creds)
        await pool.start()

        results = await asyncio.gather(
            pool.fetch("Q1"),
            pool.fetch("Q2"),
            pool.fetch("Q3"),
        )

        assert len(results) == 3
        for r in results:
            assert r == [{"ok": True}]
        # Initial build + exactly one rotation = 2 pools, 2 creds fetches.
        # This is the load-bearing assertion: concurrent auth errors
        # MUST collapse into a single rebuild.
        assert len(fake_asyncpg.created_pools) == 2
        assert len(creds_calls) == 2

        await pool.close()
    finally:
        FakePool._run = original_run  # type: ignore[assignment]


# --- executemany + lifecycle edges -----------------------------------


async def test_executemany_rotates_too(fake_asyncpg):
    fake_asyncpg.scripts = [_auth_error(), None]
    seq = iter([("u1", "p1"), ("u2", "p2")])
    pool = AsyncPool(config=_config(), creds_provider=lambda: next(seq))
    await pool.start()

    await pool.executemany("INSERT INTO t VALUES ($1)", [(1,), (2,)])

    assert len(fake_asyncpg.created_pools) == 2
    # The successful call landed on the new pool.
    assert ("executemany", ("INSERT INTO t VALUES ($1)", [(1,), (2,)])) in fake_asyncpg.created_pools[1].calls

    await pool.close()


async def test_query_before_start_raises(fake_asyncpg):
    from skynet_postgres.exceptions import PoolNotStartedError

    pool = AsyncPool(config=_config(), creds_provider=lambda: ("u", "p"))
    with pytest.raises(PoolNotStartedError):
        await pool.fetch("SELECT 1")


async def test_close_is_idempotent_and_blocks_reuse(fake_asyncpg):
    from skynet_postgres.exceptions import PoolClosedError

    pool = AsyncPool(config=_config(), creds_provider=lambda: ("u", "p"))
    await pool.start()
    await pool.close()
    await pool.close()  # no-op, must not raise

    with pytest.raises(PoolClosedError):
        await pool.start()
    with pytest.raises(PoolClosedError):
        await pool.fetch("SELECT 1")
