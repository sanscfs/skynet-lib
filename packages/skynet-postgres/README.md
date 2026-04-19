# skynet-postgres

Async Postgres pool with Vault dynamic-credentials rotation for Skynet services.

Extracted from the duplicated ~100 LOC `db.py` pattern used by `skynet-movies`
(and soon `skynet-music`) -- every service that uses Vault-issued Postgres
credentials needs the same rotate-on-auth-failure dance, so it lives here
once.

## Install

```bash
pip install skynet-postgres \
    --extra-index-url http://nexus.nexus.svc:8081/repository/pypi-group/simple/ \
    --trusted-host nexus.nexus.svc
```

## Usage

```python
from skynet_postgres import AsyncPool, PoolConfig

pool = AsyncPool(
    config=PoolConfig(
        host="skynet-postgres.postgres.svc",
        port=5432,
        database="sanscfs-movies",
        min_size=2,
        max_size=10,
        command_timeout=30,
    ),
    # Returns (username, password). Called every time the pool is built,
    # including after Vault lease expiry forces a rotation.
    creds_provider=lambda: vault_client.get_db_creds("skynet-movies-role").to_tuple(),
)

await pool.start()

# Convenience methods -- retry once on auth error with fresh Vault creds.
rows = await pool.fetch("SELECT * FROM movies WHERE id = $1", movie_id)
row = await pool.fetchrow("SELECT * FROM movies WHERE id = $1", movie_id)
val = await pool.fetchval("SELECT count(*) FROM movies")
await pool.execute("UPDATE movies SET watched = true WHERE id = $1", movie_id)
await pool.executemany("INSERT INTO ... VALUES ($1, $2)", rows)

# Raw connection access -- NO retry, you own the lifetime.
async with pool.acquire() as conn:
    async with conn.transaction():
        await conn.execute("INSERT ...")
        await conn.execute("UPDATE ...")

await pool.close()
```

## Behavior

- On `asyncpg.InvalidAuthorizationSpecificationError` or
  `InvalidPasswordError`, the pool tears itself down, calls
  `creds_provider()` again, rebuilds, and retries the failed call **once**.
- Rotation is lock-guarded: a burst of concurrent queries hitting stale
  creds triggers a single rebuild, not one-per-caller.
- If the retry with fresh creds also fails auth, `CredentialsRotationFailed`
  is raised -- no infinite loops.
- Logging goes to the `skynet_postgres` logger; rotation events log at
  INFO. The password is never logged.

## Caveats

- `pool.acquire()` does **not** retry on auth error -- a `with` block
  must own its connection. Use the convenience methods for single
  queries, or handle rotation yourself on top of `acquire()` for
  multi-statement transactions.
- `creds_provider` is a plain `Callable[[], Tuple[str, str]]`. Once
  `skynet-vault` lands, a convenience overload that takes a vault
  client + role name will be added here -- for now, wire the callback
  yourself.

## Dependencies

- `asyncpg >= 0.29, < 1.0`
- Python `>= 3.11`
