# skynet-capture

Shared ingest building blocks for Skynet services that turn free-form
Matrix messages into structured rows (Postgres + Qdrant).

Three consumers of the same flow today:

- `skynet-profiler` passively subscribes to `matrix:events` (Redis
  stream) and fans out to per-domain modules.
- `skynet-music` chat agent receives Matrix messages in its own room
  and calls MCP tools like `mark_listened` that end up in the same DB.
- `skynet-movies` mirrors music for watch events.

`skynet-capture` is the place the three converge. Phase 1 exposes two
low-level helpers used by all of them; higher-level `CaptureModule`
classes for music/movies land in later phases.

## Phase 1 API

```python
from skynet_capture.common.consumption_extractor import (
    extract_consumption, MIN_LLM_TEXT_LEN,
)
# caller injects any LLM client conforming to LLMLike Protocol
items = await extract_consumption(text, kind="movies", llm=my_llm)

from skynet_capture.common.pg import (
    get_pool, close_pool, close_all_pools, resolve_dsn, PoolLike,
)
pool = await get_pool(dsn, cache_key="movies")
```

Install with the `postgres` extra when you actually write to PG:

```
pip install 'skynet-capture[postgres]>=2026.4.24'
```

## Deliberately out of scope for Phase 1

- LLM client implementation (caller provides one that satisfies the
  Protocol — usually `skynet_profiler.llm.LLMClient`).
- Qdrant writes (separate phase, shared helper for embedding + upsert).
- Vault / DSN resolution beyond `resolve_dsn(env_var)`.
