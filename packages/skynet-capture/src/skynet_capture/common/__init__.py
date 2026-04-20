"""Phase-1 helpers: LLM-backed consumption extractor + Postgres pool.

Everything here is caller-agnostic — profiler, music, movies and future
agents import the same functions. No direct LLM SDK dependency; callers
pass a Protocol-conforming client.
"""

from .consumption_extractor import (
    MIN_LLM_TEXT_LEN,
    LLMLike,
    extract_consumption,
)
from .pg import (
    PoolLike,
    close_all_pools,
    close_pool,
    get_pool,
    resolve_dsn,
)

__all__ = [
    "MIN_LLM_TEXT_LEN",
    "LLMLike",
    "extract_consumption",
    "PoolLike",
    "close_all_pools",
    "close_pool",
    "get_pool",
    "resolve_dsn",
]
