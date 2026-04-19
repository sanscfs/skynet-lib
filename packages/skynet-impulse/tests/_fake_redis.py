"""In-memory stand-in for a redis-py client.

Implements only the commands the impulse engine actually uses:
``hgetall / hset / hdel / hget / hincrby / rpush / ltrim / lrange / llen /
set / get / zadd / zcount / zrevrange / zremrangebyscore``. Returns str (not
bytes) so callers exercise the ``decode_responses=True`` code path.
"""

from __future__ import annotations

import bisect
from typing import Any


class FakeRedis:
    def __init__(self) -> None:
        self._kv: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._lists: dict[str, list[str]] = {}
        self._zsets: dict[str, list[tuple[float, str]]] = {}

    # ---- keys / strings --------------------------------------------------
    def set(self, key: str, value: Any) -> bool:
        self._kv[key] = str(value)
        return True

    def get(self, key: str) -> str | None:
        return self._kv.get(key)

    # ---- hashes ----------------------------------------------------------
    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    def hset(self, key: str, field: str | None = None, value: Any = None, *,
             mapping: dict[str, Any] | None = None) -> int:
        h = self._hashes.setdefault(key, {})
        added = 0
        if mapping:
            for k, v in mapping.items():
                if k not in h:
                    added += 1
                h[k] = str(v)
        if field is not None:
            if field not in h:
                added += 1
            h[field] = str(value)
        return added

    def hget(self, key: str, field: str) -> str | None:
        return self._hashes.get(key, {}).get(field)

    def hincrby(self, key: str, field: str, increment: int = 1) -> int:
        h = self._hashes.setdefault(key, {})
        current = int(h.get(field, "0"))
        current += increment
        h[field] = str(current)
        return current

    def hdel(self, key: str, *fields: str) -> int:
        h = self._hashes.get(key, {})
        removed = 0
        for f in fields:
            if f in h:
                del h[f]
                removed += 1
        return removed

    # ---- lists -----------------------------------------------------------
    def rpush(self, key: str, *values: Any) -> int:
        lst = self._lists.setdefault(key, [])
        for v in values:
            lst.append(str(v))
        return len(lst)

    def ltrim(self, key: str, start: int, stop: int) -> bool:
        lst = self._lists.get(key, [])
        # Python slicing semantics mirror Redis LTRIM inclusive range.
        if stop == -1:
            self._lists[key] = lst[start:]
        else:
            self._lists[key] = lst[start : stop + 1 if stop >= 0 else stop + 1 or None]
        return True

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        lst = self._lists.get(key, [])
        if stop == -1:
            return lst[start:]
        return lst[start : stop + 1]

    def llen(self, key: str) -> int:
        return len(self._lists.get(key, []))

    # ---- sorted sets -----------------------------------------------------
    def zadd(self, key: str, mapping: dict[str, float]) -> int:
        zset = self._zsets.setdefault(key, [])
        existing = {m for _score, m in zset}
        for member, score in mapping.items():
            # Remove old entry with same member.
            zset[:] = [(s, m) for s, m in zset if m != member]
            bisect.insort(zset, (float(score), member))
        return len([m for m in mapping if m not in existing])

    def zcount(self, key: str, min_: float | str, max_: float | str) -> int:
        zset = self._zsets.get(key, [])
        lo = float("-inf") if min_ == "-inf" else float(min_)
        hi = float("inf") if max_ == "+inf" else float(max_)
        return sum(1 for score, _ in zset if lo <= score <= hi)

    def zrevrange(self, key: str, start: int, stop: int, withscores: bool = False):
        zset = sorted(self._zsets.get(key, []), reverse=True)
        if stop == -1:
            sl = zset[start:]
        else:
            sl = zset[start : stop + 1]
        if withscores:
            return [(m, s) for s, m in sl]
        return [m for _s, m in sl]

    def zremrangebyscore(self, key: str, min_: float, max_: float) -> int:
        zset = self._zsets.get(key, [])
        before = len(zset)
        self._zsets[key] = [(s, m) for s, m in zset if not (min_ <= s <= max_)]
        return before - len(self._zsets[key])
