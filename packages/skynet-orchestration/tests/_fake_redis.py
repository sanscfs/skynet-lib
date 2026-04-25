"""Minimal in-memory Redis stand-in for skynet-orchestration tests.

Only implements the commands the package actually uses. Crucially
includes a ``pipeline()`` mock that buffers commands and replays
them on ``execute()`` -- the budget module relies on that pattern.
``xadd`` is a no-op that returns a synthetic id so the chronicle
helper doesn't crash in tests.
"""

from __future__ import annotations

from typing import Any


class _Pipeline:
    def __init__(self, parent: "FakeRedis") -> None:
        self._parent = parent
        self._ops: list[tuple[str, tuple, dict]] = []

    def hsetnx(self, *args, **kwargs):
        self._ops.append(("hsetnx", args, kwargs))
        return self

    def hincrby(self, *args, **kwargs):
        self._ops.append(("hincrby", args, kwargs))
        return self

    def hset(self, *args, **kwargs):
        self._ops.append(("hset", args, kwargs))
        return self

    def expire(self, *args, **kwargs):
        self._ops.append(("expire", args, kwargs))
        return self

    def rpush(self, *args, **kwargs):
        self._ops.append(("rpush", args, kwargs))
        return self

    def ltrim(self, *args, **kwargs):
        self._ops.append(("ltrim", args, kwargs))
        return self

    def lrange(self, *args, **kwargs):
        self._ops.append(("lrange", args, kwargs))
        return self

    def execute(self):
        results = []
        for name, args, kwargs in self._ops:
            method = getattr(self._parent, name)
            results.append(method(*args, **kwargs))
        self._ops.clear()
        return results


class FakeRedis:
    def __init__(self) -> None:
        self._kv: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._lists: dict[str, list[str]] = {}
        self._streams: dict[str, list[dict]] = {}
        # ZSET storage: {key: {member: score}}
        self._zsets: dict[str, dict[str, float]] = {}

    # ---- pipelines -------------------------------------------------------
    def pipeline(self) -> _Pipeline:
        return _Pipeline(self)

    # ---- strings ---------------------------------------------------------
    def set(self, key: str, value: Any) -> bool:
        self._kv[key] = str(value)
        return True

    def get(self, key: str) -> str | None:
        return self._kv.get(key)

    def incr(self, key: str) -> int:
        cur = int(self._kv.get(key, "0"))
        cur += 1
        self._kv[key] = str(cur)
        return cur

    def delete(self, *keys: str) -> int:
        n = 0
        for k in keys:
            for store in (self._kv, self._hashes, self._lists, self._zsets):
                if k in store:
                    del store[k]
                    n += 1
        return n

    def type(self, key: str) -> str:
        if key in self._kv:
            return "string"
        if key in self._hashes:
            return "hash"
        if key in self._lists:
            return "list"
        if key in self._zsets:
            return "zset"
        if key in self._streams:
            return "stream"
        return "none"

    def expire(self, key: str, seconds: int) -> bool:
        return key in self._kv or key in self._hashes or key in self._lists or key in self._zsets

    # ---- hashes ----------------------------------------------------------
    def hsetnx(self, key: str, field: str, value: Any) -> int:
        h = self._hashes.setdefault(key, {})
        if field in h:
            return 0
        h[field] = str(value)
        return 1

    def hset(
        self, key: str, field: str | None = None, value: Any = None, *, mapping: dict[str, Any] | None = None
    ) -> int:
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

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

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

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        lst = self._lists.get(key, [])
        if stop == -1:
            return lst[start:]
        return lst[start : stop + 1]

    def ltrim(self, key: str, start: int, stop: int) -> bool:
        lst = self._lists.get(key, [])
        if stop == -1:
            self._lists[key] = lst[start:]
        elif start < 0 and stop < 0:
            self._lists[key] = lst[start:] if stop == -1 else lst[start : stop + 1]
        else:
            self._lists[key] = lst[start : stop + 1]
        return True

    def llen(self, key: str) -> int:
        return len(self._lists.get(key, []))

    # ---- sorted sets -----------------------------------------------------
    def zadd(self, key: str, mapping: dict[str, float]) -> int:
        z = self._zsets.setdefault(key, {})
        added = 0
        for member, score in mapping.items():
            if member not in z:
                added += 1
            z[member] = float(score)
        return added

    def zrange(
        self,
        key: str,
        start: int,
        stop: int,
        *,
        withscores: bool = False,
    ):
        z = self._zsets.get(key, {})
        items = sorted(z.items(), key=lambda x: (x[1], x[0]))
        if stop == -1:
            sliced = items[start:]
        else:
            sliced = items[start : stop + 1]
        if withscores:
            return [(m, s) for m, s in sliced]
        return [m for m, _ in sliced]

    def zremrangebyscore(self, key: str, min_score, max_score) -> int:
        """Mimic Redis ZREMRANGEBYSCORE — supports ``"-inf"`` / ``"+inf"``
        and Redis's ``"(N"`` exclusive notation."""
        z = self._zsets.get(key)
        if not z:
            return 0

        def _parse(b, default):
            if isinstance(b, (int, float)):
                return float(b), False
            s = str(b)
            if s == "-inf":
                return float("-inf"), False
            if s == "+inf" or s == "inf":
                return float("inf"), False
            exclusive = s.startswith("(")
            if exclusive:
                s = s[1:]
            return float(s), exclusive

        lo, lo_excl = _parse(min_score, float("-inf"))
        hi, hi_excl = _parse(max_score, float("inf"))

        to_remove = []
        for member, score in z.items():
            if lo_excl:
                if score <= lo:
                    continue
            else:
                if score < lo:
                    continue
            if hi_excl:
                if score >= hi:
                    continue
            else:
                if score > hi:
                    continue
            to_remove.append(member)
        for m in to_remove:
            del z[m]
        return len(to_remove)

    def zcard(self, key: str) -> int:
        return len(self._zsets.get(key, {}))

    # ---- streams (chronicle uses xadd only) ------------------------------
    def xadd(self, stream: str, fields: dict, *, maxlen: int | None = None, approximate: bool = True) -> str:
        s = self._streams.setdefault(stream, [])
        s.append(dict(fields))
        if maxlen and len(s) > maxlen:
            del s[: len(s) - maxlen]
        return f"{len(s)}-0"

    def stream_entries(self, stream: str) -> list[dict]:
        """Test helper -- not a Redis command."""
        return list(self._streams.get(stream, []))
