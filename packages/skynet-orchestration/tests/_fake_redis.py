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

    # ---- pipelines -------------------------------------------------------
    def pipeline(self) -> _Pipeline:
        return _Pipeline(self)

    # ---- strings ---------------------------------------------------------
    def set(self, key: str, value: Any) -> bool:
        self._kv[key] = str(value)
        return True

    def get(self, key: str) -> str | None:
        return self._kv.get(key)

    def expire(self, key: str, seconds: int) -> bool:
        return key in self._kv or key in self._hashes or key in self._lists

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
