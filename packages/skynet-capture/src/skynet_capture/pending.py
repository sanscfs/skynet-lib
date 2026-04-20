"""Pending-capture state machine.

When identification confidence is below threshold, the bot posts a
clarifying question and stores a ``PendingCapture`` in Redis keyed on
the **bot's reply event_id**. When the user replies in that thread,
``PendingStore.pop(reply_event_id)`` returns the pending state and the
watcher task is cancelled.

Timeout:
  ``timeout_watcher`` is an async coroutine that sleeps N seconds, then
  pops the pending state and calls ``on_expire``. Callers create it with
  ``asyncio.create_task`` and pass the task to ``PendingStore.attach_watcher``
  so it gets cancelled automatically on ``pop``.

Pod-restart resilience:
  Watcher tasks are in-memory only. A restart loses the task but the
  Redis TTL eventually expires; the user just gets no "time expired"
  message. This is acceptable for a 5-minute window.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from skynet_capture.sources.base import Candidate

log = logging.getLogger("skynet_capture.pending")


@dataclass
class PendingCapture:
    domain: str  # "music" | "movies"
    action: str  # "confirm_entity"
    candidates: list[Candidate]
    context: dict[str, Any] = field(default_factory=dict)


class PendingStore:
    def __init__(self, redis: Any, *, prefix: str = "skynet:capture:pending") -> None:
        self._redis = redis
        self._prefix = prefix
        self._tasks: dict[str, asyncio.Task] = {}

    def _key(self, event_id: str) -> str:
        return f"{self._prefix}:{event_id}"

    async def save(self, event_id: str, pending: PendingCapture, *, ttl: int = 300) -> None:
        payload = json.dumps(
            {
                "domain": pending.domain,
                "action": pending.action,
                "candidates": [_cand_to_dict(c) for c in pending.candidates],
                "context": pending.context,
            }
        )
        await self._redis.set(self._key(event_id), payload, ex=ttl)

    async def pop(self, event_id: str) -> PendingCapture | None:
        key = self._key(event_id)
        pipe = self._redis.pipeline()
        pipe.get(key)
        pipe.delete(key)
        results = await pipe.execute()
        raw = results[0]

        task = self._tasks.pop(event_id, None)
        if task and not task.done():
            task.cancel()

        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return PendingCapture(
                domain=data["domain"],
                action=data["action"],
                candidates=[_dict_to_cand(c) for c in data.get("candidates", [])],
                context=data.get("context", {}),
            )
        except Exception as exc:
            log.warning("PendingStore.pop: corrupt payload for %s: %s", event_id, exc)
            return None

    def attach_watcher(self, event_id: str, task: asyncio.Task) -> None:
        self._tasks[event_id] = task


async def timeout_watcher(
    *,
    store: PendingStore,
    pending_key: str,
    after_seconds: int,
    on_expire: Callable[[], Awaitable[None]],
) -> None:
    try:
        await asyncio.sleep(after_seconds)
        pending = await store.pop(pending_key)
        if pending is not None:
            await on_expire()
    except asyncio.CancelledError:
        pass


def _cand_to_dict(c: Candidate) -> dict:
    return {
        "source_id": c.source_id,
        "title": c.title,
        "subtitle": c.subtitle,
        "year": c.year,
        "type": c.type,
        "url": c.url,
        "cover_url": c.cover_url,
    }


def _dict_to_cand(d: dict) -> Candidate:
    from skynet_capture.sources.base import Candidate as C

    return C(
        source_id=d["source_id"],
        title=d["title"],
        subtitle=d.get("subtitle", ""),
        year=d.get("year"),
        type=d.get("type", "album"),  # type: ignore[arg-type]
        url=d.get("url", ""),
        cover_url=d.get("cover_url"),
    )


__all__ = ["PendingCapture", "PendingStore", "timeout_watcher"]
