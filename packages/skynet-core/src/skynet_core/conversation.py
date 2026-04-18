"""Redis-backed conversation history (per room / per thread).

Extracted from skynet-agent so any component (agent, impulse, DAGs)
can track Matrix conversation context with correct thread support.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable

from skynet_core.redis import get_redis

logger = logging.getLogger(__name__)

CONV_MAX_MESSAGES = int(os.getenv("CONV_MAX_MESSAGES", "40"))
CONV_COMPACT_THRESHOLD = int(os.getenv("CONV_COMPACT_THRESHOLD", "30"))
CONV_COMPACT_KEEP = int(os.getenv("CONV_COMPACT_KEEP", "15"))


def conv_key(room_id: str, thread_root: str | None = None) -> str:
    """Redis key for a conversation: main room or specific thread."""
    if thread_root:
        return f"skynet:conv:{room_id}:thread:{thread_root}"
    return f"skynet:conv:{room_id}:main"


def conv_load(room_id: str, thread_root: str | None = None,
              max_messages: int | None = None) -> list[dict]:
    """Load conversation history from Redis.

    Returns only entries with ctx=True (default). System notifications,
    tool noise, and other non-conversational messages are excluded.
    """
    max_msg = max_messages or CONV_MAX_MESSAGES
    try:
        r = get_redis()
        raw = r.lrange(conv_key(room_id, thread_root), 0, max_msg - 1)
        entries = []
        for item in reversed(raw):
            try:
                entry = json.loads(item)
                if entry.get("ctx", True):
                    entries.append(entry)
            except Exception:
                pass
        return entries
    except Exception as e:
        logger.warning("conv_load failed: %s", e)
        return []


def conv_append(room_id: str, role: str, content: str, sender: str = "",
                thread_root: str | None = None,
                include_in_context: bool = True,
                max_messages: int | None = None) -> None:
    """Append a message to conversation history.

    Args:
        thread_root: Matrix event_id of the thread root. When set, the
            message is stored in a thread-specific Redis key so that
            conv_load(thread_root=...) finds it later.
        include_in_context: if False, message is stored but skipped by
            conv_load. Use for system notifications, tool output noise.
    """
    max_msg = max_messages or CONV_MAX_MESSAGES
    try:
        r = get_redis()
        key = conv_key(room_id, thread_root)
        entry = json.dumps({
            "role": role,
            "content": content[:20000],
            "sender": sender,
            "ts": int(time.time()),
            "ctx": include_in_context,
        })
        r.lpush(key, entry)
        r.ltrim(key, 0, max_msg - 1)
    except Exception as e:
        logger.warning("conv_append failed: %s", e)


def conv_compact(room_id: str, thread_root: str | None = None,
                 summarize_fn: Callable[[str], str] | None = None) -> None:
    """Compact old messages into a summary when conversation exceeds threshold.

    Args:
        summarize_fn: takes concatenated old-message text, returns a summary
            string. When not provided, falls back to simple concatenation.
    """
    try:
        r = get_redis()
        key = conv_key(room_id, thread_root)
        total = r.llen(key)
        if total <= CONV_COMPACT_THRESHOLD:
            return
        n_old = total - CONV_COMPACT_KEEP
        if n_old < 5:
            return
        old_raw = r.lrange(key, CONV_COMPACT_KEEP, -1)
        old_msgs = []
        for item in old_raw:
            try:
                e = json.loads(item)
                if e.get("ctx", True):
                    old_msgs.append(e)
            except Exception:
                pass
        if not old_msgs:
            return

        lines = [f"{m.get('role', '?')}: {m.get('content', '')[:200]}"
                 for m in old_msgs]
        summary_input = "\n".join(lines)

        if summarize_fn:
            try:
                summary = summarize_fn(summary_input[:4000])
            except Exception:
                summary = _fallback_summary(old_msgs)
        else:
            summary = _fallback_summary(old_msgs)

        r.ltrim(key, 0, CONV_COMPACT_KEEP - 1)
        r.rpush(key, json.dumps({
            "role": "system",
            "content": f"[Summary of {len(old_msgs)} earlier messages]: {summary}",
            "ts": int(time.time()),
            "ctx": True,
        }))
        logger.info("Compacted %d messages in %s", len(old_msgs), key)
    except Exception as e:
        logger.warning("conv_compact failed: %s", e)


def _fallback_summary(msgs: list[dict]) -> str:
    return "; ".join(
        f"{m.get('role', '')}: {m.get('content', '')[:80]}"
        for m in msgs[:10]
    )
