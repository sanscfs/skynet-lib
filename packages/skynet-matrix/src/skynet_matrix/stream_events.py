"""Shared event schema for Redis Streams thought streaming.

Canonical source -- replaces the identical stream_events.py files
in skynet-agent and skynet-matrix-bridge.

Used by:
- Bridge: publishes Claude Code CLI events
- Agent: publishes its own LLM thinking/action events
- SRE: publishes analysis/execution events
- Consumers: Matrix live updater, agent watcher, metrics
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Stream event types."""

    # Claude Code CLI events (bridge)
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TEXT = "text"
    BLOCK = "block"
    COMPLETE = "complete"
    ERROR = "error"
    # Agent-specific events
    REASONING = "reasoning"
    ACTION = "action"
    ACTION_RESULT = "action_result"
    ITERATION = "iteration"


# Stream key patterns
BRIDGE_STREAM_PREFIX = "skynet:claude:session"
AGENT_STREAM_PREFIX = "skynet:agent:session"

# Consumer groups
BRIDGE_MATRIX_GROUP = "matrix-stream"
AGENT_WATCH_GROUP = "agent-watch"
AGENT_MATRIX_GROUP = "matrix-stream"
METRICS_GROUP = "metrics"


def stream_key(prefix: str, session_id: str) -> str:
    return f"{prefix}:{session_id}"


def new_session_id() -> str:
    return str(uuid.uuid4())


@dataclass
class StreamEvent:
    """A single event in a thought stream."""

    type: EventType
    session_id: str
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_redis_fields(self) -> dict[str, str]:
        """Serialize to flat dict for XADD."""
        fields = {
            "type": self.type.value,
            "session_id": self.session_id,
            "content": self.content[:8000],
            "timestamp": str(self.timestamp),
        }
        if self.metadata:
            fields["metadata"] = json.dumps(self.metadata, separators=(",", ":"))
        return fields

    @classmethod
    def from_redis_fields(cls, fields: dict[str, str]) -> StreamEvent:
        """Deserialize from Redis XREAD result."""
        metadata = {}
        if "metadata" in fields:
            try:
                metadata = json.loads(fields["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        return cls(
            type=EventType(fields.get("type", "text")),
            session_id=fields.get("session_id", ""),
            content=fields.get("content", ""),
            timestamp=float(fields.get("timestamp", 0)),
            metadata=metadata,
        )


def parse_xread_response(response: list | None) -> list[tuple[str, str, StreamEvent]]:
    """Parse XREAD/XREADGROUP response into (stream_key, message_id, event) tuples.

    Redis returns: [[stream_name, [[id, [f1, v1, f2, v2, ...]], ...]], ...]
    """
    if not response:
        return []
    results = []
    for stream_entry in response:
        skey = stream_entry[0] if isinstance(stream_entry[0], str) else stream_entry[0]
        messages = stream_entry[1] or []
        for msg in messages:
            msg_id = msg[0]
            flat = msg[1]
            if isinstance(flat, list):
                fields = dict(zip(flat[::2], flat[1::2]))
            elif isinstance(flat, dict):
                fields = flat
            else:
                continue
            try:
                event = StreamEvent.from_redis_fields(fields)
                results.append((skey, msg_id, event))
            except Exception:
                continue
    return results


# --- Matrix formatting helpers ---

_TYPE_ICONS = {
    EventType.THINKING: "\U0001f9e0",
    EventType.TOOL_CALL: "\U0001f527",
    EventType.TOOL_RESULT: "\U0001f4cb",
    EventType.TEXT: "\U0001f4ac",
    EventType.BLOCK: "\u26a0\ufe0f",
    EventType.COMPLETE: "\u2705",
    EventType.ERROR: "\u274c",
    EventType.REASONING: "\U0001f9e0",
    EventType.ACTION: "\u2699\ufe0f",
    EventType.ACTION_RESULT: "\U0001f4cb",
    EventType.ITERATION: "\U0001f504",
}


def format_event_for_matrix(event: StreamEvent, compact: bool = True) -> str:
    """Format a stream event for Matrix display."""
    icon = _TYPE_ICONS.get(event.type, "\U0001f4ac")

    if event.type == EventType.THINKING:
        text = event.content[:500] if compact else event.content
        return f"{icon} {text}"

    if event.type == EventType.TOOL_CALL:
        tool = event.metadata.get("tool_name", "?")
        preview = event.content[:200] if event.content else ""
        return f"{icon} `{tool}` {preview}"

    if event.type == EventType.TOOL_RESULT:
        tool = event.metadata.get("tool_name", "?")
        status = event.metadata.get("status", "")
        preview = event.content[:300] if event.content else ""
        return f"{icon} {tool} {status}: {preview}"

    if event.type == EventType.BLOCK:
        reason = event.metadata.get("block_reason", "needs input")
        return f"{icon} **BLOCKED**: {reason}\n{event.content}"

    if event.type in (EventType.REASONING, EventType.ITERATION):
        return f"{icon} {event.content[:500]}"

    if event.type == EventType.ACTION:
        cmd = event.metadata.get("command", event.content[:200])
        return f"{icon} `{cmd}`"

    if event.type == EventType.ACTION_RESULT:
        return f"{icon} {event.content[:400]}"

    if event.type == EventType.COMPLETE:
        dur = event.metadata.get("duration_s", 0)
        tokens = event.metadata.get("tokens_used", 0)
        return f"{icon} Done ({dur:.1f}s, {tokens} tokens)"

    if event.type == EventType.ERROR:
        return f"{icon} {event.content[:500]}"

    return f"{icon} {event.content[:300]}"


# Stream config defaults
DEFAULT_STREAM_MAXLEN = 1000
DEFAULT_STREAM_TTL_SECONDS = 7200
MATRIX_EDIT_DEBOUNCE_SECONDS = 2.5
