"""LiveStream -- debounced Matrix message editing for streaming agent output.

Consolidates the streaming pattern from skynet-agent and skynet-sre:
emit events to Redis Streams while editing a Matrix message with
debounced updates to avoid rate limits.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from skynet_matrix.stream_events import (
    DEFAULT_STREAM_MAXLEN,
    DEFAULT_STREAM_TTL_SECONDS,
    MATRIX_EDIT_DEBOUNCE_SECONDS,
    EventType,
    StreamEvent,
    format_event_for_matrix,
    stream_key,
)

logger = logging.getLogger(__name__)


class LiveStream:
    """Manages a live-updating Matrix message backed by Redis Streams.

    Usage:
        stream = LiveStream(redis, matrix_client, room_id, session_id)
        stream.start("Processing...")
        stream.emit(EventType.THINKING, "Analyzing the request...")
        stream.emit(EventType.TOOL_CALL, "kubectl get pods", metadata={"tool_name": "bash"})
        stream.complete(duration_s=12.3, tokens=1500)
    """

    def __init__(
        self,
        redis_client,
        matrix_client,
        room_id: str,
        session_id: str,
        *,
        stream_prefix: str = "skynet:agent:session",
        maxlen: int = DEFAULT_STREAM_MAXLEN,
        ttl_seconds: int = DEFAULT_STREAM_TTL_SECONDS,
        debounce_seconds: float = MATRIX_EDIT_DEBOUNCE_SECONDS,
    ):
        self.redis = redis_client
        self.matrix = matrix_client
        self.room_id = room_id
        self.session_id = session_id
        self.skey = stream_key(stream_prefix, session_id)
        self.maxlen = maxlen
        self.ttl_seconds = ttl_seconds
        self.debounce = debounce_seconds
        self._message_eid: str | None = None
        self._last_edit: float = 0
        self._display_lines: list[str] = []

    def start(self, initial_text: str = "Thinking...") -> str | None:
        """Send the initial Matrix message. Returns event_id."""
        self._message_eid = self.matrix.send_text(self.room_id, initial_text)
        return self._message_eid

    def emit(
        self,
        event_type: EventType,
        content: str = "",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit an event to Redis Streams and update the Matrix message."""
        evt = StreamEvent(
            type=event_type,
            session_id=self.session_id,
            content=content,
            metadata=metadata or {},
        )

        # Write to Redis Stream
        if self.redis is not None:
            try:
                self.redis.execute_command(
                    "XADD",
                    self.skey,
                    "MAXLEN",
                    "~",
                    str(self.maxlen),
                    "*",
                    *sum(evt.to_redis_fields().items(), ()),
                )
            except Exception as e:
                logger.debug("Stream XADD failed: %s", e)

        # Update display
        line = format_event_for_matrix(evt)
        self._display_lines.append(line)

        # Debounced Matrix edit
        now = time.time()
        if self._message_eid and now - self._last_edit >= self.debounce:
            display = "\n".join(self._display_lines[-20:])
            self.matrix.edit_message(self.room_id, self._message_eid, display)
            self._last_edit = now

    def complete(
        self,
        duration_s: float = 0,
        tokens: int = 0,
        *,
        final_text: str | None = None,
    ) -> None:
        """Emit COMPLETE event and finalize the Matrix message."""
        self.emit(
            EventType.COMPLETE,
            metadata={"duration_s": duration_s, "tokens_used": tokens},
        )

        # Final edit
        if self._message_eid:
            if final_text:
                self.matrix.edit_message(self.room_id, self._message_eid, final_text)
            else:
                display = "\n".join(self._display_lines[-20:])
                self.matrix.edit_message(self.room_id, self._message_eid, display)

        # Set TTL on stream
        if self.redis is not None:
            try:
                self.redis.expire(self.skey, self.ttl_seconds)
            except Exception:
                pass
