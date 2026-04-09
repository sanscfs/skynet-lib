"""Skynet Matrix -- Matrix client, stream events, and live streaming."""

from skynet_matrix.client import MatrixClient
from skynet_matrix.async_client import AsyncMatrixClient
from skynet_matrix.stream_events import (
    EventType,
    StreamEvent,
    parse_xread_response,
    format_event_for_matrix,
    stream_key,
    new_session_id,
    BRIDGE_STREAM_PREFIX,
    AGENT_STREAM_PREFIX,
    BRIDGE_MATRIX_GROUP,
    AGENT_WATCH_GROUP,
    AGENT_MATRIX_GROUP,
    METRICS_GROUP,
    DEFAULT_STREAM_MAXLEN,
    DEFAULT_STREAM_TTL_SECONDS,
    MATRIX_EDIT_DEBOUNCE_SECONDS,
)

__all__ = [
    "MatrixClient",
    "AsyncMatrixClient",
    "EventType",
    "StreamEvent",
    "parse_xread_response",
    "format_event_for_matrix",
    "stream_key",
    "new_session_id",
    "BRIDGE_STREAM_PREFIX",
    "AGENT_STREAM_PREFIX",
    "BRIDGE_MATRIX_GROUP",
    "AGENT_WATCH_GROUP",
    "AGENT_MATRIX_GROUP",
    "METRICS_GROUP",
    "DEFAULT_STREAM_MAXLEN",
    "DEFAULT_STREAM_TTL_SECONDS",
    "MATRIX_EDIT_DEBOUNCE_SECONDS",
]
