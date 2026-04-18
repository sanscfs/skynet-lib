"""Skynet Matrix -- Matrix client, stream events, and live streaming."""

from skynet_matrix.async_client import AsyncMatrixClient
from skynet_matrix.client import MatrixClient
from skynet_matrix.stream_events import (
    AGENT_MATRIX_GROUP,
    AGENT_STREAM_PREFIX,
    AGENT_WATCH_GROUP,
    BRIDGE_MATRIX_GROUP,
    BRIDGE_STREAM_PREFIX,
    DEFAULT_STREAM_MAXLEN,
    DEFAULT_STREAM_TTL_SECONDS,
    MATRIX_EDIT_DEBOUNCE_SECONDS,
    METRICS_GROUP,
    EventType,
    StreamEvent,
    format_event_for_matrix,
    new_session_id,
    parse_xread_response,
    stream_key,
)
from skynet_matrix.trace_footer import build_trace_meta, format_trace_footer

__all__ = [
    "MatrixClient",
    "AsyncMatrixClient",
    "format_trace_footer",
    "build_trace_meta",
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
