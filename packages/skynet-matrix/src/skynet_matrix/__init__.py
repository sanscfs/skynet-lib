"""Skynet Matrix -- Matrix client, stream events, live streaming, command bot."""

from skynet_matrix.async_client import AsyncMatrixClient
from skynet_matrix.bot import BotConfig, CommandBot, OnTextCallback, OnThreadReplyCallback
from skynet_matrix.chat_agent import ChatAgent, HistoryLLMCaller, LLMCaller, ToolDispatch, ToolSchema
from skynet_matrix.history_llm import build_conv_history
from skynet_matrix.client import MatrixClient
from skynet_matrix.commands import Command, parse_command_line
from skynet_matrix.state_events import (
    STATE_EVENT_TYPE,
    build_bot_commands_content,
    publish_bot_commands_state,
)
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
from skynet_matrix.wrap import build_edit_payload, build_footer_payload

__all__ = [
    "MatrixClient",
    "AsyncMatrixClient",
    "CommandBot",
    "OnTextCallback",
    "OnThreadReplyCallback",
    "BotConfig",
    "ChatAgent",
    "LLMCaller",
    "HistoryLLMCaller",
    "ToolDispatch",
    "ToolSchema",
    "build_conv_history",
    "Command",
    "parse_command_line",
    "publish_bot_commands_state",
    "build_bot_commands_content",
    "STATE_EVENT_TYPE",
    "format_trace_footer",
    "build_trace_meta",
    "build_footer_payload",
    "build_edit_payload",
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
