"""Thread streaming integration.

The orchestration library doesn't *own* Matrix streaming -- that's
``skynet-matrix.LiveStream``. But every sub-agent should stream its
events into the same Matrix thread the user opened with main, using
its own MXID for attribution. This module is the thin adapter:

- :func:`open_thread_stream` builds a ``LiveStream`` bound to the
  ThreadHandle from an AgentCall.
- :func:`emit_to_thread` posts one AgentEvent into that stream.

skynet-matrix is an *optional* dependency (extra ``[matrix]``)
because some sub-agents might run in environments without Matrix
access -- a CronJob-style background runner, for example. When the
import fails the helpers degrade to no-ops with a warning.

Design choice: each service gets ONE LiveStream per invocation,
keyed by ``invocation_id``. Concurrent edits on the same Matrix
event are serialised inside LiveStream, so two sub-agents in the
same thread won't race each other -- their MXIDs are different and
each owns its own "thinking..." message.
"""

from __future__ import annotations

import logging
import threading

from .envelopes import AgentEvent, ThreadHandle

log = logging.getLogger("skynet_orchestration.streaming")


# Per-invocation LiveStream cache. Keys are ``invocation_id`` -> stream.
_streams: dict[str, "object"] = {}
_streams_lock = threading.Lock()


def open_thread_stream(
    handle: ThreadHandle,
    *,
    invocation_id: str,
    service_name: str,
):
    """Create or retrieve the LiveStream for an invocation.

    Returns ``None`` when ``skynet-matrix`` isn't installed or the
    ThreadHandle is missing required fields. Callers must always
    handle the None case -- streaming is best-effort.
    """
    if not handle.room_id or not handle.thread_root:
        return None
    with _streams_lock:
        cached = _streams.get(invocation_id)
        if cached is not None:
            return cached
        try:
            from skynet_matrix.live_stream import LiveStream  # type: ignore
        except ImportError:
            log.debug("skynet-matrix not installed; thread streaming disabled")
            return None
        try:
            stream = LiveStream(
                room_id=handle.room_id,
                thread_root=handle.thread_root,
                # The service tag becomes the prefix on every emit so
                # @skynet-sre vs @skynet-music is visible in the
                # thread without each service having to add it.
                service=service_name,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("failed to open LiveStream: %s", e)
            return None
        _streams[invocation_id] = stream
        return stream


def emit_to_thread(
    handle: ThreadHandle,
    event: AgentEvent,
    *,
    service_name: str,
) -> None:
    """Post one AgentEvent into the invocation's thread.

    The stream is created lazily on first emit. Failures are logged
    and swallowed -- dropping a stream message never breaks the
    handler.
    """
    stream = open_thread_stream(handle, invocation_id=event.invocation_id, service_name=service_name)
    if stream is None:
        return
    try:
        # LiveStream's emit takes (event_type, body, metadata=...).
        # Translate orchestration EventType to the existing matrix
        # EventType values where possible -- the ones we use here
        # already exist (see streaming.py inside skynet-matrix).
        from skynet_matrix.stream_events import EventType  # type: ignore

        et_map = {
            "progress": EventType.THINKING,
            "tool_call": EventType.TOOL_CALL,
            "tool_result": EventType.TOOL_RESULT,
            "found": EventType.THINKING,
            "thinking": EventType.THINKING,
            "conclusion": EventType.COMPLETE,
            "budget_request": EventType.THINKING,
            "escalation_request": EventType.THINKING,
            "warning": EventType.ERROR,
        }
        mapped = et_map.get(event.type, EventType.THINKING)
        stream.emit(mapped, event.content, metadata=event.metadata)
    except Exception as e:  # noqa: BLE001
        log.warning("thread emit failed for %s: %s", event.invocation_id, e)


def close_stream(invocation_id: str) -> None:
    """Drop the stream cache entry for a finished invocation.

    The underlying Matrix message is left in place. We just stop
    tracking it so the dict doesn't grow forever.
    """
    with _streams_lock:
        _streams.pop(invocation_id, None)


def _reset_for_tests() -> None:
    """Test-only: clear the in-process stream cache."""
    with _streams_lock:
        _streams.clear()
