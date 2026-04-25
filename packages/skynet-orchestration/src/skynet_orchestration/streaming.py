"""Thread streaming integration.

The orchestration library doesn't *own* Matrix streaming -- that's
``skynet-matrix.LiveStream``. But every sub-agent should stream its
events into the same Matrix thread the user opened with main, using
its own MXID for attribution. This module is the thin adapter:

- :func:`configure_streaming` lets a service plug in a ``factory``
  that knows how to build a stream from its own (redis + matrix)
  pair. Only the service has those handles, so the library can't
  hard-code the constructor.
- :func:`open_thread_stream` calls the configured factory once per
  invocation and caches the result keyed by ``invocation_id``.
- :func:`emit_to_thread` posts one AgentEvent into that stream.
- :func:`close_stream` drops the cache entry on invocation exit.

skynet-matrix is an *optional* dependency (extra ``[matrix]``)
because some sub-agents might run in environments without Matrix
access -- a CronJob-style background runner, for example. When no
factory is configured (or the factory returns ``None`` for a given
handle, e.g. service has no Matrix client wired), the helpers
degrade to no-ops with a debug log.

Stream contract
~~~~~~~~~~~~~~~

The factory must return *something* satisfying::

    stream.emit(event_type, content, *, metadata) -> None

where ``event_type`` is a value from ``skynet_matrix.stream_events.EventType``
(the orchestration library does the mapping from ``AgentEvent.type``
to that enum before calling). ``skynet_matrix.LiveStream`` already
satisfies this contract; services may also pass a thin custom adapter
if they want to stream into something other than Matrix.

The factory may return ``None`` to signal "not streamable for this
handle" (e.g. ThreadHandle has no room_id, service can't post to
that room, or the service intentionally doesn't surface inline
updates). Library treats ``None`` as no-op.

Design choice: each service gets ONE stream per invocation, keyed by
``invocation_id``. Concurrent edits on the same Matrix event are
serialised inside LiveStream, so two sub-agents in the same thread
won't race each other -- their MXIDs are different and each owns its
own "thinking..." message.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from .envelopes import AgentEvent, ThreadHandle

log = logging.getLogger("skynet_orchestration.streaming")


# Per-invocation stream cache. Keys are ``invocation_id`` -> stream.
_streams: dict[str, Any] = {}
_streams_lock = threading.Lock()

# Factory: callable that builds a stream-shaped object from a
# ThreadHandle + invocation_id, or returns None when the service
# can't / won't stream this invocation. Set via configure_streaming.
StreamFactory = Callable[[ThreadHandle, str], Optional[Any]]
_factory: Optional[StreamFactory] = None
_factory_lock = threading.Lock()


def configure_streaming(factory: Optional[StreamFactory]) -> None:
    """Inject the per-service stream factory.

    Called once at service startup, after the service has its own
    redis client + matrix client (or whatever it streams into) wired.
    Pass ``None`` to clear the factory (used by tests).

    The factory signature is ``(handle, invocation_id) -> stream | None``.
    Any object the factory returns must implement::

        emit(event_type, content, *, metadata) -> None

    Returning ``None`` from the factory is a valid "this service can't
    stream this handle" signal -- emit_to_thread will simply skip
    posting and the chronicle write still happens.
    """

    global _factory
    with _factory_lock:
        _factory = factory


def open_thread_stream(
    handle: ThreadHandle,
    *,
    invocation_id: str,
    service_name: str,
) -> Optional[Any]:
    """Create or retrieve the stream for an invocation.

    Looks up the cache first; on miss, calls the configured factory.
    If no factory is configured, or the factory returns ``None``, this
    function returns ``None`` and the caller short-circuits.

    The cache is keyed by ``invocation_id`` only (not also by
    ``service_name``) because each service runs its own
    ``configure_streaming`` and its own process: cache collisions
    across services don't happen.
    """

    factory = _factory
    if factory is None:
        # No streaming wired: return None -- chronicle path still runs.
        return None

    with _streams_lock:
        cached = _streams.get(invocation_id)
        if cached is not None:
            return cached

    try:
        stream = factory(handle, invocation_id)
    except Exception as e:  # noqa: BLE001
        log.warning(
            "stream factory raised for invocation=%s service=%s: %s",
            invocation_id,
            service_name,
            e,
        )
        return None

    if stream is None:
        return None

    with _streams_lock:
        # Re-check under the lock in case a concurrent call lost the race.
        existing = _streams.get(invocation_id)
        if existing is not None:
            return existing
        _streams[invocation_id] = stream
        return stream


def emit_to_thread(
    handle: ThreadHandle,
    event: AgentEvent,
    *,
    service_name: str,
) -> None:
    """Post one AgentEvent into the invocation's thread.

    The stream is created lazily on first emit via the configured
    factory. Failures are logged and swallowed -- dropping a stream
    message never breaks the handler.
    """
    stream = open_thread_stream(handle, invocation_id=event.invocation_id, service_name=service_name)
    if stream is None:
        return
    try:
        # Translate orchestration EventType to skynet-matrix EventType
        # values; that's what LiveStream-shaped consumers expect. We
        # do the import lazily because skynet-matrix is in the [matrix]
        # extra and we don't want to force the dep on consumers that
        # configure a non-matrix factory.
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
    tracking it so the dict doesn't grow forever. Best-effort calls
    ``stream.complete()`` if the stream exposes it (LiveStream does);
    failures are silently swallowed -- finalisation cleanup must
    never break the handler return path.
    """

    with _streams_lock:
        stream = _streams.pop(invocation_id, None)
    if stream is None:
        return
    complete_fn = getattr(stream, "complete", None)
    if callable(complete_fn):
        try:
            complete_fn()
        except Exception as e:  # noqa: BLE001
            log.debug("stream.complete() failed for %s: %s", invocation_id, e)


def _reset_for_tests() -> None:
    """Test-only: clear the factory + in-process stream cache."""
    global _factory
    with _factory_lock:
        _factory = None
    with _streams_lock:
        _streams.clear()
