"""Tests for the streaming factory mechanism.

Phase 6 wiring: ``configure_streaming(factory)`` lets each service
plug in its own LiveStream constructor (which knows the redis +
matrix client pair the library has no business holding). Verifies:

- ``configure_streaming(None)`` -> emit_to_thread no-ops cleanly,
  no exception even when skynet-matrix isn't importable.
- A configured factory is called once per invocation and the result
  is cached: a second emit on the same invocation reuses the stream.
- Factory returning ``None`` is treated as "not streamable for this
  handle" -- chronicle path is unaffected, no exception.
- Factory exceptions are caught and logged, never propagate.
- ``close_stream`` evicts the cache entry and best-effort calls
  ``.complete()`` on the cached stream.
"""

from __future__ import annotations

import pytest
from skynet_orchestration.envelopes import AgentEvent, ThreadHandle
from skynet_orchestration.streaming import (
    _reset_for_tests,
    close_stream,
    configure_streaming,
    emit_to_thread,
    open_thread_stream,
)


@pytest.fixture(autouse=True)
def _clear_streaming_state():
    _reset_for_tests()
    yield
    _reset_for_tests()


class _FakeStream:
    """Minimal LiveStream-shaped object: only ``emit`` is required.

    Records every emit call so the tests can assert what flowed
    through. ``complete`` is also recorded so the close_stream test
    can verify finalisation runs.
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.emits: list[tuple[object, str, dict]] = []
        self.complete_called = False

    def emit(self, event_type, content, *, metadata) -> None:
        self.emits.append((event_type, content, dict(metadata or {})))

    def complete(self) -> None:
        self.complete_called = True


def _handle() -> ThreadHandle:
    return ThreadHandle(room_id="!room:matrix.example", thread_root="$root")


def _event(invocation_id: str = "inv-1", typ: str = "progress") -> AgentEvent:
    return AgentEvent(
        invocation_id=invocation_id,
        type=typ,  # type: ignore[arg-type]
        content="working",
        metadata={"k": "v"},
        ts=1700000000.0,
    )


def test_no_factory_emits_are_noops():
    """Without a factory, emit_to_thread silently skips."""
    # Factory was reset in the fixture; nothing is wired.
    emit_to_thread(_handle(), _event(), service_name="sre")
    # Nothing to assert beyond "no exception"; chronicle path is separate.


def test_factory_called_once_per_invocation():
    """Same invocation_id reuses the cached stream on second emit."""
    calls: list[str] = []
    fake = _FakeStream("only-stream")

    def factory(handle: ThreadHandle, invocation_id: str):
        calls.append(invocation_id)
        return fake

    configure_streaming(factory)

    emit_to_thread(_handle(), _event("inv-A"), service_name="sre")
    emit_to_thread(_handle(), _event("inv-A"), service_name="sre")

    assert calls == ["inv-A"], "factory should be called once per invocation"
    # Both emits landed on the same stream.
    assert len(fake.emits) == 2


def test_factory_distinct_invocations_get_separate_streams():
    """Two invocation ids -> two factory calls -> two streams."""
    streams: dict[str, _FakeStream] = {}

    def factory(handle: ThreadHandle, invocation_id: str):
        s = _FakeStream(invocation_id)
        streams[invocation_id] = s
        return s

    configure_streaming(factory)

    emit_to_thread(_handle(), _event("inv-1"), service_name="music")
    emit_to_thread(_handle(), _event("inv-2"), service_name="music")

    assert set(streams) == {"inv-1", "inv-2"}
    assert len(streams["inv-1"].emits) == 1
    assert len(streams["inv-2"].emits) == 1


def test_factory_returning_none_is_not_cached_and_skips():
    """Factory may bail out with None; emit becomes a no-op."""
    calls: list[str] = []

    def factory(handle: ThreadHandle, invocation_id: str):
        calls.append(invocation_id)
        return None  # service can't / won't stream this handle

    configure_streaming(factory)

    emit_to_thread(_handle(), _event("inv-X"), service_name="movies")
    emit_to_thread(_handle(), _event("inv-X"), service_name="movies")

    # Factory invoked on each emit because we never cached the None result.
    assert calls == ["inv-X", "inv-X"]


def test_factory_exception_is_swallowed():
    """A buggy factory must not break the handler path."""

    def factory(handle: ThreadHandle, invocation_id: str):
        raise RuntimeError("boom")

    configure_streaming(factory)

    # Should not raise.
    emit_to_thread(_handle(), _event("inv-err"), service_name="sre")


def test_open_thread_stream_returns_none_without_factory():
    """Direct call surface returns None when streaming isn't wired."""
    result = open_thread_stream(_handle(), invocation_id="x", service_name="sre")
    assert result is None


def test_close_stream_evicts_and_finalises():
    """close_stream removes the cache entry + invokes ``.complete()``."""
    fake = _FakeStream("to-close")

    def factory(handle: ThreadHandle, invocation_id: str):
        return fake

    configure_streaming(factory)

    emit_to_thread(_handle(), _event("inv-close"), service_name="sre")
    assert fake.complete_called is False

    close_stream("inv-close")
    assert fake.complete_called is True

    # Subsequent close is a no-op (not in cache anymore).
    close_stream("inv-close")
    assert fake.complete_called is True  # still True, no exception


def test_close_stream_complete_failure_is_swallowed():
    """If ``stream.complete()`` raises, close_stream still succeeds."""

    class _BadStream(_FakeStream):
        def complete(self) -> None:  # pragma: no cover -- exercised via close
            raise RuntimeError("complete blew up")

    bad = _BadStream("bad")

    def factory(handle: ThreadHandle, invocation_id: str):
        return bad

    configure_streaming(factory)
    emit_to_thread(_handle(), _event("inv-bad"), service_name="sre")

    # No exception even though complete() raises.
    close_stream("inv-bad")


def test_emit_with_no_invocation_match_after_close():
    """After close_stream, a new emit with the same id rebuilds via factory."""
    streams: list[_FakeStream] = []

    def factory(handle: ThreadHandle, invocation_id: str):
        s = _FakeStream(invocation_id)
        streams.append(s)
        return s

    configure_streaming(factory)
    emit_to_thread(_handle(), _event("inv-rebuild"), service_name="sre")
    close_stream("inv-rebuild")
    emit_to_thread(_handle(), _event("inv-rebuild"), service_name="sre")

    assert len(streams) == 2, "post-close emit should rebuild the stream"
