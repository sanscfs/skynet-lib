"""Signal primitives -- re-exports from ``skynet_core.impulses``.

There is exactly ONE source of truth for the ``Signal`` dataclass and signal
bus constants, and it lives in ``skynet-core``. This module exists so callers
can write::

    from skynet_impulse import Signal, SignalKind, STREAM_NAME

instead of reaching across packages. Do not define a parallel ``Signal`` here
-- any divergence would silently corrupt the Redis stream contract that every
Skynet producer + the agent consumer share.
"""

from __future__ import annotations

from skynet_core.impulses import (
    DEFAULT_CONSUMER_GROUP,
    DEFAULT_MAXLEN,
    STREAM_NAME,
    Signal,
    SignalKind,
    SignalSource,
    ack_signal,
    ack_signals,
    drain_signals,
    emit_signal,
    emit_signal_async,
    ensure_consumer_group,
)

__all__ = [
    "Signal",
    "SignalKind",
    "SignalSource",
    "STREAM_NAME",
    "DEFAULT_CONSUMER_GROUP",
    "DEFAULT_MAXLEN",
    "emit_signal",
    "emit_signal_async",
    "ensure_consumer_group",
    "drain_signals",
    "ack_signal",
    "ack_signals",
]
