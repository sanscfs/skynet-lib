"""Tests for the impulses signal bus primitives.

Pure-Python unit tests: Signal serialization + salience validation.
The Redis-XADD / XREADGROUP plumbing is covered by the agent-side
integration tests (which run against a real Redis), so here we only
pin down the contract shape.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from skynet_core.impulses import (
    STREAM_NAME,
    Signal,
    emit_signal,
)

# --- Signal round-trip -------------------------------------------------


def test_signal_to_fields_round_trip_preserves_all_fields():
    sig = Signal(
        kind="novelty",
        source="analyzer",
        salience=0.73,
        anchor="trait:focus",
        payload={"z_score": 2.4, "feature": "activity.gadgetbridge"},
    )
    fields = sig.to_fields()
    restored = Signal.from_fields(fields)
    assert restored.kind == sig.kind
    assert restored.source == sig.source
    assert restored.salience == pytest.approx(sig.salience, abs=1e-4)
    assert restored.anchor == sig.anchor
    assert restored.payload == sig.payload
    assert restored.ts_logical == sig.ts_logical
    assert restored.id == sig.id


def test_signal_handles_none_anchor_round_trip():
    sig = Signal(kind="spoke", source="self", salience=0.2, anchor=None)
    restored = Signal.from_fields(sig.to_fields())
    assert restored.anchor is None


def test_signal_payload_supports_cyrillic():
    sig = Signal(
        kind="memory_activation",
        source="identity",
        salience=0.6,
        anchor="тема:здоров'я",
        payload={"нотатка": "згадалось про режим сну"},
    )
    restored = Signal.from_fields(sig.to_fields())
    # Cyrillic must not get mojibake'd through the JSON round-trip.
    assert restored.anchor == "тема:здоров'я"
    assert restored.payload["нотатка"] == "згадалось про режим сну"


def test_signal_autoassigns_id_and_timestamp():
    a = Signal(kind="spoke", source="self", salience=0.1)
    b = Signal(kind="spoke", source="self", salience=0.1)
    assert a.id != b.id
    assert len(a.id) == 12
    assert a.ts_logical > 0


# --- salience validation ----------------------------------------------


def test_emit_signal_rejects_out_of_range_salience():
    with pytest.raises(ValueError):
        emit_signal("novelty", "analyzer", 1.2, redis_client=MagicMock())
    with pytest.raises(ValueError):
        emit_signal("novelty", "analyzer", -0.1, redis_client=MagicMock())


def test_emit_signal_accepts_boundary_values():
    client = MagicMock()
    client.xadd.return_value = "1-0"
    assert emit_signal("spoke", "self", 0.0, redis_client=client) == "1-0"
    assert emit_signal("concern", "sre", 1.0, redis_client=client) == "1-0"


# --- emit_signal wiring ------------------------------------------------


def test_emit_signal_writes_to_configured_stream_with_maxlen():
    client = MagicMock()
    client.xadd.return_value = "17-0"
    entry_id = emit_signal(
        "trait_drift",
        "profile-synthesis",
        salience=0.55,
        anchor="trait:sleep",
        payload={"delta": 0.08},
        redis_client=client,
        maxlen=1234,
    )
    assert entry_id == "17-0"
    assert client.xadd.call_count == 1
    args, kwargs = client.xadd.call_args
    assert args[0] == STREAM_NAME
    # Fields are the second positional arg — a dict.
    fields = args[1]
    assert fields["kind"] == "trait_drift"
    assert fields["source"] == "profile-synthesis"
    assert fields["anchor"] == "trait:sleep"
    assert fields["salience"] == "0.5500"
    assert kwargs["maxlen"] == 1234
    assert kwargs["approximate"] is True


def test_emit_signal_empty_anchor_stored_as_empty_string():
    # Redis streams reject None field values; we normalize to "".
    client = MagicMock()
    client.xadd.return_value = "1-0"
    emit_signal("spoke", "self", 0.3, redis_client=client)
    fields = client.xadd.call_args[0][1]
    assert fields["anchor"] == ""


def test_signal_from_fields_treats_empty_anchor_as_none():
    # Symmetric with the emit_signal behavior: round-trip through Redis
    # should yield anchor=None when no anchor was set.
    restored = Signal.from_fields(
        {
            "kind": "spoke",
            "source": "self",
            "salience": "0.3",
            "anchor": "",
            "payload": "{}",
            "ts_logical": "1234567",
            "id": "abc123",
        }
    )
    assert restored.anchor is None
