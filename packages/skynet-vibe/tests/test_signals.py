"""Tests for VibeSignal / FacetVectors / Source."""

from __future__ import annotations

from datetime import datetime, timezone

from skynet_vibe import FacetVectors, Source, VibeSignal


def test_vibe_signal_roundtrip() -> None:
    vecs = FacetVectors(content=[0.1, 0.2, 0.3], context=[0.4, 0.5, 0.6], user_state=None)
    src = Source(type="chat", room_id="!room", agent_inferred=False)
    ts = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)
    sig = VibeSignal(
        id="abc",
        text_raw="this album is exactly my evening mood",
        vectors=vecs,
        source=src,
        timestamp=ts,
        confidence=0.9,
        linked_rec_id="rec_xyz",
        extra_payload={"tag": "feedback"},
    )
    as_dict = sig.to_dict()
    round_tripped = VibeSignal.from_dict(as_dict)
    assert round_tripped.id == sig.id
    assert round_tripped.text_raw == sig.text_raw
    assert round_tripped.vectors.content == sig.vectors.content
    assert round_tripped.vectors.context == sig.vectors.context
    assert round_tripped.vectors.user_state is None
    assert round_tripped.source.type == "chat"
    assert round_tripped.source.room_id == "!room"
    assert round_tripped.timestamp == ts
    assert round_tripped.confidence == 0.9
    assert round_tripped.linked_rec_id == "rec_xyz"
    assert round_tripped.extra_payload == {"tag": "feedback"}


def test_vibe_signal_has_no_domain_field() -> None:
    """Design axiom: domain membership is run-time geometry, not a stored label."""
    ts = datetime.now(timezone.utc)
    sig = VibeSignal(
        id="id1",
        text_raw="text",
        vectors=FacetVectors(content=[0.0]),
        source=Source(type="chat"),
        timestamp=ts,
    )
    assert not hasattr(sig, "domain")
    assert "domain" not in sig.to_dict()


def test_new_id_unique() -> None:
    ids = {VibeSignal.new_id() for _ in range(10)}
    assert len(ids) == 10


def test_timezone_naive_input_becomes_utc() -> None:
    naive_ts = datetime(2026, 1, 1, 0, 0, 0)
    restored = VibeSignal.from_dict(
        {
            "id": "x",
            "text_raw": "t",
            "vectors": {"content": [0.0]},
            "source": {"type": "chat"},
            "timestamp": naive_ts.isoformat(),
        }
    )
    assert restored.timestamp.tzinfo is not None
