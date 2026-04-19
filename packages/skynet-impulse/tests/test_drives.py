"""Drive + DriveState + SignalToDrive config validation."""

from __future__ import annotations

import pytest
from skynet_impulse.drives import Drive, DriveState, SignalToDrive


def test_drive_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty"):
        Drive(name="")


def test_drive_rejects_bad_decay():
    with pytest.raises(ValueError, match="decay_rate"):
        Drive(name="x", decay_rate=0.0)
    with pytest.raises(ValueError, match="decay_rate"):
        Drive(name="x", decay_rate=1.5)


def test_drive_rejects_negative_growth():
    with pytest.raises(ValueError, match="growth_per_tick"):
        Drive(name="x", growth_per_tick=-0.1)


def test_drive_rejects_initial_out_of_range():
    with pytest.raises(ValueError, match="initial"):
        Drive(name="x", initial=1.5)


def test_drivestate_clip():
    st = DriveState({"a": 1.5, "b": -0.3, "c": 0.5})
    st.clip()
    assert st.get("a") == 1.0
    assert st.get("b") == 0.0
    assert st.get("c") == 0.5


def test_drivestate_dominant_respects_exclude():
    st = DriveState({"boredom": 0.9, "curiosity": 0.5, "concern": 0.2})
    k, v = st.dominant()
    assert k == "boredom"
    k, v = st.dominant(exclude={"boredom"})
    assert k == "curiosity"
    assert v == 0.5


def test_drivestate_dominant_empty_with_all_excluded():
    st = DriveState({"a": 0.5})
    k, v = st.dominant(exclude={"a"})
    assert k is None
    assert v == 0.0


def test_drivestate_redis_roundtrip():
    drives = [Drive("a", 0.9, initial=0.2), Drive("b", 0.8)]
    mapping = {"a": "0.7500", "b": "0.1250"}
    st = DriveState.from_redis_mapping(mapping, drives)
    assert st.get("a") == pytest.approx(0.75)
    assert st.get("b") == pytest.approx(0.125)
    # Roundtrip back -- values should be serializable.
    back = st.to_redis_mapping()
    assert back["a"] == "0.7500"


def test_drivestate_missing_drives_fallback_to_initial():
    drives = [Drive("a", 0.9, initial=0.3), Drive("b", 0.8, initial=0.0)]
    st = DriveState.from_redis_mapping({}, drives)
    assert st.get("a") == 0.3
    assert st.get("b") == 0.0


def test_drivestate_garbage_falls_back_to_initial():
    drives = [Drive("a", 0.9, initial=0.1)]
    st = DriveState.from_redis_mapping({"a": "not a float"}, drives)
    assert st.get("a") == 0.1


def test_signal_to_drive_optional_dampen():
    wire = SignalToDrive("spoke", "need_to_share", dampen_multiply=0.6)
    assert wire.dampen_multiply == 0.6
    assert wire.multiplier == 1.0  # default preserved
