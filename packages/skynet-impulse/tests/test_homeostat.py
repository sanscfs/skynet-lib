"""Homeostat accumulation + decay + config validation."""

from __future__ import annotations

import pytest
from _fake_redis import FakeRedis
from skynet_impulse.drives import Drive, SignalToDrive
from skynet_impulse.homeostat import Homeostat
from skynet_impulse.signals import Signal


def _make_homeostat() -> Homeostat:
    drives = [
        Drive("boredom", decay_rate=0.9, growth_per_tick=0.08),
        Drive("curiosity", decay_rate=0.85),
        Drive("concern", decay_rate=0.80),
        Drive("need_to_share", decay_rate=0.90),
    ]
    wirings = [
        SignalToDrive("novelty", "curiosity", multiplier=0.30),
        SignalToDrive("concern", "concern", multiplier=0.40),
        SignalToDrive("memory_activation", "need_to_share", multiplier=0.35),
        SignalToDrive("spoke", "need_to_share", dampen_multiply=0.6),
    ]
    return Homeostat(drives=drives, signal_to_drive=wirings, state_key="test:state")


def test_rejects_unknown_drive_wiring():
    drives = [Drive("a")]
    wirings = [SignalToDrive("x", "nonexistent")]
    with pytest.raises(ValueError, match="unknown drive"):
        Homeostat(drives=drives, signal_to_drive=wirings, state_key="k")


def test_rejects_empty_drives():
    with pytest.raises(ValueError, match="at least one"):
        Homeostat(drives=[], signal_to_drive=[], state_key="k")


def test_apply_signals_accumulates():
    h = _make_homeostat()
    state = h.load_state(FakeRedis())
    sig_nov = Signal(kind="novelty", source="analyzer", salience=0.8)
    anchor, pushed = h.apply_signals(state, [sig_nov])
    # curiosity += 0.8 * 0.30 = 0.24
    assert state.get("curiosity") == pytest.approx(0.24)
    assert "curiosity" in pushed
    assert anchor is None


def test_apply_signals_returns_top_anchor():
    h = _make_homeostat()
    state = h.load_state(FakeRedis())
    sigs = [
        Signal(kind="novelty", source="analyzer", salience=0.2, anchor="low"),
        Signal(kind="novelty", source="analyzer", salience=0.9, anchor="high"),
        Signal(kind="novelty", source="analyzer", salience=0.5, anchor="mid"),
    ]
    anchor, _ = h.apply_signals(state, sigs)
    assert anchor == "high"


def test_apply_signals_dampen_multiply():
    h = _make_homeostat()
    state = h.load_state(FakeRedis())
    state.set("need_to_share", 0.5)
    spoke = Signal(kind="spoke", source="self", salience=0.3)
    h.apply_signals(state, [spoke])
    assert state.get("need_to_share") == pytest.approx(0.5 * 0.6)


def test_apply_decay_skips_pushed_drives():
    h = _make_homeostat()
    state = h.load_state(FakeRedis())
    state.values = {"boredom": 0.4, "curiosity": 0.5, "concern": 0.3, "need_to_share": 0.2}
    # pretend curiosity was pushed this tick
    h.apply_decay(state, pushed_drives={"curiosity"})
    assert state.get("curiosity") == pytest.approx(0.5)  # untouched
    assert state.get("concern") == pytest.approx(0.3 * 0.80)
    assert state.get("need_to_share") == pytest.approx(0.2 * 0.90)
    # boredom: 0.4 * 0.9 + 0.08 = 0.44
    assert state.get("boredom") == pytest.approx(0.4 * 0.9 + 0.08)


def test_apply_decay_always_adds_growth():
    h = _make_homeostat()
    state = h.load_state(FakeRedis())
    state.values = {"boredom": 0.0, "curiosity": 0.0, "concern": 0.0, "need_to_share": 0.0}
    h.apply_decay(state, pushed_drives={"boredom"})
    # boredom pushed -> skip decay, but growth_per_tick still applies.
    assert state.get("boredom") == pytest.approx(0.08)


def test_persist_roundtrip():
    h = _make_homeostat()
    r = FakeRedis()
    state = h.load_state(r)
    state.values = {"boredom": 0.1, "curiosity": 0.5, "concern": 0.0, "need_to_share": 0.25}
    h.save_state(r, state)
    state2 = h.load_state(r)
    assert state2.get("boredom") == pytest.approx(0.1)
    assert state2.get("curiosity") == pytest.approx(0.5)
    assert state2.get("need_to_share") == pytest.approx(0.25)


def test_clip_enforced_on_save():
    h = _make_homeostat()
    r = FakeRedis()
    state = h.load_state(r)
    state.values = {"boredom": 5.0, "curiosity": -2.0, "concern": 0.0, "need_to_share": 0.0}
    h.save_state(r, state)
    state2 = h.load_state(r)
    assert state2.get("boredom") == 1.0
    assert state2.get("curiosity") == 0.0
