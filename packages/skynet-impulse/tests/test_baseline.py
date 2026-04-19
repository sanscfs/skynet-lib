"""AdaptiveBaseline windowing + refractory cooldown."""

from __future__ import annotations

from _fake_redis import FakeRedis
from skynet_impulse.baseline import AdaptiveBaseline, BaselineConfig


def _make(min_history: int = 30) -> tuple[AdaptiveBaseline, FakeRedis]:
    cfg = BaselineConfig(
        prefix="test",
        window=100,
        percentile=75.0,
        cold_start_threshold=0.35,
        min_history=min_history,
        refractory_cap_ticks=4,
        mentions_cap=10,
    )
    return AdaptiveBaseline(cfg), FakeRedis()


def test_cold_start_returns_default():
    b, r = _make()
    for v in [0.1] * 5:
        b.append_history(r, v)
    assert b.p75(r) == 0.35


def test_p75_matches_sorted_rank():
    b, r = _make(min_history=5)
    vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for v in vals:
        b.append_history(r, v)
    # nearest-rank at p75 of 10 -> idx = int(10 * 0.75) = 7 -> values[7] = 0.8
    assert b.p75(r) == 0.8


def test_window_trims_old_values():
    b, r = _make(min_history=5)
    for v in range(150):
        b.append_history(r, v / 150.0)
    # window=100 so only the last 100 survive.
    assert b.history_len(r) == 100


def test_refractory_fresh_anchor_is_zero():
    b, r = _make()
    assert b.remaining_refractory(r, None) == 0
    assert b.remaining_refractory(r, "music:artist:foo") == 0


def test_refractory_bump_and_tick():
    b, r = _make()
    b.bump_refractory(r, "A")
    assert b.remaining_refractory(r, "A") == 1
    b.tick_refractories(r)
    # was 1 -> decrements to 0 -> evicted
    assert b.remaining_refractory(r, "A") == 0


def test_refractory_growth_caps_at_cap_ticks():
    b, r = _make()
    # Bump ten times; first mention=1 clamped to cap=4 floor(growth).
    for _ in range(10):
        b.bump_refractory(r, "A")
    # mentions=10, but cap_ticks=4 -> cooldown <= 4
    assert b.remaining_refractory(r, "A") == 4


def test_refractory_mentions_cap():
    b, r = _make()
    for _ in range(50):
        b.bump_refractory(r, "A")
    # mentions_cap=10 -> hset overwrites to 10
    stored = int(r.hget("test:mentions", "A") or 0)
    assert stored == 10


def test_list_active_refractories():
    b, r = _make()
    b.bump_refractory(r, "A")
    b.bump_refractory(r, "B")
    active = b.list_active_refractories(r)
    keys = {a for a, _ in active}
    assert keys == {"A", "B"}


def test_p75_with_many_samples():
    b, r = _make(min_history=10)
    for v in [0.5] * 20:
        b.append_history(r, v)
    assert b.p75(r) == 0.5
