"""Two-speed EMA calibration + continuous anchor penalty tests.

Covers the 12 cases spec'd for the adaptive half-life work:

1.  Cold start — prior, no observations.
2.  Convergence on a constant gap stream.
3.  Sudden regime shift — fast tracks, slow lags, effective blends.
4.  Outlier resistance — single spike recovers cleanly.
5.  Penalty math — half-life decay accuracy.
6.  Gate enforcement — fire sets penalty, cooldown unlocks.
7.  Safety clamps — HALF_LIFE_MIN / HALF_LIFE_MAX.
8.  Persistence roundtrip.
9.  Cross-domain isolation.
10. Deprecation warning on refractory_cap_ticks.
11. Blend edge case — equal EMAs.
12. Blend disagreement at 1.0 — fast dominates.

The calibration math is fully unit-testable without Redis, so most
tests go directly against :class:`HalfLifeCalibrator`.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import pytest

# Same shim as conftest.py — register tests dir on sys.path so _fake_redis imports cleanly.
sys.path.insert(0, str(Path(__file__).parent))

from _fake_redis import FakeRedis  # noqa: E402
from skynet_impulse.calibration import (  # noqa: E402
    HALF_LIFE_MAX,
    HALF_LIFE_MIN,
    PENALTY_NOISE_FLOOR,
    CalibrationPersistence,
    CalibrationState,
    HalfLifeCalibrator,
)
from skynet_impulse.drives import Drive, SignalToDrive  # noqa: E402
from skynet_impulse.engine import EngineConfig  # noqa: E402

# ---- 1. Cold start -------------------------------------------------------


def test_cold_start_returns_prior_and_zero_observations():
    calib = HalfLifeCalibrator(prior=33.0)
    assert calib.h_fast == 33.0
    assert calib.h_slow == 33.0
    assert calib.observations == 0
    assert calib.global_signal_count == 0
    # Effective half-life = prior when fast == slow == prior.
    assert calib.effective_half_life() == 33.0
    diag = calib.diagnostics()
    assert diag["observations"] == 0
    assert diag["h_fast"] == 33.0


# ---- 2. Convergence on constant gap stream ------------------------------


def _feed_gap_stream(calib: HalfLifeCalibrator, gap: int, n: int) -> None:
    """Feed ``n`` observations of the same gap by emitting a synthetic
    alternating-anchor stream spaced ``gap`` events apart.

    The cleanest way to generate a precise gap=G observation is:
    feed (G-1) no-anchor signals, then one sighting of a fixed anchor.
    First sighting records last_seen only; each subsequent sighting
    after (G-1) empties yields a gap of exactly G.
    """
    anchor = "constant:a"
    # Seed the anchor so the NEXT sighting records a gap.
    calib.observe_signal(anchor)
    for _ in range(n):
        for _ in range(gap - 1):
            calib.observe_signal(None)
        calib.observe_signal(anchor)


def test_convergence_on_constant_gap():
    calib = HalfLifeCalibrator(prior=33.0)
    _feed_gap_stream(calib, gap=75, n=50)
    # After ~50 observations the fast EMA (window ~10) should be right at
    # the target; the slow EMA (window ~50) lags but is converging.
    assert calib.h_fast == pytest.approx(75.0, rel=0.05)
    # Slow at 50 obs should be somewhere between prior and target.
    assert 50.0 < calib.h_slow < 75.0

    _feed_gap_stream(calib, gap=75, n=50)  # total obs: 100
    # Fast still at 75; slow closer.
    assert calib.h_fast == pytest.approx(75.0, rel=0.01)
    assert calib.h_slow == pytest.approx(75.0, rel=0.15)

    _feed_gap_stream(calib, gap=75, n=100)  # total obs: 200
    # Both tightly converged.
    assert calib.h_fast == pytest.approx(75.0, rel=0.01)
    assert calib.h_slow == pytest.approx(75.0, rel=0.05)


# ---- 3. Sudden regime shift ---------------------------------------------


def test_sudden_shift_fast_tracks_slow_lags():
    calib = HalfLifeCalibrator(prior=33.0)
    # Establish a stable regime at gap=75.
    _feed_gap_stream(calib, gap=75, n=150)
    assert calib.h_fast == pytest.approx(75.0, rel=0.05)
    assert calib.h_slow == pytest.approx(75.0, rel=0.1)

    # Snapshot then shock to gap=15. Checkpoint after ~5 shock obs —
    # fast EMA is well into the new regime, slow has barely started moving.
    _feed_gap_stream(calib, gap=15, n=5)
    pre_fast, pre_slow = calib.h_fast, calib.h_slow
    # Fast already below 50 after 5 obs (alpha=0.3 has effective window ~10);
    # slow still clearly above fast (alpha=0.05 has effective window ~50).
    assert pre_fast < 50.0
    assert pre_slow > pre_fast + 10.0
    # Disagreement is significant.
    assert calib.disagreement() > 0.3
    # Effective blends between fast and slow.
    effective = calib.effective_half_life()
    assert pre_fast <= effective <= pre_slow

    # After a long run at gap=15 both converge down.
    _feed_gap_stream(calib, gap=15, n=200)
    assert calib.h_fast == pytest.approx(15.0, rel=0.1)
    assert calib.h_slow == pytest.approx(15.0, abs=5.0)


# ---- 4. Outlier resistance ----------------------------------------------


def test_single_outlier_transient_recovers():
    calib = HalfLifeCalibrator(prior=33.0)
    _feed_gap_stream(calib, gap=75, n=80)
    pre_fast, pre_slow = calib.h_fast, calib.h_slow
    assert pre_fast == pytest.approx(75.0, rel=0.05)

    # Single outlier observation = 500.
    calib.observe_signal("outlier:a")  # seed last_seen
    for _ in range(499):
        calib.observe_signal(None)
    calib.observe_signal("outlier:a")  # gap = 500
    # Fast jumps a lot; slow barely.
    assert calib.h_fast > pre_fast + 50
    assert calib.h_slow < pre_slow + 40

    # Recover by feeding 20 more normal observations.
    _feed_gap_stream(calib, gap=75, n=20)
    # Fast should be drifting back towards 75; slow still elevated but
    # less than fast was at peak.
    assert calib.h_fast < 150.0
    # Effective is always clamped to HALF_LIFE_MAX regardless.
    assert calib.effective_half_life() <= HALF_LIFE_MAX


# ---- 5. Penalty math ----------------------------------------------------


def test_penalty_halves_at_one_half_life():
    # Force a stable half-life by priming the EMAs manually.
    calib = HalfLifeCalibrator(prior=33.0)
    calib.state.h_fast = 33.0
    calib.state.h_slow = 33.0
    calib.assign_full_penalty("X")
    assert calib.penalty_for("X") == 1.0

    # Decay 33 times — one half-life worth of events. Penalty should be ~0.5.
    for _ in range(33):
        calib.decay_penalties()
    assert calib.penalty_for("X") == pytest.approx(0.5, abs=0.02)

    # Decay until below noise floor. ln(0.1) / ln(0.5) ≈ 3.32 half-lives ≈ 110 events.
    calib2 = HalfLifeCalibrator(prior=33.0)
    calib2.state.h_fast = 33.0
    calib2.state.h_slow = 33.0
    calib2.assign_full_penalty("Y")
    for _ in range(110):
        calib2.decay_penalties()
    assert calib2.penalty_for("Y") == pytest.approx(PENALTY_NOISE_FLOOR, abs=0.02)


# ---- 6. Gate enforcement ------------------------------------------------


def test_gate_blocks_then_unblocks_after_decay():
    calib = HalfLifeCalibrator(prior=33.0)
    calib.state.h_fast = 33.0
    calib.state.h_slow = 33.0
    calib.assign_full_penalty("X")
    assert calib.is_under_penalty("X") is True

    # After 110 decay steps (penalty ≈ noise floor) anchor should be allowed.
    for _ in range(110):
        calib.decay_penalties()
    # Right at the floor: the gate's >= comparison will barely still block,
    # but one more decay step definitely unlocks.
    calib.decay_penalties()
    assert calib.is_under_penalty("X") is False


# ---- 7. Safety clamps ---------------------------------------------------


def test_effective_half_life_clamps_to_min():
    # Pathological: gap=1 forever pushes both EMAs below HALF_LIFE_MIN.
    calib = HalfLifeCalibrator(prior=33.0)
    # A rapid-fire stream: alternating sightings of one anchor at gap=1
    # means the signal counter advances and we see a gap of 1 each time.
    anchor = "hot:a"
    calib.observe_signal(anchor)
    for _ in range(200):
        calib.observe_signal(anchor)
    # Raw blend would be ~1; clamp keeps us at HALF_LIFE_MIN.
    assert calib.effective_half_life() == HALF_LIFE_MIN


def test_effective_half_life_clamps_to_max():
    # Corrupt observation: inject 10000 directly.
    calib = HalfLifeCalibrator(prior=33.0)
    calib._apply_observation(10_000)
    # Raw h_fast = 0.3*10000 + 0.7*33 ≈ 3023, clamped to HALF_LIFE_MAX.
    assert calib.effective_half_life() == HALF_LIFE_MAX


# ---- 8. Persistence roundtrip -------------------------------------------


def test_persistence_roundtrip_preserves_state():
    calib = HalfLifeCalibrator(prior=33.0)
    _feed_gap_stream(calib, gap=60, n=30)
    calib.assign_full_penalty("roundtrip:a")
    calib.decay_penalties()

    redis = FakeRedis()
    # Use a typical key_prefix format from EngineConfig.
    persist = CalibrationPersistence(prefix="skynet:impulses:movies")
    persist.save(redis, calib.state)

    restored = persist.load(redis, prior=33.0)
    assert restored.h_fast == pytest.approx(calib.h_fast, abs=1e-4)
    assert restored.h_slow == pytest.approx(calib.h_slow, abs=1e-4)
    assert restored.global_signal_count == calib.global_signal_count
    assert restored.observations == calib.observations
    assert set(restored.anchor_last_seen.keys()) == set(calib.state.anchor_last_seen.keys())
    assert math.isclose(
        restored.penalties.get("roundtrip:a", 0.0),
        calib.penalty_for("roundtrip:a"),
        abs_tol=1e-4,
    )


# ---- 9. Cross-domain isolation ------------------------------------------


def test_cross_domain_isolation():
    redis = FakeRedis()
    movies_persist = CalibrationPersistence(prefix="skynet:impulses:movies")
    music_persist = CalibrationPersistence(prefix="skynet:impulses:music")

    movies_calib = HalfLifeCalibrator(prior=33.0)
    _feed_gap_stream(movies_calib, gap=20, n=40)
    movies_persist.save(redis, movies_calib.state)

    music_calib = HalfLifeCalibrator(prior=33.0)
    _feed_gap_stream(music_calib, gap=100, n=40)
    music_persist.save(redis, music_calib.state)

    # Load each — values must diverge.
    movies_restored = movies_persist.load(redis, prior=33.0)
    music_restored = music_persist.load(redis, prior=33.0)
    assert movies_restored.h_fast < 40.0
    assert music_restored.h_fast > 60.0
    # Underlying Redis keys must be distinct.
    assert movies_persist.calibration_key != music_persist.calibration_key


# ---- 10. Deprecation warning --------------------------------------------


def test_refractory_cap_ticks_emits_deprecation_warning():
    cfg = EngineConfig(
        domain="deprecation",
        drives=[Drive("curiosity", 0.85)],
        signal_to_drive=[SignalToDrive("novelty", "curiosity", 0.3)],
        refractory_cap_ticks=12,  # non-default: should warn
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg.validate()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert "refractory_cap_ticks" in str(deprecations[0].message)


def test_refractory_cap_ticks_default_is_silent():
    # Default value must not warn — adaptive still uses the legacy baseline
    # refractory as a mid-upgrade fallback.
    cfg = EngineConfig(
        domain="deprecation-silent",
        drives=[Drive("curiosity", 0.85)],
        signal_to_drive=[SignalToDrive("novelty", "curiosity", 0.3)],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg.validate()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 0


# ---- 11. Blend edge case: h_fast == h_slow ------------------------------


def test_blend_equal_emas_returns_slow():
    calib = HalfLifeCalibrator(prior=33.0)
    calib.state.h_fast = 50.0
    calib.state.h_slow = 50.0
    # Disagreement = 0, weight_fast = 0, blend = slow.
    assert calib.effective_half_life() == 50.0


# ---- 12. Blend disagreement at/above 1.0 --------------------------------


def test_blend_disagreement_at_one_dominates_fast():
    calib = HalfLifeCalibrator(prior=33.0)
    calib.state.h_fast = 10.0
    calib.state.h_slow = 100.0
    # Disagreement = 0.9, weight_fast = 0.9, blend = 0.9*10 + 0.1*100 = 19.
    assert calib.effective_half_life() == pytest.approx(19.0, abs=0.01)


def test_blend_disagreement_clamped_at_one():
    calib = HalfLifeCalibrator(prior=33.0)
    calib.state.h_fast = 5.0
    calib.state.h_slow = 100.0
    # Disagreement = 0.95 — still under 1.
    assert calib.disagreement() < 1.0
    # Push further so disagreement > 1.
    calib.state.h_fast = 300.0
    calib.state.h_slow = 50.0
    assert calib.disagreement() == pytest.approx(5.0, abs=0.01)
    # weight_fast = min(disagreement, 1) = 1, so blend = h_fast = 300.
    # But then clamped to HALF_LIFE_MAX=500 — so 300 passes through.
    assert calib.effective_half_life() == 300.0


# ---- Bonus: observe_signal returns gap ----------------------------------


def test_observe_signal_returns_gap_on_second_sighting():
    calib = HalfLifeCalibrator(prior=33.0)
    assert calib.observe_signal("x") is None  # first sighting
    assert calib.observe_signal(None) is None  # non-anchored
    assert calib.observe_signal("x") == 2  # count was 1 at first, now 3 -> gap 2


def test_observe_signal_no_anchor_increments_counter():
    calib = HalfLifeCalibrator(prior=33.0)
    for _ in range(5):
        calib.observe_signal(None)
    assert calib.global_signal_count == 5
    assert calib.observations == 0


# ---- Persistence failure is non-fatal -----------------------------------


class BrokenRedis:
    def hgetall(self, *_a, **_kw):
        raise RuntimeError("redis down")

    def hset(self, *_a, **_kw):
        raise RuntimeError("redis down")

    def hdel(self, *_a, **_kw):
        raise RuntimeError("redis down")


def test_persistence_failure_does_not_raise():
    persist = CalibrationPersistence(prefix="skynet:impulses:broken")
    state = CalibrationState(h_fast=42.0, h_slow=40.0)
    state.penalties["x"] = 0.5
    # Must not raise.
    persist.save(BrokenRedis(), state)
    # Load under failure returns a default state (prior).
    restored = persist.load(BrokenRedis(), prior=33.0)
    assert restored.h_fast == 33.0
