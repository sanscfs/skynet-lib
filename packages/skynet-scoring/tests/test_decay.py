"""Tests for logical-time decay and memory classification."""

from __future__ import annotations

import math

import pytest
from skynet_scoring import (
    DEFAULT_LAMBDAS,
    LAMBDA_IDENTITY,
    MEMORY_CLASSES,
    NEUTRAL_SALIENCE,
    classify_memory,
    compute_decay_factor,
    compute_decay_factor_calendar,
    compute_decay_factor_logical,
    default_salience_for,
)

# --- classify_memory ---------------------------------------------------


def test_classify_explicit_memory_class_wins():
    # Explicit field beats every heuristic.
    assert classify_memory({"memory_class": "identity", "source": "chat"}) == "identity"
    assert classify_memory({"memory_class": "trait"}) == "trait"


def test_classify_unknown_explicit_falls_through():
    # Unknown value falls through to heuristics, then default.
    assert classify_memory({"memory_class": "alien"}) == "raw"


def test_classify_memory_tier_legacy_synonym():
    assert classify_memory({"memory_tier": "episodic"}) == "episodic"
    assert classify_memory({"memory_tier": "trait-semantic"}) == "trait"


def test_classify_source_heuristics():
    assert classify_memory({"source": "consolidation"}) == "trait"
    assert classify_memory({"source": "wiki"}) == "semantic"
    assert classify_memory({"source": "skynet_episodic"}) == "episodic"
    assert classify_memory({"source": "session"}) == "working"
    assert classify_memory({"source": "google_takeout_gemini"}) == "raw"
    assert classify_memory({"source": ""}) == "raw"


def test_classify_category_heuristics():
    assert classify_memory({"category": "identity"}) == "identity"
    assert classify_memory({"category": "knowledge"}) == "semantic"
    assert classify_memory({"category": "episodic_daily"}) == "episodic"


def test_classify_tags():
    assert classify_memory({"tags": ["trait", "stable"]}) == "trait"
    assert classify_memory({"tags": ["episodic"]}) == "episodic"


def test_classify_default_raw():
    assert classify_memory({}) == "raw"
    assert classify_memory({"source": "unknown-origin"}) == "raw"


def test_classify_non_dict_is_raw():
    # Defensive: corrupt payloads get the most-decaying class.
    assert classify_memory("garbage") == "raw"  # type: ignore[arg-type]
    assert classify_memory(None) == "raw"  # type: ignore[arg-type]


def test_all_memory_classes_have_lambdas():
    # Every class must have an entry in DEFAULT_LAMBDAS or fall back
    # cleanly to LAMBDA_RAW.
    for cls in MEMORY_CLASSES:
        # _resolve_lambda would fall back to raw, so this should never error.
        assert cls in DEFAULT_LAMBDAS or DEFAULT_LAMBDAS.get("raw") is not None


# --- compute_decay_factor_logical --------------------------------------


def test_logical_zero_missed_opportunities_is_full():
    # Silence case: nothing happened, no decay.
    payload = {"memory_class": "raw", "missed_opportunities": 0}
    assert compute_decay_factor_logical(payload) == pytest.approx(1.0)


def test_logical_silence_over_time_keeps_score_at_one():
    # The whole point of logical time: the datetime doesn't matter.
    payload = {
        "memory_class": "raw",
        "missed_opportunities": 0,
        "last_accessed": "2020-01-01T00:00:00Z",
    }
    assert compute_decay_factor_logical(payload) == pytest.approx(1.0)


def test_logical_identity_never_decays():
    # LAMBDA_IDENTITY defaults to 0 -> constant 1.0 regardless of misses.
    if LAMBDA_IDENTITY != 0:
        pytest.skip("LAMBDA_IDENTITY overridden by env, can't assert immortality")
    payload = {"memory_class": "identity", "missed_opportunities": 10_000}
    assert compute_decay_factor_logical(payload) == 1.0


def test_logical_decay_monotone_with_misses():
    # More misses -> strictly lower decay factor for a non-zero lambda.
    payload_lo = {"memory_class": "raw", "missed_opportunities": 10}
    payload_hi = {"memory_class": "raw", "missed_opportunities": 100}
    assert compute_decay_factor_logical(payload_lo) > compute_decay_factor_logical(payload_hi)


def test_logical_salience_slows_decay():
    # High salience should yield a larger (less decayed) factor than
    # default salience at the same miss count.
    base = {"memory_class": "raw", "missed_opportunities": 50}
    high_salience = {**base, "salience": 0.95}
    default_salience = {**base, "salience": 0.5}
    assert compute_decay_factor_logical(high_salience) > compute_decay_factor_logical(default_salience)


def test_logical_compression_level_slows_decay():
    """Phase 9: higher compression_level dampens decay so summaries
    survive retrieval pressure longer than the raws they replaced."""
    base = {"memory_class": "raw", "missed_opportunities": 100, "salience": 0.5}
    level_0 = {**base, "compression_level": 0}
    level_2 = {**base, "compression_level": 2}
    level_5 = {**base, "compression_level": 5}
    f0 = compute_decay_factor_logical(level_0)
    f2 = compute_decay_factor_logical(level_2)
    f5 = compute_decay_factor_logical(level_5)
    # Strict monotonic: more compression = less decay.
    assert f5 > f2 > f0


def test_logical_compression_dampening_stacks_with_salience():
    """Both modulators apply multiplicatively, so a salient level-2
    summary decays slower than a non-salient level-2 (which already
    decays slower than a level-0)."""
    base = {"memory_class": "raw", "missed_opportunities": 100}
    plain_l0 = {**base, "salience": 0.5, "compression_level": 0}
    salient_l0 = {**base, "salience": 0.95, "compression_level": 0}
    plain_l2 = {**base, "salience": 0.5, "compression_level": 2}
    salient_l2 = {**base, "salience": 0.95, "compression_level": 2}
    f_plain_l0 = compute_decay_factor_logical(plain_l0)
    f_salient_l0 = compute_decay_factor_logical(salient_l0)
    f_plain_l2 = compute_decay_factor_logical(plain_l2)
    f_salient_l2 = compute_decay_factor_logical(salient_l2)
    assert f_salient_l2 > f_plain_l2 > f_plain_l0
    assert f_salient_l2 > f_salient_l0


def test_logical_compression_handles_garbage():
    """Corrupt compression_level (None / negative / string) collapses
    to level 0 — never raises, never inverts the dampen direction."""
    base = {"memory_class": "raw", "missed_opportunities": 50, "salience": 0.5}
    f_baseline = compute_decay_factor_logical({**base, "compression_level": 0})
    for bad in [None, -3, "garbage", 0.5]:
        f = compute_decay_factor_logical({**base, "compression_level": bad})
        # All garbage clamps to 0 → same factor as explicit 0.
        if bad in (0.5,):
            # 0.5 → int(0.5) = 0, same as level 0.
            assert f == f_baseline
        else:
            assert f == f_baseline


def test_logical_explicit_lambda_override():
    # Caller-supplied lambdas take precedence over env defaults.
    payload = {"memory_class": "raw", "missed_opportunities": 100, "salience": 0.0}
    # With lambda = 0 the override makes the point immortal.
    immortal = compute_decay_factor_logical(payload, lambdas={"raw": 0.0})
    assert immortal == 1.0
    # With a huge lambda the point is crushed.
    crushed = compute_decay_factor_logical(payload, lambdas={"raw": 1.0})
    assert crushed == pytest.approx(math.exp(-100), abs=1e-9)


def test_logical_non_dict_payload_is_full():
    # Corrupt payload defensively returns 1.0 instead of raising.
    assert compute_decay_factor_logical(None) == 1.0  # type: ignore[arg-type]


# --- compute_decay_factor dispatcher -----------------------------------


def test_dispatcher_default_is_logical():
    payload = {"memory_class": "raw", "missed_opportunities": 0}
    assert compute_decay_factor(payload) == pytest.approx(1.0)


def test_dispatcher_calendar_path():
    # Calendar path still works for A/B tests / legacy callers.
    payload = {"last_accessed": "2020-01-01T00:00:00Z"}
    factor = compute_decay_factor(payload, time_basis="calendar")
    # Something decayed is better than nothing decayed — just sanity check.
    assert 0.0 <= factor <= 1.0


def test_dispatcher_rejects_unknown_basis():
    with pytest.raises(ValueError, match="time_basis"):
        compute_decay_factor({}, time_basis="wall-clock")


# --- compute_decay_factor_calendar -------------------------------------


def test_calendar_missing_timestamp_returns_neutral():
    # No timestamp -> neutral 0.5 (can't reason about age).
    assert compute_decay_factor_calendar({}) == 0.5


def test_calendar_custom_tau():
    # With tau_days=1, a 1-day-old entry decays by ~1/e.
    payload = {"last_accessed": "2020-01-01T00:00:00Z"}
    factor = compute_decay_factor_calendar(payload, tau_days=1.0)
    # Can't assert exact value without fixing `now`, but should be
    # strictly below 0.01 for a point that's years old.
    assert factor < 0.01


# --- default_salience_for ----------------------------------------------


def test_salience_explicit_wins():
    # An explicit numeric salience always takes precedence.
    assert default_salience_for({"salience": 0.77, "source": "phone_app_activity"}) == pytest.approx(0.77)
    # Clamped into [0, 1].
    assert default_salience_for({"salience": -5.0}) == 0.0
    assert default_salience_for({"salience": 17.0}) == 1.0


def test_salience_source_high_trust():
    # Source-based bases for authored / ground-truth signals.
    assert default_salience_for({"source": "wiki:entities/user.md"}) == pytest.approx(0.95)
    assert default_salience_for({"source": "feedback"}) == pytest.approx(0.85)
    assert default_salience_for({"source": "identity"}) == pytest.approx(0.90)


def test_salience_source_low_noise():
    # Telemetry sources get the low band and decay fast.
    assert default_salience_for({"source": "phone_app_activity"}) == pytest.approx(0.15)
    assert default_salience_for({"source": "k8s"}) == pytest.approx(0.25)


def test_salience_prefix_fallback():
    # Unknown phone_* source falls through to the phone_ prefix rule.
    assert default_salience_for({"source": "phone_telemetry_sleep"}) == pytest.approx(0.15)
    # Unknown wiki path -> wiki: prefix rule.
    assert default_salience_for({"source": "wiki:concepts/tools.md"}) == pytest.approx(0.80)


def test_salience_confirmed_boost():
    # Confirmed_count >= 3 adds +0.1.
    payload = {"source": "chat", "confirmed_count": 5}
    assert default_salience_for(payload) == pytest.approx(0.70)  # 0.60 + 0.10


def test_salience_contradicted_and_strike_penalties():
    # Both penalties stack.
    payload = {"source": "chat", "contradicted_count": 1, "decay_strikes": 2}
    # 0.60 - 0.10 - 0.15 = 0.35
    assert default_salience_for(payload) == pytest.approx(0.35)


def test_salience_clamps_after_modifier():
    # Stacked penalties can't drag salience below 0.
    payload = {
        "source": "phone_app_activity",
        "contradicted_count": 1,
        "decay_strikes": 5,
    }
    assert default_salience_for(payload) == 0.0


def test_salience_unknown_source_is_neutral():
    assert default_salience_for({"source": "alien-origin"}) == NEUTRAL_SALIENCE
    assert default_salience_for({}) == NEUTRAL_SALIENCE


def test_salience_non_dict_is_neutral():
    # Corrupt input -> neutral fallback, no raise.
    assert default_salience_for(None) == NEUTRAL_SALIENCE  # type: ignore[arg-type]
    assert default_salience_for("garbage") == NEUTRAL_SALIENCE  # type: ignore[arg-type]


def test_logical_decay_picks_up_heuristic_salience():
    # A phone_app_activity point has low base salience -> lambda_eff
    # should be larger -> decay factor smaller than a high-salience
    # wiki point at the same miss count.
    shared = {"memory_class": "raw", "missed_opportunities": 50}
    noise = compute_decay_factor_logical({**shared, "source": "phone_app_activity"})
    wiki = compute_decay_factor_logical({**shared, "source": "wiki:entities/user.md"})
    assert wiki > noise
