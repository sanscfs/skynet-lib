"""Calibration: history k-NN + adaptive percentile thresholds."""

from __future__ import annotations

from skynet_orchestration import calibration
from skynet_orchestration.envelopes import WorkActuals


def test_record_and_load_history_round_trip(redis):
    actuals = WorkActuals(tokens_used=4000, tool_calls_made=3, time_ms=15_000)
    calibration.record_outcome(redis, "sre", "inv1", "check postgres", actuals)
    calibration.record_outcome(redis, "sre", "inv2", "check qdrant", actuals)
    hist = calibration.load_history(redis, "sre")
    assert len(hist) == 2
    queries = {h.query for h in hist}
    assert queries == {"check postgres", "check qdrant"}


def test_history_trim_keeps_most_recent(redis):
    """Trim semantics: only the last ``keep_last`` survive."""
    actuals = WorkActuals(tokens_used=100, tool_calls_made=1, time_ms=100)
    for i in range(15):
        calibration.record_outcome(redis, "music", f"i{i}", f"q{i}", actuals, keep_last=10)
    hist = calibration.load_history(redis, "music")
    assert len(hist) == 10
    # First 5 should be gone.
    seen_queries = {h.query for h in hist}
    assert "q0" not in seen_queries
    assert "q14" in seen_queries


def test_baseline_estimate_uses_median(redis):
    actuals_a = WorkActuals(tokens_used=1000, tool_calls_made=1, time_ms=5_000)
    actuals_b = WorkActuals(tokens_used=2000, tool_calls_made=2, time_ms=10_000)
    actuals_c = WorkActuals(tokens_used=8000, tool_calls_made=10, time_ms=60_000)  # outlier
    for i, a in enumerate([actuals_a, actuals_b, actuals_b, actuals_b, actuals_c]):
        calibration.record_outcome(redis, "sre", f"i{i}", "investigate", a)
    history = calibration.load_history(redis, "sre")
    # Trivial similarity: same query means cosine = 1 for all.
    est = calibration.baseline_estimate(
        "investigate",
        history,
        similarity_fn=lambda a, b: 1.0,
    )
    assert est is not None
    # Median of [1000, 2000, 2000, 2000, 8000] = 2000 -- outlier ignored.
    assert est.tokens_needed == 2000


def test_baseline_estimate_returns_none_below_min(redis):
    actuals = WorkActuals(tokens_used=100, tool_calls_made=1, time_ms=100)
    calibration.record_outcome(redis, "sre", "i0", "q", actuals)
    history = calibration.load_history(redis, "sre")
    est = calibration.baseline_estimate("q", history, similarity_fn=lambda a, b: 1.0)
    assert est is None


def test_threshold_snapshot_returns_percentiles(redis):
    for v in [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
        0.05,
    ]:
        calibration.record_threshold_sample(
            redis,
            caller="music",
            target="sre",
            metric="estimate_accuracy_tokens",
            value=v,
        )
    snap = calibration.threshold_snapshot(
        redis,
        caller="music",
        target="sre",
        metric="estimate_accuracy_tokens",
        min_samples=10,
    )
    assert snap is not None
    assert snap.sample_size == 20
    assert 0.0 <= snap.p25 < snap.p50 < snap.p75 <= 1.0


def test_threshold_snapshot_none_below_min_samples(redis):
    for v in [0.1, 0.2, 0.3]:
        calibration.record_threshold_sample(
            redis,
            caller="music",
            target="sre",
            metric="m",
            value=v,
        )
    snap = calibration.threshold_snapshot(
        redis,
        caller="music",
        target="sre",
        metric="m",
        min_samples=20,
    )
    assert snap is None
