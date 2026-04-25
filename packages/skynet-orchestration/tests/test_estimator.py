"""Estimator + grant conversion."""

from __future__ import annotations

from skynet_orchestration import calibration, estimator
from skynet_orchestration.envelopes import WorkActuals


def test_structural_fallback_scales_with_entities():
    """A query packed with entity anchors gets a larger estimate."""
    plain = estimator.structural_fallback("hello there")
    rich = estimator.structural_fallback("investigate skynet-ingest namespace pod NodeOOM /var/log err 503")
    assert rich.tokens_needed > plain.tokens_needed
    # Confidence stays low because this is the fallback path.
    assert rich.confidence < 0.5


def test_structural_features_no_phrase_lists():
    """structural_features uses regex anchors, not vocabulary."""
    feats = estimator.structural_features("kubectl get pods -n skynet-ingest")
    assert feats["token_count"] >= 4
    assert feats["entity_anchors"] >= 1


def test_grant_from_estimate_inflates_low_confidence():
    """A low-confidence estimate gets a bigger buffer."""
    from skynet_orchestration.envelopes import WorkEstimate

    cert = WorkEstimate(
        tokens_needed=1000,
        tool_calls_expected=3,
        time_ms=10_000,
        confidence=1.0,
        complexity="low",
    )
    unsure = WorkEstimate(
        tokens_needed=1000,
        tool_calls_expected=3,
        time_ms=10_000,
        confidence=0.0,
        complexity="unknown",
    )
    g_cert = estimator.grant_from_estimate(cert)
    g_unsure = estimator.grant_from_estimate(unsure)
    assert g_unsure.tokens > g_cert.tokens


def test_composite_uses_history_when_available(redis):
    """When history has enough samples, baseline beats structural fallback."""
    target = "sre"
    # Seed history: the same query has cost 6000 tokens repeatedly.
    for i in range(10):
        actuals = WorkActuals(tokens_used=6000, tool_calls_made=4, time_ms=30_000)
        calibration.record_outcome(redis, target, f"i{i}", "investigate postgres timeout", actuals)

    history = calibration.load_history(redis, target)

    def jaccard(a, b):
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    composite = estimator.CompositeEstimator(similarity_fn=jaccard, history=history)
    est = composite.estimate("investigate postgres timeout from music pod")
    # Median historical actuals = 6000 tokens; the estimate should be near
    # that, not the conservative ~800-1500 structural fallback.
    assert 5000 <= est.tokens_needed <= 7000


def test_composite_falls_back_when_history_thin(redis):
    """k-NN with too few samples falls back to structural."""
    target = "sre"
    # Only one record -- below min_neighbours.
    actuals = WorkActuals(tokens_used=6000, tool_calls_made=4, time_ms=30_000)
    calibration.record_outcome(redis, target, "i0", "blah", actuals)
    history = calibration.load_history(redis, target)

    composite = estimator.CompositeEstimator(
        similarity_fn=lambda a, b: 0.5,
        history=history,
    )
    est = composite.estimate("hello world")
    # Structural fallback for a 2-token query gives a small estimate.
    assert est.tokens_needed < 3000
