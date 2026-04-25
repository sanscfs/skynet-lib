"""AgentServer derives gate thresholds from history (Phase 7).

Cold-start fallback:
- empty corpus → static constructor default
- < min_samples → static constructor default

Once enough samples are recorded, the server uses
- p25 of ``<metric>.accept`` for justification thresholds
- p75 of ``<metric>.reject`` for the repeat threshold

The recording happens inline in the gates themselves: every cosine
they compute pushes one sample (split by accept/reject polarity into
separate metric suffixes) so the corpus grows from real traffic.
"""

from __future__ import annotations

from skynet_orchestration import calibration, tokens
from skynet_orchestration.envelopes import (
    AgentCall,
    AgentResult,
    BudgetGrant,
    ThreadHandle,
    WorkEstimate,
)
from skynet_orchestration.server import AgentServer, HandlerContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_handler(ctx: HandlerContext, call: AgentCall) -> AgentResult:
    ctx.add_tokens(10)
    return AgentResult(
        invocation_id=call.invocation_id,
        status="ok",
        output="ok",
        actuals=ctx.actuals(),
    )


def _build_call(
    *,
    invocation_id="inv1",
    caller="music",
    target="sre",
    purpose="self_recovery",
    reason="postgres connection timeout from music pod",
    query="postgres reachable from music pod",
    call_chain=("main", "music"),
) -> AgentCall:
    return AgentCall(
        invocation_id=invocation_id,
        root_invocation_id=f"root_{invocation_id}",
        target=target,
        caller=caller,
        query=query,
        purpose=purpose,
        reason=reason,
        call_chain=list(call_chain),
        thread=ThreadHandle(room_id="!r", thread_root="$t"),
        estimate=WorkEstimate(
            tokens_needed=500,
            tool_calls_expected=2,
            time_ms=5000,
            confidence=0.5,
        ),
        granted=BudgetGrant(tokens=1000, tool_calls=4, time_ms=10_000),
        caller_token=tokens.mint(invocation_id=invocation_id, caller=caller),
    )


def _make_server(redis, cosine, specificity, **kw) -> AgentServer:
    defaults = dict(
        name="sre",
        target_description="postgres connection timeout investigate kubernetes",
        handler=_ok_handler,
        redis_client=redis,
        similarity_fn=cosine,
        specificity_fn=specificity,
        caller_state_fn=lambda: "postgres connection timeout",
        # Use slightly looser-than-default thresholds so the
        # cosine-jaccard fixture passes the test calls.
        repeat_threshold=0.85,
        justification_target_threshold=0.20,
        justification_state_threshold=0.20,
    )
    defaults.update(kw)
    return AgentServer(**defaults)


# ---------------------------------------------------------------------------
# 1. Cold start: no samples → constructor default
# ---------------------------------------------------------------------------


def test_cold_start_uses_constructor_default(redis, cosine, specificity):
    """Empty corpus: derived threshold == constructor default."""
    server = _make_server(redis, cosine, specificity)

    # No history yet, so all derived helpers must return the static defaults.
    assert server._derive_repeat_threshold("music", "sre") == 0.85
    assert server._derive_justification_threshold("music", "sre", "justification_target_cosine", 0.20) == 0.20
    assert server._derive_justification_threshold("music", "sre", "justification_state_cosine", 0.20) == 0.20


# ---------------------------------------------------------------------------
# 2. Insufficient samples: <20 → still default
# ---------------------------------------------------------------------------


def test_insufficient_samples_falls_back(redis, cosine, specificity):
    """Below min_samples=20 the snapshot returns None and we fall back."""
    for v in [0.3, 0.4, 0.5]:  # 3 samples, way under threshold
        calibration.record_threshold_sample(
            redis,
            caller="music",
            target="sre",
            metric="justification_target_cosine.accept",
            value=v,
        )
    server = _make_server(redis, cosine, specificity)
    derived = server._derive_justification_threshold("music", "sre", "justification_target_cosine", 0.40)
    assert derived == 0.40


# ---------------------------------------------------------------------------
# 3. Sufficient accept samples → p25 of accepts
# ---------------------------------------------------------------------------


def test_sufficient_accepts_yields_p25(redis, cosine, specificity):
    """Justification threshold settles at p25 of historical accepts."""
    # 20 evenly-spaced accept-side cosines in [0.30, 0.68]
    accept_values = [0.30 + i * 0.02 for i in range(20)]
    for v in accept_values:
        calibration.record_threshold_sample(
            redis,
            caller="music",
            target="sre",
            metric="justification_target_cosine.accept",
            value=v,
        )

    snap = calibration.threshold_snapshot(
        redis,
        caller="music",
        target="sre",
        metric="justification_target_cosine.accept",
    )
    assert snap is not None
    expected_p25 = snap.p25

    server = _make_server(redis, cosine, specificity, justification_target_threshold=0.40)
    derived = server._derive_justification_threshold("music", "sre", "justification_target_cosine", 0.40)
    assert derived == expected_p25
    # And: p25 of these accepts should be ≤ the static default of 0.40,
    # i.e. the gate has *learned to be looser* — admitting historically
    # acceptable calls that the operator-guess constant would block.
    assert derived <= 0.40


def test_sufficient_rejects_yields_p75_for_repeat(redis, cosine, specificity):
    """Repeat threshold settles at p75 of historical rejects."""
    # 20 reject-side cosines in [0.86, 0.96] — these are similarities
    # that the gate previously blocked.
    reject_values = [0.86 + (i % 11) * 0.01 for i in range(20)]
    for v in reject_values:
        calibration.record_threshold_sample(
            redis,
            caller="music",
            target="sre",
            metric="repeat_cosine.reject",
            value=v,
        )

    snap = calibration.threshold_snapshot(
        redis,
        caller="music",
        target="sre",
        metric="repeat_cosine.reject",
    )
    assert snap is not None
    expected_p75 = snap.p75

    server = _make_server(redis, cosine, specificity, repeat_threshold=0.85)
    derived = server._derive_repeat_threshold("music", "sre")
    assert derived == expected_p75
    # The reject bucket lives above the static default by definition
    # (only hits go in there), so derived ≥ default.
    assert derived >= 0.85


# ---------------------------------------------------------------------------
# 4. Mixed samples: snapshot reader correctly partitions accept vs reject
# ---------------------------------------------------------------------------


def test_mixed_polarities_partition_correctly(redis, cosine, specificity):
    """Accept and reject buckets are independent; no cross-talk."""
    # 25 accepts in [0.10, 0.34] and 25 rejects in [0.90, 0.99].
    for i in range(25):
        calibration.record_threshold_sample(
            redis,
            caller="music",
            target="sre",
            metric="repeat_cosine.accept",
            value=0.10 + i * 0.01,
        )
    for i in range(25):
        calibration.record_threshold_sample(
            redis,
            caller="music",
            target="sre",
            metric="repeat_cosine.reject",
            value=0.90 + (i % 10) * 0.01,
        )

    accept_snap = calibration.threshold_snapshot(
        redis,
        caller="music",
        target="sre",
        metric="repeat_cosine.accept",
    )
    reject_snap = calibration.threshold_snapshot(
        redis,
        caller="music",
        target="sre",
        metric="repeat_cosine.reject",
    )
    assert accept_snap is not None
    assert reject_snap is not None
    # The two distributions are well separated; if reads were leaking
    # across buckets these inequalities would fail.
    assert accept_snap.p75 < reject_snap.p25

    # The repeat-threshold derivation reads ONLY the reject side, so
    # adding accept samples must not move it.
    server = _make_server(redis, cosine, specificity)
    derived = server._derive_repeat_threshold("music", "sre")
    assert derived == reject_snap.p75


# ---------------------------------------------------------------------------
# 5. End-to-end: server.handle() records both polarities
# ---------------------------------------------------------------------------


def test_handle_records_accept_samples_into_calibration(redis, cosine, specificity):
    """A passing call must drop accept-side samples into the corpus."""
    server = _make_server(redis, cosine, specificity)

    # Calls into the same server: cosine-jaccard between
    # reason="postgres connection timeout from music pod" and
    # target_description="postgres connection timeout investigate kubernetes"
    # gives sim ≈ 4/9 = 0.44, well above the loosened 0.20 threshold.
    for i in range(3):
        result = server.handle(_build_call(invocation_id=f"inv_ok_{i}"))
        assert result.status == "ok", f"call {i} failed: {result.error}"

    # Each pass records one target_cosine + one state_cosine accept.
    # Storage is now a ZSET (logical-time decay, Phase 8) so we read
    # via ``zrange`` instead of the legacy ``lrange``.
    target_key = "orchestration:calibration:thresh:music:sre:justification_target_cosine.accept"
    state_key = "orchestration:calibration:thresh:music:sre:justification_state_cosine.accept"
    assert len(redis.zrange(target_key, 0, -1)) == 3
    assert len(redis.zrange(state_key, 0, -1)) == 3

    # And no reject-side samples for justification (every comparison passed).
    assert (
        len(
            redis.zrange(
                "orchestration:calibration:thresh:music:sre:justification_target_cosine.reject",
                0,
                -1,
            )
        )
        == 0
    )


def test_handle_records_reject_samples_when_gate_blocks(redis, cosine, specificity):
    """A justification-rejected call records a reject-side sample."""
    server = AgentServer(
        name="sre",
        target_description="diagnose kubernetes pods nodes argocd",
        handler=_ok_handler,
        redis_client=redis,
        similarity_fn=cosine,
        specificity_fn=specificity,
        caller_state_fn=lambda: "music engine working ok",
        justification_target_threshold=0.30,
    )
    # reason has no overlap with the target_description → cosine ≈ 0 → reject.
    call = _build_call(
        invocation_id="inv_reject",
        caller="music",
        target="sre",
        reason="my recommendations feel weird and i want vibes",
        purpose="self_recovery",
        query="recommend a track",
    )
    result = server.handle(call)
    assert result.status == "rejected"
    assert result.rejected_by_gate == "justification"

    reject_key = "orchestration:calibration:thresh:music:sre:justification_target_cosine.reject"
    samples = redis.zrange(reject_key, 0, -1)
    assert len(samples) == 1
