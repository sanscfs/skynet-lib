"""Phase 8 deliverables: non-gating user_task samples, prometheus
metrics, logical-time decay calibration, LLM-judge estimator.

Each test isolates one cross-cutting change so failures finger the
culprit precisely.
"""

from __future__ import annotations

from skynet_orchestration import calibration, gates, metrics, tokens
from skynet_orchestration.envelopes import (
    AgentCall,
    AgentResult,
    BudgetGrant,
    ThreadHandle,
    WorkActuals,
    WorkEstimate,
)
from skynet_orchestration.estimator import (
    CompositeEstimator,
    LLMJudgeEstimator,
    _coerce_work_estimate,
)
from skynet_orchestration.server import AgentServer

# ---------------------------------------------------------------------------
# Item 1: user_task records accept-side samples without gating
# ---------------------------------------------------------------------------


def test_user_task_with_reason_records_accept_samples(cosine):
    """purpose=user_task + reason supplied: cosines fed into corpus
    as ``accept`` samples without being gated."""
    recorded: list[tuple[str, float, bool]] = []

    def _rec(metric: str, value: float, *, accepted: bool) -> None:
        recorded.append((metric, value, accepted))

    call = AgentCall(
        invocation_id="inv1",
        root_invocation_id="root1",
        target="sre",
        caller="main",
        query="check pods",
        purpose="user_task",
        reason="user asked about pods in skynet-ingest",
        call_chain=["main"],
        thread=ThreadHandle(room_id="!r", thread_root="$t"),
        estimate=WorkEstimate(
            tokens_needed=100,
            tool_calls_expected=1,
            time_ms=1000,
            confidence=0.5,
        ),
        granted=BudgetGrant(tokens=200, tool_calls=2, time_ms=2000),
        caller_token="tok",
    )
    rej = gates.check_justification(
        call,
        target_description="diagnose pods kubernetes investigate",
        caller_state="sre ok kubectl reachable",
        cosine_fn=cosine,
        target_threshold=0.99,  # absurdly tight; would fail if it gated
        record_sample=_rec,
    )
    # User-task always passes regardless of cosine threshold.
    assert rej is None
    # Both target and state cosines were recorded as accepts.
    metric_names = {m for m, _, _ in recorded}
    assert "justification_target_cosine" in metric_names
    assert "justification_state_cosine" in metric_names
    assert all(accepted for _, _, accepted in recorded)


def test_user_task_without_reason_records_nothing(cosine):
    """No reason → no cosine to compute → no sample recorded."""
    recorded: list[tuple[str, float, bool]] = []

    def _rec(metric: str, value: float, *, accepted: bool) -> None:
        recorded.append((metric, value, accepted))

    call = AgentCall(
        invocation_id="inv2",
        root_invocation_id="root2",
        target="sre",
        caller="main",
        query="check pods",
        purpose="user_task",
        reason=None,  # no reason — nothing to compare
        call_chain=["main"],
        thread=ThreadHandle(room_id="!r", thread_root="$t"),
        estimate=WorkEstimate(
            tokens_needed=100,
            tool_calls_expected=1,
            time_ms=1000,
            confidence=0.5,
        ),
        granted=BudgetGrant(tokens=200, tool_calls=2, time_ms=2000),
        caller_token="tok",
    )
    rej = gates.check_justification(
        call,
        target_description="diagnose pods kubernetes",
        caller_state="sre ok",
        cosine_fn=cosine,
        record_sample=_rec,
    )
    assert rej is None
    assert recorded == []


def test_user_task_records_target_only_when_state_missing(cosine):
    """No caller_state → only target cosine recorded."""
    recorded: list[tuple[str, float, bool]] = []

    def _rec(metric: str, value: float, *, accepted: bool) -> None:
        recorded.append((metric, value, accepted))

    call = AgentCall(
        invocation_id="inv3",
        root_invocation_id="root3",
        target="sre",
        caller="main",
        query="anything",
        purpose="user_task",
        reason="some reason",
        call_chain=["main"],
        thread=ThreadHandle(room_id="!r", thread_root="$t"),
        estimate=WorkEstimate(
            tokens_needed=100,
            tool_calls_expected=1,
            time_ms=1000,
            confidence=0.5,
        ),
        granted=BudgetGrant(tokens=200, tool_calls=2, time_ms=2000),
        caller_token="tok",
    )
    gates.check_justification(
        call,
        target_description="something",
        caller_state=None,
        cosine_fn=cosine,
        record_sample=_rec,
    )
    metric_names = {m for m, _, _ in recorded}
    assert metric_names == {"justification_target_cosine"}


# ---------------------------------------------------------------------------
# Item 4: prometheus metrics — record_invocation / record_rejection / latest_text
# ---------------------------------------------------------------------------


def test_metrics_module_no_op_when_unavailable():
    """Calls must not raise even when prometheus-client is missing."""
    # We don't simulate the missing-import path here (it's exercised
    # at import time in the test environment). What we do verify is
    # that the public helpers tolerate ``None`` / unknown labels.
    metrics.record_invocation(
        caller="main",
        target="sre",
        purpose="user_task",
        status="ok",
        duration_seconds=0.123,
    )
    metrics.record_rejection(caller="main", target="sre", gate="cycle")
    body, ctype = metrics.latest_text()
    assert isinstance(body, (bytes, bytearray))
    assert "text" in ctype


def test_metrics_increments_through_agent_server(redis, cosine, specificity):
    """An ``ok`` AgentServer.handle() bumps the invocations counter."""
    if not metrics.is_available():
        # No-op build; skip the counter assertion but still execute
        # the server to make sure metrics calls don't raise.
        pass

    def _ok(ctx, call):
        ctx.add_tokens(10)
        return AgentResult(
            invocation_id=call.invocation_id,
            status="ok",
            output="ok",
            actuals=ctx.actuals(),
        )

    server = AgentServer(
        name="sre",
        target_description="diag k8s pods",
        handler=_ok,
        redis_client=redis,
        similarity_fn=cosine,
        specificity_fn=specificity,
    )
    invocation_id = "inv_metrics"
    call = AgentCall(
        invocation_id=invocation_id,
        root_invocation_id=invocation_id,
        target="sre",
        caller="main",
        query="check pods",
        purpose="user_task",
        reason=None,
        call_chain=["main"],
        thread=ThreadHandle(room_id="!r", thread_root="$t"),
        estimate=WorkEstimate(
            tokens_needed=500,
            tool_calls_expected=2,
            time_ms=5000,
            confidence=0.5,
        ),
        granted=BudgetGrant(tokens=1000, tool_calls=4, time_ms=10_000),
        caller_token=tokens.mint(invocation_id=invocation_id, caller="main"),
    )
    before = 0.0
    if metrics.is_available():
        # Sample the current value; the counter is process-global.
        sample = metrics.INVOCATIONS_TOTAL.labels(  # type: ignore[union-attr]
            caller="main", target="sre", purpose="user_task", status="ok"
        )
        before = sample._value.get()
    result = server.handle(call)
    assert result.status == "ok"
    if metrics.is_available():
        sample = metrics.INVOCATIONS_TOTAL.labels(  # type: ignore[union-attr]
            caller="main", target="sre", purpose="user_task", status="ok"
        )
        after = sample._value.get()
        assert after == before + 1.0


def test_metrics_records_gate_rejection(redis, cosine, specificity):
    """Cycle rejection bumps the gate_rejections counter."""

    def _ok(ctx, call):
        return AgentResult(
            invocation_id=call.invocation_id,
            status="ok",
            output="",
            actuals=ctx.actuals(),
        )

    server = AgentServer(
        name="sre",
        target_description="x",
        handler=_ok,
        redis_client=redis,
        similarity_fn=cosine,
        specificity_fn=specificity,
    )
    invocation_id = "inv_cyc"
    call = AgentCall(
        invocation_id=invocation_id,
        root_invocation_id=invocation_id,
        target="sre",
        caller="main",
        query="check",
        purpose="user_task",
        call_chain=["main", "sre", "music"],  # sre already on chain → cycle
        thread=ThreadHandle(room_id="!r", thread_root="$t"),
        estimate=WorkEstimate(
            tokens_needed=10,
            tool_calls_expected=1,
            time_ms=100,
            confidence=0.5,
        ),
        granted=BudgetGrant(tokens=20, tool_calls=2, time_ms=200),
        caller_token=tokens.mint(invocation_id=invocation_id, caller="main"),
    )
    before = 0.0
    if metrics.is_available():
        sample = metrics.GATE_REJECTIONS_TOTAL.labels(  # type: ignore[union-attr]
            caller="main", target="sre", gate="cycle"
        )
        before = sample._value.get()
    result = server.handle(call)
    assert result.status == "rejected"
    assert result.rejected_by_gate == "cycle"
    if metrics.is_available():
        sample = metrics.GATE_REJECTIONS_TOTAL.labels(  # type: ignore[union-attr]
            caller="main", target="sre", gate="cycle"
        )
        after = sample._value.get()
        assert after == before + 1.0


# ---------------------------------------------------------------------------
# Item 6: logical-time decay calibration — old samples weigh less
# ---------------------------------------------------------------------------


def test_threshold_snapshot_decay_weights_old_samples_less(redis):
    """30 old high-value samples + 30 fresh low-value samples should
    yield a percentile snapshot biased toward the FRESH distribution
    once the half-life kicks in.
    """
    # Half-life=5 ticks → after writing 30 fresh samples the original
    # 30 are 6+ half-lives old (weight < 0.02 each); the percentile
    # is dominated by the recent low-value distribution.
    for v in [0.95] * 30:
        calibration.record_threshold_sample(
            redis,
            caller="x",
            target="y",
            metric="m",
            value=v,
            half_life_ticks=5,
        )
    for v in [0.05] * 30:
        calibration.record_threshold_sample(
            redis,
            caller="x",
            target="y",
            metric="m",
            value=v,
            half_life_ticks=5,
        )
    snap = calibration.threshold_snapshot(
        redis,
        caller="x",
        target="y",
        metric="m",
        half_life_ticks=5,
    )
    assert snap is not None
    # All percentiles pulled toward the fresh 0.05 distribution.
    assert snap.p50 < 0.5
    assert snap.p75 < 0.5


def test_threshold_snapshot_window_trim_by_ticks(redis):
    """``zremrangebyscore`` removes samples older than the window."""
    for v in [0.5] * 50:
        calibration.record_threshold_sample(
            redis,
            caller="x",
            target="y",
            metric="m",
            value=v,
            half_life_ticks=2,  # window = 5 * 2 = 10 ticks
        )
    # ZSET should have only ~10 entries left (one per tick within window).
    raw = redis.zrange("orchestration:calibration:thresh:x:y:m", 0, -1)
    # Allow some slack for the (tick-window) inclusivity edge case.
    assert len(raw) <= 12
    assert len(raw) >= 8


def test_legacy_list_migrates_to_zset_on_read(redis):
    """A pre-Phase-8 LIST is converted on first ``threshold_snapshot``."""
    key = "orchestration:calibration:thresh:m:s:legacy"
    # Seed a legacy LIST directly (mimicking pre-upgrade state).
    for v in [f"{0.30 + i * 0.02:.6f}" for i in range(20)]:
        redis.rpush(key, v)
    assert redis.type(key) == "list"
    # First snapshot triggers migration.
    snap = calibration.threshold_snapshot(
        redis,
        caller="m",
        target="s",
        metric="legacy",
    )
    assert snap is not None
    assert redis.type(key) == "zset"


def test_record_outcome_history_round_trip(redis):
    """Outcomes still load back after ZSET migration."""
    actuals = WorkActuals(tokens_used=4000, tool_calls_made=3, time_ms=15_000)
    calibration.record_outcome(redis, "sre", "inv1", "check postgres", actuals)
    calibration.record_outcome(redis, "sre", "inv2", "check qdrant", actuals)
    hist = calibration.load_history(redis, "sre")
    queries = {h.query for h in hist}
    assert "check postgres" in queries
    assert "check qdrant" in queries


# ---------------------------------------------------------------------------
# Item 5: LLMJudgeEstimator
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _FakeHttp:
    def __init__(self, payload, status_code: int = 200, raise_exc=None) -> None:
        self._payload = payload
        self._status_code = status_code
        self._raise_exc = raise_exc
        self.calls: list[dict] = []

    def post(self, url, *, json, timeout):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self._raise_exc is not None:
            raise self._raise_exc
        return _FakeResp(self._payload, status_code=self._status_code)


def test_llm_judge_returns_work_estimate_on_valid_response():
    http = _FakeHttp(
        {
            "tokens_needed": 1500,
            "tool_calls_expected": 3,
            "time_ms": 5000,
            "confidence": 0.7,
            "complexity": "medium",
        }
    )
    est = LLMJudgeEstimator(
        url="http://test/estimate",
        target="music",
        http_client=http,
    )
    result = est.estimate("recommend chill folk")
    assert result is not None
    assert result.tokens_needed == 1500
    assert result.confidence == 0.7
    assert result.complexity == "medium"
    assert http.calls and http.calls[0]["json"]["target"] == "music"


def test_llm_judge_returns_none_on_http_failure():
    http = _FakeHttp(payload={}, raise_exc=RuntimeError("network down"))
    est = LLMJudgeEstimator(
        url="http://test/estimate",
        target="music",
        http_client=http,
    )
    assert est.estimate("anything") is None


def test_llm_judge_returns_none_on_non_200():
    http = _FakeHttp(payload={"tokens_needed": 100, "time_ms": 100}, status_code=502)
    est = LLMJudgeEstimator(
        url="http://test/estimate",
        target="music",
        http_client=http,
    )
    assert est.estimate("anything") is None


def test_llm_judge_returns_none_on_invalid_payload():
    """Missing tokens_needed → coercion fails → caller falls back."""
    http = _FakeHttp(payload={"some_other_field": 42})
    est = LLMJudgeEstimator(
        url="http://test/estimate",
        target="music",
        http_client=http,
    )
    assert est.estimate("anything") is None


def test_composite_falls_back_when_judge_returns_none(redis):
    """An LLM judge that returns None must not break the composite."""
    actuals = WorkActuals(tokens_used=6000, tool_calls_made=4, time_ms=30_000)
    for i in range(10):
        calibration.record_outcome(redis, "sre", f"i{i}", "investigate postgres timeout", actuals)
    history = calibration.load_history(redis, "sre")

    class _DeadJudge:
        def estimate(self, query: str):
            return None

    composite = CompositeEstimator(
        similarity_fn=lambda a, b: 1.0,
        history=history,
        caller_estimator=_DeadJudge(),
    )
    est = composite.estimate("investigate postgres timeout")
    # History dominates; the dead judge is silently ignored.
    assert 5000 <= est.tokens_needed <= 7000


def test_agent_client_configure_llm_judge_swaps_estimator(redis):
    """Calling ``configure_llm_judge`` replaces the default estimator
    with one that consults the LLM judge first."""
    from skynet_orchestration import AgentClient

    http = _FakeHttp(
        {
            "tokens_needed": 12345,
            "tool_calls_expected": 7,
            "time_ms": 11000,
            "confidence": 0.6,
            "complexity": "medium",
        }
    )
    client = AgentClient(
        caller_name="test",
        redis_client=redis,
        target_registry={"sre": "http://x"},
    )
    # Before wiring: structural fallback (small).
    pre = client.estimate_query("sre", "investigate")
    assert pre.tokens_needed < 5000

    client.configure_llm_judge(url="http://test/estimate", http_client=http)
    post = client.estimate_query("sre", "investigate")
    # After wiring: judge response (12345 tokens) blended with
    # structural (~800-1500). Result should be markedly larger than
    # the structural-only baseline because the judge is the only
    # opinion in the room (history is empty).
    assert post.tokens_needed > pre.tokens_needed
    # Empty URL disables it.
    client.configure_llm_judge(url="")
    again = client.estimate_query("sre", "investigate")
    assert again.tokens_needed == pre.tokens_needed


def test_coerce_work_estimate_clamps_invalid_complexity():
    out = _coerce_work_estimate(
        {
            "tokens_needed": 100,
            "tool_calls_expected": 1,
            "time_ms": 100,
            "confidence": 0.5,
            "complexity": "absurd",
        }
    )
    assert out is not None
    assert out.complexity == "unknown"
