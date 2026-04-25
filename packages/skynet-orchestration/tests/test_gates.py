"""Each gate is independent; first failure short-circuits."""

from __future__ import annotations

from skynet_orchestration import gates
from skynet_orchestration.envelopes import (
    AgentCall,
    BudgetGrant,
    ThreadHandle,
    WorkEstimate,
)


def _call(
    *, target="sre", caller="main", query="check pods", purpose="user_task", reason=None, call_chain=None, **kw
) -> AgentCall:
    base = dict(
        invocation_id="inv_x",
        root_invocation_id="root1",
        target=target,
        caller=caller,
        query=query,
        purpose=purpose,
        reason=reason,
        call_chain=list(call_chain) if call_chain else [],
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
    base.update(kw)
    return AgentCall(**base)


# -- cycle ---------------------------------------------------------------


def test_cycle_blocks_self_re_entry():
    call = _call(target="music", call_chain=["main", "music", "sre"])
    rej = gates.check_cycle(call)
    assert rej is not None
    assert rej.gate == "cycle"


def test_cycle_allows_main_as_re_entry_point():
    """main is always allowed as a re-entry point (orchestrator)."""
    call = _call(target="main", call_chain=["main", "music"])
    assert gates.check_cycle(call) is None


def test_cycle_allows_first_call():
    call = _call(target="sre", call_chain=["main"])
    assert gates.check_cycle(call) is None


# -- repeat --------------------------------------------------------------


def test_repeat_blocks_near_identical_query(cosine):
    history = [
        gates.HistoricalCall(
            invocation_id="prev1",
            caller="music",
            target="sre",
            query="postgres timeout investigate",
        ),
    ]
    call = _call(
        caller="music",
        target="sre",
        query="postgres timeout investigate",  # identical
        purpose="self_recovery",
        reason="my db connections fail",
    )
    rej = gates.check_repeat(call, history=history, cosine_fn=cosine, threshold=0.85)
    assert rej is not None
    assert rej.gate == "repeat"


def test_repeat_allows_different_query_same_pair(cosine):
    history = [
        gates.HistoricalCall(
            invocation_id="prev1",
            caller="music",
            target="sre",
            query="postgres timeout investigate",
        ),
    ]
    call = _call(
        caller="music",
        target="sre",
        query="grafana dashboard render is broken",  # entirely different
        purpose="self_recovery",
        reason="my UI fails",
    )
    assert gates.check_repeat(call, history=history, cosine_fn=cosine, threshold=0.85) is None


def test_repeat_ignores_different_pairs(cosine):
    """Same query but different (caller, target) is not a repeat."""
    history = [
        gates.HistoricalCall(
            invocation_id="prev1",
            caller="movies",
            target="sre",
            query="postgres timeout",
        ),
    ]
    call = _call(
        caller="music",
        target="sre",  # different caller
        query="postgres timeout",
        purpose="self_recovery",
        reason="my db",
    )
    assert gates.check_repeat(call, history=history, cosine_fn=cosine, threshold=0.85) is None


# -- justification --------------------------------------------------------


def test_justification_skipped_for_user_task(cosine):
    """Top-level main delegations don't need to justify."""
    call = _call(purpose="user_task", reason=None)
    rej = gates.check_justification(
        call,
        target_description="anything",
        caller_state=None,
        cosine_fn=cosine,
    )
    assert rej is None


def test_justification_requires_reason_for_lateral():
    # The pydantic validator already enforces this; gates.check_justification
    # is defensive in case a future code path bypasses validation.
    import pytest

    with pytest.raises(Exception):
        _call(purpose="self_recovery", reason=None)


def test_justification_blocks_off_topic_reason(cosine):
    call = _call(
        caller="music",
        target="sre",
        purpose="self_recovery",
        reason="my recommendations feel off",
        query="something",
    )
    rej = gates.check_justification(
        call,
        target_description="diagnose kubernetes infrastructure pods nodes argocd",
        caller_state="my recommendations feel off",
        cosine_fn=cosine,
        target_threshold=0.40,
    )
    assert rej is not None
    assert rej.gate == "justification"


def test_justification_passes_when_reason_matches_target(cosine):
    call = _call(
        caller="music",
        target="sre",
        purpose="self_recovery",
        reason="postgres connection timeout from music pod",
        query="postgres reachable from music pod",
    )
    rej = gates.check_justification(
        call,
        target_description="postgres connection timeout investigate",
        caller_state="postgres connection timeout",
        cosine_fn=cosine,
        target_threshold=0.20,  # cosine-jaccard is conservative; relax for the test
        state_threshold=0.20,
    )
    assert rej is None


# -- convergence --------------------------------------------------------


def test_convergence_allows_first_call_on_pair(specificity):
    call = _call(caller="main", target="sre", query="check pods")
    assert gates.check_convergence(call, history=[], specificity_fn=specificity) is None


def test_convergence_blocks_less_specific_followup(specificity):
    history = [
        gates.HistoricalCall(
            invocation_id="prev",
            caller="main",
            target="sre",
            query="investigate skynet-ingest OOMKilled at 12:00 in skynet-ingest namespace",
        ),
    ]
    call = _call(
        caller="main",
        target="sre",
        query="check pods",  # vaguer than the previous, fewer entities
        purpose="user_task",
    )
    rej = gates.check_convergence(call, history=history, specificity_fn=specificity)
    assert rej is not None
    assert rej.gate == "convergence"


def test_convergence_allows_more_specific_followup(specificity):
    history = [
        gates.HistoricalCall(
            invocation_id="prev",
            caller="main",
            target="sre",
            query="check pods",
        ),
    ]
    call = _call(
        caller="main",
        target="sre",
        query="investigate skynet-ingest OOMKilled in skynet-ingest namespace at 12:00",
        purpose="user_task",
    )
    assert gates.check_convergence(call, history=history, specificity_fn=specificity) is None
