"""Envelope validation rules."""

from __future__ import annotations

import pytest
from skynet_orchestration.envelopes import (
    AgentCall,
    AgentResult,
    BudgetGrant,
    ThreadHandle,
    WorkActuals,
    WorkEstimate,
)


def _thread() -> ThreadHandle:
    return ThreadHandle(room_id="!room:matrix", thread_root="$thread")


def _grant() -> BudgetGrant:
    return BudgetGrant(tokens=1000, tool_calls=3, time_ms=10_000)


def _estimate() -> WorkEstimate:
    return WorkEstimate(
        tokens_needed=600,
        tool_calls_expected=2,
        time_ms=5_000,
        confidence=0.6,
        complexity="medium",
    )


def _call(**kw) -> AgentCall:
    base = dict(
        invocation_id="inv1",
        root_invocation_id="inv1",
        target="sre",
        caller="main",
        query="check pods",
        purpose="user_task",
        thread=_thread(),
        estimate=_estimate(),
        granted=_grant(),
        caller_token="placeholder",
    )
    base.update(kw)
    return AgentCall(**base)


def test_user_task_does_not_require_reason():
    """Top-level main delegations skip the reason check."""
    call = _call(purpose="user_task", reason=None)
    assert call.reason is None


def test_self_recovery_requires_reason():
    """Lateral self-recovery calls must justify themselves."""
    with pytest.raises(ValueError):
        _call(purpose="self_recovery", reason=None)


def test_delegation_requires_reason():
    """Delegation between siblings must justify itself."""
    with pytest.raises(ValueError):
        _call(purpose="delegation", reason=None)


def test_estimate_scaled_preserves_confidence():
    """``WorkEstimate.scaled`` only multiplies the resource fields."""
    e = _estimate().scaled(2.0)
    assert e.tokens_needed == 1200
    assert e.tool_calls_expected == 4
    assert e.confidence == 0.6
    assert e.complexity == "medium"


def test_result_carries_actuals():
    """AgentResult round-trips actuals + status."""
    r = AgentResult(
        invocation_id="inv1",
        status="ok",
        output="done",
        actuals=WorkActuals(tokens_used=400, tool_calls_made=1, time_ms=2000),
    )
    assert r.status == "ok"
    assert r.actuals.tokens_used == 400
    assert r.error is None


def test_protocol_version_is_pinned():
    """A new envelope picks up the current protocol version."""
    from skynet_orchestration.envelopes import PROTOCOL_VERSION

    assert _call().protocol_version == PROTOCOL_VERSION
