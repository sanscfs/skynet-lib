"""End-to-end server flow with all gates and a fake handler.

These tests bypass HTTP/FastAPI -- AgentServer.handle() is the
synchronous entry point and the FastAPI router just calls it.
"""

from __future__ import annotations

from skynet_orchestration import tokens
from skynet_orchestration.envelopes import (
    AgentCall,
    AgentResult,
    BudgetGrant,
    ThreadHandle,
    WorkEstimate,
)
from skynet_orchestration.server import AgentServer, HandlerContext


def _build_call(
    *,
    target="sre",
    caller="main",
    purpose="user_task",
    reason=None,
    query="check pods",
    call_chain=None,
    invocation_id="inv1",
    root_invocation_id=None,
) -> AgentCall:
    root = root_invocation_id or invocation_id
    return AgentCall(
        invocation_id=invocation_id,
        root_invocation_id=root,
        target=target,
        caller=caller,
        query=query,
        purpose=purpose,
        reason=reason,
        call_chain=list(call_chain) if call_chain else [],
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


def _ok_handler(ctx: HandlerContext, call: AgentCall) -> AgentResult:
    """Minimal handler: emits one progress event and returns ok."""
    ctx.add_tokens(200)
    ctx.add_tool_call(1)
    ctx.emit("progress", "looked at thing")
    return AgentResult(
        invocation_id=call.invocation_id,
        status="ok",
        output="thing checked",
        actuals=ctx.actuals(),
    )


def _make_server(redis, cosine, specificity, handler=_ok_handler) -> AgentServer:
    return AgentServer(
        name="sre",
        target_description="diagnose kubernetes infrastructure pods nodes argocd drift",
        handler=handler,
        redis_client=redis,
        similarity_fn=cosine,
        specificity_fn=specificity,
        caller_state_fn=lambda: "current state ok",
    )


def test_happy_path_top_level_call(redis, cosine, specificity):
    server = _make_server(redis, cosine, specificity)
    call = _build_call(purpose="user_task", reason=None)
    result = server.handle(call)
    assert result.status == "ok"
    assert result.actuals.tokens_used == 200
    assert result.actuals.tool_calls_made == 1


def test_token_verification_rejects_bad_token(redis, cosine, specificity):
    server = _make_server(redis, cosine, specificity)
    call = _build_call()
    bad = call.model_copy(update={"caller_token": "v1.AAAA.BBBB"})
    result = server.handle(bad)
    assert result.status == "rejected"
    assert result.rejected_by_gate == "token"


def test_call_chain_appended_server_side(redis, cosine, specificity):
    captured = {}

    def handler(ctx, call):
        captured["chain"] = list(call.call_chain)
        ctx.add_tokens(10)
        return AgentResult(
            invocation_id=call.invocation_id,
            status="ok",
            output="",
            actuals=ctx.actuals(),
        )

    server = _make_server(redis, cosine, specificity, handler=handler)
    call = _build_call(call_chain=["main"])
    server.handle(call)
    assert captured["chain"] == ["main", "sre"]


def test_cycle_gate_rejects(redis, cosine, specificity):
    server = _make_server(redis, cosine, specificity)
    # Try to call sre when sre is already on the stack.
    call = _build_call(call_chain=["main", "sre", "music"])
    result = server.handle(call)
    assert result.status == "rejected"
    assert result.rejected_by_gate == "cycle"


def test_handler_exception_becomes_error_result(redis, cosine, specificity):
    def boom(ctx, call):
        raise RuntimeError("kaboom")

    server = _make_server(redis, cosine, specificity, handler=boom)
    result = server.handle(_build_call())
    assert result.status == "error"
    assert "kaboom" in (result.error or "")


def test_lateral_call_passes_when_reason_matches(redis, cosine, specificity):
    """Music → SRE for a postgres recovery scenario."""
    invocation_id = "inv_lat"
    call = AgentCall(
        invocation_id=invocation_id,
        root_invocation_id="root_user_turn",
        target="sre",
        caller="music",
        query="postgres reachable from music pod",
        purpose="self_recovery",
        reason="postgres connection timeout from music recommendations pipeline",
        call_chain=["main", "music"],
        thread=ThreadHandle(room_id="!r", thread_root="$t"),
        estimate=WorkEstimate(
            tokens_needed=500,
            tool_calls_expected=2,
            time_ms=5000,
            confidence=0.5,
        ),
        granted=BudgetGrant(tokens=1000, tool_calls=4, time_ms=10_000),
        caller_token=tokens.mint(invocation_id=invocation_id, caller="music"),
    )

    server = AgentServer(
        name="sre",
        target_description="postgres connection timeout investigate kubernetes",
        handler=_ok_handler,
        redis_client=redis,
        similarity_fn=cosine,
        specificity_fn=specificity,
        caller_state_fn=lambda: "postgres connection timeout",
        # Conservative cosine-jaccard means we need to relax the threshold
        # for the test; production uses real embeddings with much higher overlap.
        justification_target_threshold=0.20,
        justification_state_threshold=0.20,
    )
    result = server.handle(call)
    assert result.status == "ok"


def test_lateral_call_blocked_off_topic(redis, cosine, specificity):
    """Music asks SRE for something semantically unrelated."""
    invocation_id = "inv_off"
    call = AgentCall(
        invocation_id=invocation_id,
        root_invocation_id="root2",
        target="sre",
        caller="music",
        query="recommend a track",
        purpose="self_recovery",
        reason="my recommendations feel weird and i want vibes",
        call_chain=["main", "music"],
        thread=ThreadHandle(room_id="!r", thread_root="$t"),
        estimate=WorkEstimate(
            tokens_needed=500,
            tool_calls_expected=2,
            time_ms=5000,
            confidence=0.5,
        ),
        granted=BudgetGrant(tokens=1000, tool_calls=4, time_ms=10_000),
        caller_token=tokens.mint(invocation_id=invocation_id, caller="music"),
    )
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
    result = server.handle(call)
    assert result.status == "rejected"
    assert result.rejected_by_gate == "justification"
