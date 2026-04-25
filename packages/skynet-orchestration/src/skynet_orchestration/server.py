"""Server side of the protocol -- one ``AgentServer`` per sub-agent.

Each Skynet sub-agent (sre / music / movies / ...) instantiates an
``AgentServer`` and registers a handler. The server is responsible
for everything the protocol guarantees:

- HMAC token verification (caller is who they say they are);
- appending its own name to ``call_chain`` so the topology is
  honest;
- running all four gates on every incoming call;
- enforcing the budget grant by feeding the handler a
  ``HandlerContext`` that tracks usage;
- emitting a synthetic ``progress`` event on call entry and
  ``conclusion`` on call exit;
- shape-conforming the AgentResult before responding.

Handler signature::

    def handler(ctx: HandlerContext, call: AgentCall) -> AgentResult: ...

The handler is the only place where domain logic lives. Everything
above it (gates, tokens, chronicle, streaming) is the library.

Optional: if FastAPI is installed (``[server]`` extra), the server
can hand back an ``APIRouter`` mounted at ``/invoke`` so a service
can ``app.include_router(server.router())`` and be done.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import budget as budget_mod
from . import gates, tokens
from .chronicle import emit_call_start
from .chronicle import emit_event as chronicle_emit
from .envelopes import (
    AgentCall,
    AgentEvent,
    AgentResult,
    BudgetGrant,
    WorkActuals,
)
from .streaming import close_stream, emit_to_thread

log = logging.getLogger("skynet_orchestration.server")


# ---------------------------------------------------------------------------
# Handler-side context (passed into the agent's handler)
# ---------------------------------------------------------------------------


@dataclass
class HandlerContext:
    """Per-invocation handle the handler uses for everything orchestration-y.

    It owns:
    - the caller's :class:`ThreadHandle` so emit() can stream there;
    - mutable usage counters that the server compares to the grant
      when the handler returns;
    - a hook to request a mid-flight budget extension.

    Handlers should NOT cache HandlerContext across invocations --
    each call gets a fresh one.
    """

    call: AgentCall
    service_name: str
    redis_client: object
    started_at: float = field(default_factory=time.time)
    tokens_used: int = 0
    tool_calls_made: int = 0
    extensions_used: int = 0

    def emit(self, event_type: str, content: str, **metadata) -> None:
        """Stream an event to the user's thread + chronicle."""
        evt = AgentEvent(
            invocation_id=self.call.invocation_id,
            type=event_type,  # type: ignore[arg-type]
            content=content,
            metadata=dict(metadata),
            ts=time.time(),
        )
        chronicle_emit(
            evt,
            root_invocation_id=self.call.root_invocation_id,
            target=self.call.target,
            caller=self.call.caller,
        )
        emit_to_thread(self.call.thread, evt, service_name=self.service_name)

    def add_tokens(self, n: int) -> None:
        self.tokens_used += max(0, n)

    def add_tool_call(self, n: int = 1) -> None:
        self.tool_calls_made += max(0, n)

    def request_extension(self, additional: BudgetGrant, reason: str) -> bool:
        """Ask the root pool for more budget.

        Returns True if granted. Counts against the per-tree
        extension cap. The handler should treat False as a polite
        signal to wrap up.
        """
        if self.call.granted.extensions_remaining <= self.extensions_used:
            self.emit("warning", f"extension cap reached for invocation; reason={reason}")
            return False
        ok = budget_mod.grant_extension(self.redis_client, self.call.root_invocation_id, additional)
        if ok:
            self.extensions_used += 1
            self.emit(
                "budget_request",
                content=reason,
                granted=True,
                additional_tokens=additional.tokens,
            )
        else:
            self.emit("budget_request", content=reason, granted=False)
        return ok

    def actuals(self) -> WorkActuals:
        return WorkActuals(
            tokens_used=self.tokens_used,
            tool_calls_made=self.tool_calls_made,
            time_ms=int((time.time() - self.started_at) * 1000),
            extensions_used=self.extensions_used,
        )


# ---------------------------------------------------------------------------
# Server itself
# ---------------------------------------------------------------------------

# Type aliases for the caller-injected callables. Both kept narrow so
# tests can pass plain functions without importing skynet_embedding.
SimilarityFn = Callable[[str, str], float]
SpecificityFn = Callable[[str], float]


class AgentServer:
    """One sub-agent's incoming-side dispatcher.

    Construction::

        server = AgentServer(
            name="sre",
            target_description="Diagnose Kubernetes/ArgoCD infrastructure...",
            handler=my_handler,
            redis_client=redis,
            similarity_fn=lambda a, b: skynet_embedding.cosine(a, b),
            specificity_fn=structural_specificity,
            caller_state_fn=lambda: my_self_check_summary(),
        )

    Then mount in FastAPI::

        app.include_router(server.router())
    """

    def __init__(
        self,
        *,
        name: str,
        target_description: str,
        handler: Callable[[HandlerContext, AgentCall], AgentResult],
        redis_client,
        similarity_fn: SimilarityFn,
        specificity_fn: SpecificityFn,
        caller_state_fn: Optional[Callable[[], str]] = None,
        repeat_threshold: float = 0.85,
        justification_target_threshold: float = 0.40,
        justification_state_threshold: float = 0.30,
    ):
        self.name = name
        self.target_description = target_description
        self._handler = handler
        self._redis = redis_client
        self._sim = similarity_fn
        self._specificity = specificity_fn
        self._caller_state_fn = caller_state_fn
        self._repeat_threshold = repeat_threshold
        self._just_target_thr = justification_target_threshold
        self._just_state_thr = justification_state_threshold

    # -- main entry point --------------------------------------------------

    def handle(self, call: AgentCall) -> AgentResult:
        """Verify, gate, dispatch, finalize. Synchronous to keep the
        handler's mental model simple."""

        # 1. Token check -- proves the call wasn't fabricated client-side.
        try:
            tokens.verify(
                call.caller_token,
                invocation_id=call.invocation_id,
                caller=call.caller,
            )
        except tokens.TokenError as e:
            return AgentResult(
                invocation_id=call.invocation_id,
                status="rejected",
                output="",
                actuals=WorkActuals(tokens_used=0, tool_calls_made=0, time_ms=0),
                error=f"token verification failed: {e}",
                rejected_by_gate="token",
            )

        # 2. Run all four gates against the INCOMING chain (i.e. without
        #    our own name yet). cycle check would otherwise trivially
        #    fire on every call because we'd be matching ourselves.
        history = self._load_invocation_history(call.root_invocation_id)
        rejection = (
            gates.check_cycle(call)
            or gates.check_repeat(
                call,
                history=history,
                cosine_fn=self._sim,
                threshold=self._repeat_threshold,
            )
            or gates.check_justification(
                call,
                target_description=self.target_description,
                caller_state=self._caller_state_fn() if self._caller_state_fn else None,
                cosine_fn=self._sim,
                target_threshold=self._just_target_thr,
                state_threshold=self._just_state_thr,
            )
            or gates.check_convergence(
                call,
                history=history,
                specificity_fn=self._specificity,
            )
        )
        if rejection is not None:
            log.info(
                "gate rejection invocation=%s gate=%s reason=%s",
                call.invocation_id,
                rejection.gate,
                rejection.reason,
            )
            return AgentResult(
                invocation_id=call.invocation_id,
                status="rejected",
                output="",
                actuals=WorkActuals(tokens_used=0, tool_calls_made=0, time_ms=0),
                error=rejection.reason,
                rejected_by_gate=rejection.gate,
            )

        # 3. Gates passed -- now write our own name onto the chain so
        #    downstream handlers and any further sub-calls have honest
        #    topology. Server-side authority on chain integrity:
        #    a buggy/spoofy client that omitted itself doesn't escape
        #    detection because the server stamps itself unconditionally.
        if not call.call_chain or call.call_chain[-1] != self.name:
            call = call.model_copy(update={"call_chain": [*call.call_chain, self.name]})

        # 4. Attach our MXID to the thread handle so streamed events
        #    attribute correctly in the user's Matrix thread.
        if not call.thread.service_mxid:
            call = call.model_copy(
                update={
                    "thread": call.thread.model_copy(update={"service_mxid": f"@skynet-{self.name}:matrix.sanscfs.dev"})
                }
            )

        # 5. Mark accepted-on-the-wire timestamp before any handler work.
        emit_call_start(call, target_received_at=time.time())

        # 6. Run the handler. Wrap in try/except so any exception
        #    becomes a structured error rather than a 500 to the caller.
        ctx = HandlerContext(
            call=call,
            service_name=self.name,
            redis_client=self._redis,
        )
        try:
            result = self._handler(ctx, call)
        except Exception as e:  # noqa: BLE001
            log.exception("handler raised for invocation=%s", call.invocation_id)
            actuals = ctx.actuals()
            result = AgentResult(
                invocation_id=call.invocation_id,
                status="error",
                output="",
                actuals=actuals,
                error=f"handler exception: {type(e).__name__}: {e}",
            )

        # 7. Emit terminal event so the thread / chronicle have a
        #    clean closing record.
        ctx.emit(
            "conclusion",
            content=result.output[:300] or f"status={result.status}",
            status=result.status,
        )
        close_stream(call.invocation_id)

        # 8. Sanity: ensure invocation_id matches even if handler set it
        #    to something else (defensive).
        if result.invocation_id != call.invocation_id:
            result = result.model_copy(update={"invocation_id": call.invocation_id})

        return result

    # -- helpers -----------------------------------------------------------

    def _load_invocation_history(self, root_invocation_id: str) -> list[gates.HistoricalCall]:
        """Pull this tree's previous calls from chronicle for repeat/convergence."""
        # We mirror to a per-tree Redis list so the chronicle stream
        # can be consumed by external observers without coupling. A
        # background tail process would be cleaner; for now the
        # client also writes to this list so reads are cheap.
        key = f"orchestration:tree_calls:{root_invocation_id}"
        raw = self._redis.lrange(key, 0, -1) or []
        out: list[gates.HistoricalCall] = []
        for entry in raw:
            text = entry.decode() if isinstance(entry, bytes) else entry
            parts = text.split("\t", 3)
            if len(parts) != 4:
                continue
            out.append(
                gates.HistoricalCall(
                    invocation_id=parts[0],
                    caller=parts[1],
                    target=parts[2],
                    query=parts[3],
                )
            )
        return out

    # -- optional FastAPI integration --------------------------------------

    def router(self):
        """Build a FastAPI APIRouter exposing ``POST /invoke``.

        Lazy-imports fastapi so the package's base import doesn't
        require it. Install with the ``[server]`` extra.
        """
        from fastapi import APIRouter, HTTPException

        router = APIRouter()

        @router.post("/invoke")
        def invoke(payload: dict):
            try:
                call = AgentCall.model_validate(payload)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"invalid AgentCall: {e}")
            result = self.handle(call)
            return result.model_dump(mode="json")

        @router.get("/healthz")
        def healthz():
            return {"agent": self.name, "ok": True}

        return router
