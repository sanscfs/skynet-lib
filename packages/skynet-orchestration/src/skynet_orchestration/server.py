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
from . import gates, metrics, tokens
from .calibration import record_threshold_sample, threshold_snapshot
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

# How many gate checks to absorb before logging a one-line summary of
# the live adaptive thresholds. Pure observability — no behaviour
# depends on this knob.
_ADAPTIVE_LOG_EVERY = 50

# Minimum samples on the relevant accept/reject bucket before the
# server trusts the percentile snapshot. Below this the static
# constructor default is used (cold-start fallback).
_ADAPTIVE_MIN_SAMPLES = 20


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
        # Cold-start fallbacks: used until the calibration corpus has
        # >= _ADAPTIVE_MIN_SAMPLES on the relevant accept/reject bucket.
        self._repeat_threshold = repeat_threshold
        self._just_target_thr = justification_target_threshold
        self._just_state_thr = justification_state_threshold
        # Per-(caller, target, metric) counter, in-process only — rate-
        # limits the "adaptive threshold for X: static→derived" INFO
        # log. Best-effort: replicas don't share this so each emits
        # their own log every N checks, which is fine for observability.
        self._adaptive_log_counter: dict[tuple[str, str, str], int] = {}

    # -- main entry point --------------------------------------------------

    def handle(self, call: AgentCall) -> AgentResult:
        """Verify, gate, dispatch, finalize. Synchronous to keep the
        handler's mental model simple."""

        _started = time.time()

        def _finalize(result: AgentResult) -> AgentResult:
            """Stamp prometheus metrics + return.

            We funnel every exit through here so the metric counters
            never miss a code path (token reject, gate reject, handler
            crash, normal return). Metrics calls are no-ops when
            prometheus-client isn't installed.
            """
            try:
                metrics.record_invocation(
                    caller=call.caller,
                    target=call.target,
                    purpose=str(call.purpose) if call.purpose else "unknown",
                    status=str(result.status) if result.status else "unknown",
                    duration_seconds=time.time() - _started,
                )
            except Exception:  # noqa: BLE001
                pass
            if result.rejected_by_gate:
                try:
                    metrics.record_rejection(
                        caller=call.caller,
                        target=call.target,
                        gate=str(result.rejected_by_gate),
                    )
                except Exception:  # noqa: BLE001
                    pass
            return result

        # 1. Token check -- proves the call wasn't fabricated client-side.
        try:
            tokens.verify(
                call.caller_token,
                invocation_id=call.invocation_id,
                caller=call.caller,
            )
        except tokens.TokenError as e:
            return _finalize(
                AgentResult(
                    invocation_id=call.invocation_id,
                    status="rejected",
                    output="",
                    actuals=WorkActuals(tokens_used=0, tool_calls_made=0, time_ms=0),
                    error=f"token verification failed: {e}",
                    rejected_by_gate="token",
                )
            )

        # 2. Run all four gates against the INCOMING chain (i.e. without
        #    our own name yet). cycle check would otherwise trivially
        #    fire on every call because we'd be matching ourselves.
        #
        #    Adaptive thresholds: for cosine-based gates we look up the
        #    historical distribution per (caller, target) and derive
        #    today's threshold from it. The static constructor default
        #    is used until the bucket has enough samples.
        history = self._load_invocation_history(call.root_invocation_id)

        repeat_thr = self._derive_repeat_threshold(call.caller, call.target)
        just_target_thr = self._derive_justification_threshold(
            call.caller, call.target, "justification_target_cosine", self._just_target_thr
        )
        just_state_thr = self._derive_justification_threshold(
            call.caller, call.target, "justification_state_cosine", self._just_state_thr
        )

        # Sample-recording callback: every cosine the gates compute
        # gets pushed into the calibration corpus as
        # ``<metric>.accept`` or ``<metric>.reject`` so future calls
        # can use the percentile snapshot to tighten / loosen.
        def _record(metric: str, value: float, *, accepted: bool) -> None:
            suffix = "accept" if accepted else "reject"
            try:
                record_threshold_sample(
                    self._redis,
                    caller=call.caller,
                    target=call.target,
                    metric=f"{metric}.{suffix}",
                    value=value,
                )
            except Exception as e:  # noqa: BLE001
                # Calibration write-back must never break a request.
                log.warning("threshold sample write failed: %s", e)

        rejection = (
            gates.check_cycle(call)
            or gates.check_repeat(
                call,
                history=history,
                cosine_fn=self._sim,
                threshold=repeat_thr,
                record_sample=_record,
            )
            or gates.check_justification(
                call,
                target_description=self.target_description,
                caller_state=self._caller_state_fn() if self._caller_state_fn else None,
                cosine_fn=self._sim,
                target_threshold=just_target_thr,
                state_threshold=just_state_thr,
                record_sample=_record,
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
            return _finalize(
                AgentResult(
                    invocation_id=call.invocation_id,
                    status="rejected",
                    output="",
                    actuals=WorkActuals(tokens_used=0, tool_calls_made=0, time_ms=0),
                    error=rejection.reason,
                    rejected_by_gate=rejection.gate,
                )
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

        return _finalize(result)

    # -- adaptive threshold derivation -------------------------------------

    def _derive_repeat_threshold(self, caller: str, target: str) -> float:
        """Derive the repeat-gate threshold from the *reject* bucket.

        Logic: ``repeat_cosine.reject`` is the distribution of
        similarities the gate has historically blocked. We hold today's
        call to the p75 of that distribution, i.e. anything above the
        historical p75-of-blocked-similarities continues to be blocked.
        Falls back to the constructor default while the reject bucket
        is empty / sparse — the cold start is conservative.
        """
        snap = self._safe_snapshot(caller, target, "repeat_cosine.reject")
        if snap is None:
            return self._repeat_threshold
        derived = snap.p75
        self._maybe_log_adaptive(caller, target, "repeat_cosine", self._repeat_threshold, derived, snap.sample_size)
        return derived

    def _derive_justification_threshold(self, caller: str, target: str, base_metric: str, default: float) -> float:
        """Derive a justification threshold from the *accept* bucket.

        Logic: ``<metric>.accept`` is the distribution of cosines that
        previously passed the gate. We use p25 — the bottom of the
        historically-accepted range — so anything as relevant as the
        weakest historical accept is admissible. Falls back to the
        constructor default until the accept bucket is populated.
        """
        snap = self._safe_snapshot(caller, target, f"{base_metric}.accept")
        if snap is None:
            return default
        derived = snap.p25
        self._maybe_log_adaptive(caller, target, base_metric, default, derived, snap.sample_size)
        return derived

    def _safe_snapshot(self, caller: str, target: str, metric: str):
        """Wrap ``threshold_snapshot`` so a Redis hiccup never breaks
        gate evaluation — the static fallback handles it transparently."""
        try:
            return threshold_snapshot(
                self._redis,
                caller=caller,
                target=target,
                metric=metric,
                min_samples=_ADAPTIVE_MIN_SAMPLES,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("threshold snapshot read failed for %s/%s/%s: %s", caller, target, metric, e)
            return None

    def _maybe_log_adaptive(
        self,
        caller: str,
        target: str,
        metric: str,
        static_value: float,
        derived_value: float,
        sample_size: int,
    ) -> None:
        """Emit a one-line INFO every _ADAPTIVE_LOG_EVERY checks per pair+metric."""
        key = (caller, target, metric)
        n = self._adaptive_log_counter.get(key, 0) + 1
        self._adaptive_log_counter[key] = n
        if n % _ADAPTIVE_LOG_EVERY == 1:
            # log on the 1st, 51st, 101st, ... so a fresh process
            # surfaces the current state immediately.
            log.info(
                "adaptive threshold for %s→%s metric=%s: %.3f→%.3f (sample_size=%d)",
                caller,
                target,
                metric,
                static_value,
                derived_value,
                sample_size,
            )

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

        # /metrics: prometheus text exposition. The function is a
        # graceful no-op when prometheus-client isn't installed —
        # callers still get a 200 with a "# not installed" comment so
        # scraping doesn't 500. Mounted unconditionally so a single
        # ServiceMonitor selector works across services regardless of
        # whether they happen to have the [server] extra at runtime.
        from fastapi.responses import Response  # local import: optional dep

        @router.get("/metrics")
        def metrics_endpoint() -> Response:
            body, content_type = metrics.latest_text()
            return Response(content=body, media_type=content_type)

        return router
