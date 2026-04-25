"""Caller side of the protocol.

An ``AgentClient`` is what main / music / sre etc. use to invoke
*another* agent. It does the boring plumbing:

- generates ``invocation_id`` and seeds the tree on first call
  (when there's no parent), consults :mod:`budget` to allocate from
  the root pool;
- mints the HMAC token via :mod:`tokens`;
- POSTs the AgentCall to the target's ``/invoke`` endpoint;
- receives the AgentResult and refunds unused budget;
- streams progress emits to the thread on the caller's behalf when
  the handler doesn't run inside this process (cross-service case).

This file is intentionally transport-agnostic where it can be:
the HTTP layer is httpx but a memory-only test transport can be
swapped in by overriding :meth:`AgentClient._dispatch`.
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Callable, Optional

import httpx

from . import budget, tokens
from .calibration import record_outcome, record_threshold_sample
from .envelopes import (
    AgentCall,
    AgentEvent,
    AgentResult,
    BudgetGrant,
    CallPurpose,
    ThreadHandle,
    WorkActuals,
    WorkEstimate,
)
from .estimator import grant_from_estimate

log = logging.getLogger("skynet_orchestration.client")


def new_invocation_id() -> str:
    """Short opaque id; chronicle entries are keyed by this."""
    return secrets.token_hex(8)


class AgentClient:
    """Per-process orchestration client.

    Instantiated once per service. Holds the Redis client (for
    budget + calibration), the registry of target endpoints, and
    the caller's own name.
    """

    def __init__(
        self,
        *,
        caller_name: str,
        redis_client,
        target_registry: dict[str, str],
        http_timeout_seconds: float = 120.0,
    ):
        self._caller = caller_name
        self._redis = redis_client
        self._registry = dict(target_registry)
        self._http_timeout = http_timeout_seconds
        # caller-controlled hooks; tests / agents may override
        self.estimate_query: Callable[[str, str], WorkEstimate] = self._default_estimate

    # -- public API ---------------------------------------------------------

    def invoke(
        self,
        target: str,
        query: str,
        *,
        purpose: CallPurpose,
        thread: ThreadHandle,
        reason: Optional[str] = None,
        parent_invocation_id: Optional[str] = None,
        root_invocation_id: Optional[str] = None,
        call_chain: Optional[list[str]] = None,
        initiator_user: Optional[str] = None,
    ) -> AgentResult:
        """Dispatch an AgentCall to ``target`` and return its AgentResult."""
        invocation_id = new_invocation_id()
        root = root_invocation_id or invocation_id
        chain = list(call_chain or [])
        # The caller appends its OWN name; the receiving server appends
        # the target on entry. That way both ends contribute to the
        # chain and a malicious caller can't fake who they are.
        if self._caller not in chain:
            chain.append(self._caller)

        estimate = self.estimate_query(target, query)
        grant = grant_from_estimate(estimate)

        if root_invocation_id is None:
            # We're the tree root. Initialise the budget pool with
            # 4× the estimate, leaving headroom for the rest of the
            # tree (downstream calls will reserve from this pool).
            root_cap = budget.RootBudget(
                tokens=grant.tokens * 4,
                tool_calls=grant.tool_calls * 4,
                time_ms=grant.time_ms * 4,
            )
            budget.init_root(self._redis, root, root_cap)

        if not budget.try_reserve(self._redis, root, grant):
            log.warning(
                "budget exhausted before dispatch: target=%s root=%s grant=%s",
                target,
                root,
                grant,
            )
            return AgentResult(
                invocation_id=invocation_id,
                status="budget_exhausted",
                output="",
                actuals=WorkActuals(tokens_used=0, tool_calls_made=0, time_ms=0),
                error="root budget exhausted before reservation",
            )

        token = tokens.mint(invocation_id=invocation_id, caller=self._caller)
        call = AgentCall(
            invocation_id=invocation_id,
            root_invocation_id=root,
            parent_invocation_id=parent_invocation_id,
            call_chain=chain,
            target=target,
            caller=self._caller,
            query=query,
            purpose=purpose,
            reason=reason,
            thread=thread,
            estimate=estimate,
            granted=grant,
            caller_token=token,
            initiator_user=initiator_user,
        )

        try:
            result = self._dispatch(target, call)
        except Exception as e:  # noqa: BLE001
            log.warning("dispatch failed for target=%s: %s", target, e)
            # full grant goes back since the call never started
            budget.refund(
                self._redis,
                root,
                actuals=WorkActuals(tokens_used=0, tool_calls_made=0, time_ms=0),
                granted=grant,
            )
            return AgentResult(
                invocation_id=invocation_id,
                status="error",
                output="",
                actuals=WorkActuals(tokens_used=0, tool_calls_made=0, time_ms=0),
                error=f"dispatch failed: {e}",
            )

        # Refund unused budget so siblings/parents can reuse it.
        budget.refund(self._redis, root, actuals=result.actuals, granted=grant)

        # Calibration write-back: only successful invocations contribute
        # to the history corpus, otherwise a runaway loop would poison
        # the median.
        if result.status == "ok":
            try:
                record_outcome(self._redis, target, invocation_id, query, result.actuals)
                # Also record the estimate accuracy ratio so the agent's
                # self-estimator can be audited later.
                if estimate.tokens_needed > 0:
                    ratio = result.actuals.tokens_used / estimate.tokens_needed
                    record_threshold_sample(
                        self._redis,
                        caller=self._caller,
                        target=target,
                        metric="estimate_accuracy_tokens",
                        value=ratio,
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("calibration write-back failed: %s", e)

        return result

    # -- transport ---------------------------------------------------------

    def _dispatch(self, target: str, call: AgentCall) -> AgentResult:
        """POST the call to target's /invoke endpoint."""
        endpoint = self._registry.get(target)
        if not endpoint:
            raise RuntimeError(f"no endpoint registered for target={target}")
        with httpx.Client(timeout=self._http_timeout) as http:
            resp = http.post(
                endpoint.rstrip("/") + "/invoke",
                json=call.model_dump(mode="json"),
                headers={"x-skynet-orchestration-version": call.protocol_version},
            )
            resp.raise_for_status()
            payload = resp.json()
        return AgentResult.model_validate(payload)

    # -- defaults ----------------------------------------------------------

    def _default_estimate(self, target: str, query: str) -> WorkEstimate:
        """Conservative structural fallback for callers that don't plug
        in a custom estimator. Most agents will replace this with a
        :class:`CompositeEstimator` against their own history."""
        from .estimator import structural_fallback

        return structural_fallback(query)


# ---------------------------------------------------------------------------
# Per-call helpers (used inside a server handler)
# ---------------------------------------------------------------------------


def request_budget_extension(
    redis_client,
    *,
    root_invocation_id: str,
    additional: BudgetGrant,
    reason: str,
) -> bool:
    """Mid-flight extension request from inside a handler.

    Returns True if granted. ``reason`` is for the chronicle entry,
    not for gating -- the budget check is the actual gate, and
    pool-level ``max_extensions_per_tree`` caps how often it can fire.
    """
    log.info(
        "budget extension requested: root=%s additional=%s reason=%s",
        root_invocation_id,
        additional,
        reason,
    )
    return budget.grant_extension(redis_client, root_invocation_id, additional)


def emit_event(
    handle: ThreadHandle,
    event: AgentEvent,
    *,
    service_name: str,
    root_invocation_id: str,
    target: str,
    caller: Optional[str] = None,
) -> None:
    """Single call site for streaming + chronicling an event.

    Most handlers will bind these args via functools.partial at the
    top of the function and then just call ``emit("found", "...")``.
    """
    from .chronicle import emit_event as chronicle_emit
    from .streaming import emit_to_thread

    chronicle_emit(event, root_invocation_id=root_invocation_id, target=target, caller=caller)
    emit_to_thread(handle, event, service_name=service_name)


def now() -> float:
    """Monotonic-ish wall-clock for AgentEvent.ts."""
    return time.time()
