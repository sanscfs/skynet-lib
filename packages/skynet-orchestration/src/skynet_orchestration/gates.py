"""Adaptive safeguards on every incoming AgentCall.

Four orthogonal checks. Each is its own function so the server can
short-circuit on the first ``GateRejection`` and report which gate
fired. Thresholds are not hardcoded constants; they are read from a
Redis-backed ``AdaptiveThreshold`` per (caller, target) so the
system tunes itself from history.

Why four small gates rather than one big rule:
- ``cycle`` -- topological. Already-on-the-stack target = re-entry.
- ``repeat`` -- semantic. Same target invoked with the same query
  twice in this tree = stuck loop, even if topologically clean.
- ``justification`` -- relevance. The caller's stated ``reason``
  must point at the target's purpose, otherwise it's a wandering
  off-topic call.
- ``convergence`` -- progress. Each step must be more specific
  than the last; an agent that keeps issuing vaguer queries is
  thrashing.

Convention: every gate returns ``None`` to allow the call, or a
:class:`GateRejection` to block it. Callers (the AgentServer)
collapse a rejection into ``status=rejected, rejected_by_gate=<name>``
so the parent invocation can decide what to do.

Self-learning threshold encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cosine-based gates (``check_repeat`` and ``check_justification``)
take an optional ``record_sample`` callback. Whenever they compute a
similarity, they emit one sample with the polarity baked into the
metric *name suffix*::

    <base_metric>.accept   # cosine of an accepted call
    <base_metric>.reject   # cosine of a rejected call

Splitting at the metric-name level (rather than packing polarity
into the value) keeps the snapshot reader trivially symmetric: it
just calls ``threshold_snapshot`` on whichever bucket it wants
percentiles from, without needing to know how the polarity is
encoded. Three base metrics in use:

- ``repeat_cosine``                — emitted by check_repeat
- ``justification_target_cosine``  — emitted by the target check
- ``justification_state_cosine``   — emitted by the state check

The server reads ``<metric>.reject`` p75 for the repeat threshold
(stay above the historical p75 of blocked similarities) and
``<metric>.accept`` p25 for the justification thresholds (admit
the bottom of the historically-accepted distribution). See
``server.py`` for the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

from .envelopes import AgentCall


class RecordSampleFn(Protocol):
    """Signature of the optional sample-recording callback.

    Cosine-based gates call ``record_sample(metric, value, accepted=...)``
    once per similarity they compute, so the calibration corpus sees
    *both* sides of the gate decision (not just the accepts).
    """

    def __call__(self, metric: str, value: float, *, accepted: bool) -> None: ...


@dataclass(frozen=True)
class GateRejection:
    """A gate refused the call. ``gate`` is the rule's short name;
    ``reason`` is operator-grade prose for logs/observability."""

    gate: str
    reason: str


# ---------------------------------------------------------------------------
# Gate 1: cycle (topological)
# ---------------------------------------------------------------------------


def check_cycle(call: AgentCall) -> Optional[GateRejection]:
    """Reject if ``target`` is already in ``call_chain``.

    main → music → main is allowed (main is the orchestrator and a
    legitimate re-entry point). Any *other* re-entry is a cycle.
    """
    if call.target == "main":
        return None
    if call.target in call.call_chain:
        return GateRejection(
            gate="cycle",
            reason=f"target={call.target} already in call_chain={call.call_chain}",
        )
    return None


# ---------------------------------------------------------------------------
# Gate 2: repeat (semantic similarity over history)
# ---------------------------------------------------------------------------


def check_repeat(
    call: AgentCall,
    *,
    history: Iterable["HistoricalCall"],
    cosine_fn,
    threshold: float = 0.85,
    record_sample: Optional[RecordSampleFn] = None,
) -> Optional[GateRejection]:
    """Reject if the same (caller, target) has issued a near-identical
    query already in this tree.

    The caller passes ``history`` as the list of HistoricalCall
    snapshots from this tree's chronicle. ``cosine_fn(a, b) -> float``
    is supplied by the server (typically wraps skynet_embedding). The
    threshold is adaptive in the sense that the *server* may pull it
    from a per-pair AdaptiveThreshold; this function only enforces
    the comparison itself, which is what's testable in isolation.

    When ``record_sample`` is supplied, every computed similarity
    feeds back into the calibration corpus under metric
    ``repeat_cosine`` with ``accepted=`` reflecting the gate's
    decision for that comparison. The recording happens *before* the
    short-circuit return so a hit is still observed.
    """
    for prev in history:
        if prev.caller != call.caller or prev.target != call.target:
            continue
        sim = cosine_fn(prev.query, call.query)
        accepted = sim < threshold
        if record_sample is not None:
            record_sample("repeat_cosine", float(sim), accepted=accepted)
        if not accepted:
            return GateRejection(
                gate="repeat",
                reason=f"prev_invocation_id={prev.invocation_id} cos={sim:.2f} >= {threshold:.2f}",
            )
    return None


@dataclass(frozen=True)
class HistoricalCall:
    """Lightweight record stored in chronicle for repeat-check.

    Server-side the chronicle stream stores more (timestamps,
    actuals, status) but the gate only needs identity + query.
    """

    invocation_id: str
    caller: str
    target: str
    query: str


# ---------------------------------------------------------------------------
# Gate 3: justification (cosine vs target description / caller_state)
# ---------------------------------------------------------------------------


def check_justification(
    call: AgentCall,
    *,
    target_description: str,
    caller_state: Optional[str],
    cosine_fn,
    target_threshold: float = 0.40,
    state_threshold: float = 0.30,
    record_sample: Optional[RecordSampleFn] = None,
) -> Optional[GateRejection]:
    """Reject if ``reason`` doesn't tie the call to the target *or*
    to the caller's own state.

    Two checks, both must pass when applicable:

    1. ``cos(reason, target_description) >= target_threshold``: the
       caller's stated reason has to point at what the target *does*.
       A music agent invoking ``sre_investigate("kubernetes vibes")``
       with reason="i feel weird about my recommendations" should
       fail this -- the reason doesn't describe SRE work.

    2. ``cos(reason, caller_state) >= state_threshold`` -- only
       evaluated when ``purpose == self_recovery`` and a
       ``caller_state`` is supplied. Couples the reason to the
       observed failure of the calling agent so spurious recovery
       calls (caller_state="ok" but reason="postgres broken") fail.

    purpose=user_task does NOT gate on the cosine check (top-level
    main delegations don't need to justify themselves to the user).
    However, when a ``reason`` IS supplied for a user_task call, we
    still COMPUTE both cosines and feed them into ``record_sample``
    as ``accept`` samples. Rationale: the
    ``justification_target_cosine.accept`` corpus is what the server's
    adaptive-threshold logic consults; without user_task feeding it,
    the corpus only ever fills from rare lateral calls (self_recovery
    / delegation) and stays cold-start indefinitely. Recording without
    gating warms the corpus from real top-level traffic so future
    lateral calls have a real distribution to be compared against.

    When ``record_sample`` is supplied, both the target-cosine and
    (when applicable) the state-cosine are recorded under metrics
    ``justification_target_cosine`` and ``justification_state_cosine``.
    On a target-side rejection the state-side is never computed, so
    only the failing comparison ends up in the reject bucket.
    """
    if call.purpose == "user_task":
        # Non-gating sample writes: feed the .accept corpus from
        # top-level traffic without enforcing the cosine threshold.
        # Skip cosine computation entirely when reason is None — there
        # is nothing to compare against and a synthetic sample would
        # poison the distribution.
        if call.reason and record_sample is not None:
            try:
                target_sim = cosine_fn(call.reason, target_description)
                record_sample(
                    "justification_target_cosine",
                    float(target_sim),
                    accepted=True,
                )
                if caller_state:
                    state_sim = cosine_fn(call.reason, caller_state)
                    record_sample(
                        "justification_state_cosine",
                        float(state_sim),
                        accepted=True,
                    )
            except Exception:
                # Sample writes are best-effort; never fail user_task
                # because the calibration corpus had a hiccup.
                pass
        return None
    if not call.reason:
        return GateRejection(
            gate="justification",
            reason=f"missing reason for purpose={call.purpose}",
        )
    target_sim = cosine_fn(call.reason, target_description)
    target_accepted = target_sim >= target_threshold
    if record_sample is not None:
        record_sample("justification_target_cosine", float(target_sim), accepted=target_accepted)
    if not target_accepted:
        return GateRejection(
            gate="justification",
            reason=(f"cos(reason, target_description)={target_sim:.2f} < {target_threshold:.2f}"),
        )
    if call.purpose == "self_recovery" and caller_state:
        state_sim = cosine_fn(call.reason, caller_state)
        state_accepted = state_sim >= state_threshold
        if record_sample is not None:
            record_sample("justification_state_cosine", float(state_sim), accepted=state_accepted)
        if not state_accepted:
            return GateRejection(
                gate="justification",
                reason=(f"cos(reason, caller_state)={state_sim:.2f} < {state_threshold:.2f}"),
            )
    return None


# ---------------------------------------------------------------------------
# Gate 4: convergence (each step more specific than the last)
# ---------------------------------------------------------------------------


def check_convergence(
    call: AgentCall,
    *,
    history: Iterable[HistoricalCall],
    specificity_fn,
    min_progress: float = 0.0,
) -> Optional[GateRejection]:
    """Reject if this call is *less specific* than the previous call
    on the same (caller, target) pair within this tree.

    ``specificity_fn(query) -> float`` is supplied by the server.
    A reasonable default is the count of named entities + token
    length; the server can replace with a learned scorer. The gate
    fires only when there is a previous call to compare against and
    progress is below ``min_progress`` (default 0 = strict
    monotonicity).
    """
    last_spec: Optional[float] = None
    for prev in history:
        if prev.caller == call.caller and prev.target == call.target:
            last_spec = specificity_fn(prev.query)
    if last_spec is None:
        return None
    cur_spec = specificity_fn(call.query)
    if cur_spec - last_spec < min_progress:
        return GateRejection(
            gate="convergence",
            reason=(
                f"specificity didn't grow: prev={last_spec:.2f} cur={cur_spec:.2f} min_progress={min_progress:.2f}"
            ),
        )
    return None
