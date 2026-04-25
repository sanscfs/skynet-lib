"""Wire-format envelopes for agent-to-agent calls.

These Pydantic models are the *protocol*. Every sub-agent service
serializes/deserializes through them; nothing else crosses the
boundary. Adding a field means bumping ``PROTOCOL_VERSION`` and
calling out the migration in skynet-orchestration release notes.

Design rules:
- Identifier fields (``invocation_id``, ``root_invocation_id``,
  ``parent_invocation_id``) form a tree -- the root id is shared by
  every node in the tree, the parent id is the immediate caller.
- ``call_chain`` is the agent-name path from root to current node and
  is *appended by the receiving server*, never by the client. That
  closes the spoofing hole where a buggy client could omit itself.
- ``WorkEstimate`` and ``WorkActuals`` use the same shape so the
  calibration loop can diff them directly.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

PROTOCOL_VERSION = "1"

CallPurpose = Literal[
    "user_task",  # main delegating a top-level user request
    "self_recovery",  # sub-agent asking another for help with its own failure
    "delegation",  # sub-agent splitting a task across siblings
    "background",  # impulse-driven autonomous work, no user waiting
]

ResultStatus = Literal[
    "ok",  # task completed normally
    "degraded",  # completed but partial / sub-quality
    "error",  # failure during execution
    "budget_exhausted",  # ran out of allocated budget
    "rejected",  # gate (cycle/repeat/justification) blocked the call
    "escalated",  # handed off to a higher-tier model
]

EventType = Literal[
    "progress",  # generic status update
    "tool_call",  # invoking a tool inside the handler
    "tool_result",  # tool returned
    "found",  # discovered a fact relevant to the task
    "thinking",  # exposed reasoning step (optional)
    "conclusion",  # final summary the handler produced
    "budget_request",  # asking caller for more budget mid-flight
    "escalation_request",  # asking caller for a bigger model
    "warning",  # non-fatal anomaly worth surfacing
]


class ThreadHandle(BaseModel):
    """Where to stream events so the user sees them inline.

    Sub-agents post events to ``room_id``+``thread_root`` using
    their own ``service_mxid`` so attribution is clear.
    """

    room_id: str
    thread_root: str
    service_mxid: Optional[str] = None  # set by AgentServer at receive time


class WorkEstimate(BaseModel):
    """An agent's pre-flight guess at what a task will cost.

    Confidence is in ``[0, 1]``; below 0.5 the caller will tend to
    consult historical k-NN before granting.
    """

    tokens_needed: int = Field(ge=0)
    tool_calls_expected: int = Field(ge=0)
    time_ms: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    complexity: Literal["trivial", "low", "medium", "high", "unknown"] = "unknown"

    def scaled(self, factor: float) -> "WorkEstimate":
        """Return a copy with all numeric fields scaled by ``factor``.

        Used to apply a buffer (typically 1.3–1.7x depending on
        confidence) before turning an estimate into a grant.
        """
        return WorkEstimate(
            tokens_needed=int(self.tokens_needed * factor),
            tool_calls_expected=int(self.tool_calls_expected * factor),
            time_ms=int(self.time_ms * factor),
            confidence=self.confidence,
            complexity=self.complexity,
        )


class BudgetGrant(BaseModel):
    """Concrete resources allocated for one invocation.

    ``extensions_remaining`` lets a handler request more during
    execution, capped at this number.
    """

    tokens: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    time_ms: int = Field(ge=0)
    extensions_remaining: int = Field(default=3, ge=0)


class WorkActuals(BaseModel):
    """Resources actually consumed; written into AgentResult.

    Same shape as WorkEstimate so calibration can compute the
    ratio per dimension directly.
    """

    tokens_used: int = Field(ge=0)
    tool_calls_made: int = Field(ge=0)
    time_ms: int = Field(ge=0)
    extensions_used: int = Field(default=0, ge=0)


class AgentCall(BaseModel):
    """One agent → another invocation request."""

    protocol_version: str = PROTOCOL_VERSION

    # tree topology
    invocation_id: str
    root_invocation_id: str
    parent_invocation_id: Optional[str] = None
    call_chain: list[str] = Field(default_factory=list)

    # task content
    target: str  # agent name being invoked, e.g. "sre"
    caller: str  # agent name making the call, e.g. "main"
    query: str  # the task itself
    purpose: CallPurpose
    reason: Optional[str] = None  # required for lateral calls (purpose != user_task)

    # streaming + attribution
    thread: ThreadHandle

    # resource envelope
    estimate: WorkEstimate  # the caller's expectation
    granted: BudgetGrant  # what caller allocated

    # provenance
    caller_token: str  # HMAC, server verifies
    initiator_user: Optional[str] = None  # MXID of the human that started this tree

    @field_validator("reason")
    @classmethod
    def _reason_required_for_lateral(cls, v: Optional[str], info) -> Optional[str]:
        purpose = info.data.get("purpose")
        if purpose in ("self_recovery", "delegation") and not v:
            raise ValueError(f"reason is required for purpose={purpose} (lateral calls must justify themselves)")
        return v


class AgentEvent(BaseModel):
    """Streamed during execution. Persisted to chronicle + surfaced in thread."""

    invocation_id: str
    type: EventType
    content: str
    metadata: dict = Field(default_factory=dict)
    ts: float  # unix epoch seconds


class AgentResult(BaseModel):
    """Final return value from an invocation."""

    protocol_version: str = PROTOCOL_VERSION

    invocation_id: str
    status: ResultStatus
    output: str  # human-readable summary
    self_check: dict[str, bool] = Field(default_factory=dict)  # health probes the agent ran
    suggested_next: Optional[str] = None  # hint the caller can act on (or ignore)
    actuals: WorkActuals
    error: Optional[str] = None  # non-empty when status in {error, rejected, budget_exhausted}
    rejected_by_gate: Optional[str] = None  # which gate said no, if any
