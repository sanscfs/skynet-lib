"""Skynet Orchestration -- one protocol for every sub-agent.

The library every Skynet sub-agent (sre / music / movies / ...)
imports to talk to every other one. Closes the door on:

- the chat misfire pattern where the main agent has every tool in
  its catalog and fires a kubectl on a philosophical prompt --
  domain tools are now hidden behind agent-as-tool calls;
- ad-hoc shell dispatch from JSON-plan actions, which bypassed
  tool gating entirely;
- runaway loops and cycles, because every call passes through
  cycle / repeat / justification / convergence gates;
- token budget blowouts, because a tree shares one allocated pool.

Public surface
~~~~~~~~~~~~~~

Envelopes::

    from skynet_orchestration import (
        AgentCall, AgentResult, AgentEvent,
        WorkEstimate, BudgetGrant, WorkActuals,
        ThreadHandle, PROTOCOL_VERSION,
    )

Caller side::

    from skynet_orchestration import AgentClient
    from skynet_orchestration.client import (
        new_invocation_id, request_budget_extension, emit_event, now,
    )

Server side::

    from skynet_orchestration import AgentServer, HandlerContext

Adaptive components (gates / estimation / calibration)::

    from skynet_orchestration import gates, estimator, calibration, budget

Streaming + chronicle helpers::

    from skynet_orchestration import streaming, chronicle, tokens

Each submodule has its own docstring with the design rationale.
"""

from __future__ import annotations

from . import budget, calibration, chronicle, estimator, gates, streaming, tokens
from .chronicle import configure_chronicle
from .client import AgentClient
from .envelopes import (
    PROTOCOL_VERSION,
    AgentCall,
    AgentEvent,
    AgentResult,
    BudgetGrant,
    CallPurpose,
    EventType,
    ResultStatus,
    ThreadHandle,
    WorkActuals,
    WorkEstimate,
)
from .server import AgentServer, HandlerContext
from .streaming import configure_streaming

__all__ = [
    # envelopes
    "PROTOCOL_VERSION",
    "AgentCall",
    "AgentEvent",
    "AgentResult",
    "BudgetGrant",
    "CallPurpose",
    "EventType",
    "ResultStatus",
    "ThreadHandle",
    "WorkActuals",
    "WorkEstimate",
    # actors
    "AgentClient",
    "AgentServer",
    "HandlerContext",
    # cross-cutting wiring
    "configure_chronicle",
    "configure_streaming",
    # submodules with internal helpers
    "budget",
    "calibration",
    "chronicle",
    "estimator",
    "gates",
    "streaming",
    "tokens",
]
