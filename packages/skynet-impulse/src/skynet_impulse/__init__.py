"""Skynet Impulse -- domain-agnostic autonomous-impulse engine.

Generalizes the original ``skynet_agent.modules.impulse`` package into a
reusable library. Same mechanisms (homeostat -> adaptive baseline -> per-
anchor refractory -> cheap LLM gate -> full LLM compose), now parameterised
by domain so music / movies / the main chat agent can all run their own
curiosity loops sharing one signal bus (``skynet_core.impulses``).

Public surface
~~~~~~~~~~~~~~

Engine orchestration::

    from skynet_impulse import (
        ImpulseEngine, EngineConfig, EngineState, TickResult,
        Drive, SignalToDrive, DriveState,
    )

Protocols for caller-injected LLM clients::

    from skynet_impulse import GateClient, ComposeClient

Reference HTTP implementations (optional -- caller can bring their own)::

    from skynet_impulse import DefaultHttpGateClient, DefaultOpenAIComposeClient

Archetype bandit (Thompson-sampling over question shapes)::

    from skynet_impulse import Archetype, ArchetypeBandit, default_archetypes

Detectors (MVP set per research doc)::

    from skynet_impulse.detectors import (
        CentroidNoveltyDetector,
        PoissonRepeatDetector,
        UncertaintySamplingDetector,
    )

Signal primitives (re-exported from ``skynet_core.impulses`` for convenience,
do NOT fork)::

    from skynet_impulse import Signal, SignalKind, STREAM_NAME
"""

from __future__ import annotations

from .archetypes import Archetype, ArchetypeBandit, default_archetypes
from .baseline import AdaptiveBaseline, BaselineConfig
from .compose import (
    ComposeClient,
    DefaultOpenAIComposeClient,
    format_compose_prompt,
)
from .drives import Drive, DriveState, SignalToDrive
from .engine import EngineConfig, EngineState, ImpulseEngine, TickResult
from .exceptions import ConfigError, ImpulseError, OptionalDependencyError
from .gate import (
    DefaultHttpGateClient,
    GateClient,
    format_gate_prompt,
)
from .homeostat import Homeostat
from .signals import (
    DEFAULT_CONSUMER_GROUP,
    DEFAULT_MAXLEN,
    STREAM_NAME,
    Signal,
    SignalKind,
    SignalSource,
    ack_signal,
    ack_signals,
    drain_signals,
    emit_signal,
    emit_signal_async,
    ensure_consumer_group,
)

__version__ = "2026.4.20"

__all__ = [
    "__version__",
    # Engine
    "ImpulseEngine",
    "EngineConfig",
    "EngineState",
    "TickResult",
    # Drives
    "Drive",
    "DriveState",
    "SignalToDrive",
    "Homeostat",
    # Baseline
    "AdaptiveBaseline",
    "BaselineConfig",
    # Gate / Compose
    "GateClient",
    "ComposeClient",
    "DefaultHttpGateClient",
    "DefaultOpenAIComposeClient",
    "format_gate_prompt",
    "format_compose_prompt",
    # Archetypes
    "Archetype",
    "ArchetypeBandit",
    "default_archetypes",
    # Signals (re-export)
    "Signal",
    "SignalKind",
    "SignalSource",
    "STREAM_NAME",
    "DEFAULT_CONSUMER_GROUP",
    "DEFAULT_MAXLEN",
    "emit_signal",
    "emit_signal_async",
    "ensure_consumer_group",
    "drain_signals",
    "ack_signal",
    "ack_signals",
    # Errors
    "ImpulseError",
    "ConfigError",
    "OptionalDependencyError",
]
