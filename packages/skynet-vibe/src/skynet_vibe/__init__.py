"""Skynet Vibe -- universal gradient signal system for preference capture and retrieval.

Core idea: signals do NOT carry a domain label. Domain membership is computed at
retrieval time as cosine similarity to domain prototype centroids. One signal can
legitimately contribute to many domains with naturally different weights.

Public entry point: :class:`VibeEngine`.
"""

from __future__ import annotations

from skynet_vibe.affinity import SOURCE_TRUST, cosine, signal_weight
from skynet_vibe.emoji import EMOJI_TO_PHRASE, embed_emoji
from skynet_vibe.engine import SuggestResult, VibeEngine
from skynet_vibe.exceptions import (
    EmbeddingError,
    PrototypeNotFoundError,
    PrototypeWarmingUpError,
    SignalNotFoundError,
    VibeError,
)
from skynet_vibe.explain import describe_current_vibe, explain_signal
from skynet_vibe.prototypes import DomainPrototype, PrototypeRegistry
from skynet_vibe.signals import FacetVectors, Source, VibeSignal
from skynet_vibe.store import VibeStore

__all__ = [
    "VibeSignal",
    "FacetVectors",
    "Source",
    "DomainPrototype",
    "PrototypeRegistry",
    "VibeStore",
    "VibeEngine",
    "SuggestResult",
    "cosine",
    "signal_weight",
    "SOURCE_TRUST",
    "EMOJI_TO_PHRASE",
    "embed_emoji",
    "describe_current_vibe",
    "explain_signal",
    "VibeError",
    "PrototypeNotFoundError",
    "PrototypeWarmingUpError",
    "SignalNotFoundError",
    "EmbeddingError",
]

__version__ = "2026.4.20.1"
