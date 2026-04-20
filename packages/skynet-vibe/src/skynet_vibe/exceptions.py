"""Exception types for the vibe signal system."""

from __future__ import annotations


class VibeError(Exception):
    """Base class for all skynet-vibe errors."""


class PrototypeNotFoundError(VibeError):
    """Raised when a domain prototype is requested but not registered."""


class SignalNotFoundError(VibeError):
    """Raised when a signal lookup by id misses."""


class EmbeddingError(VibeError):
    """Raised when the injected embedder fails to return a usable vector."""


class PrototypeWarmingUpError(VibeError):
    """Prototypes still loading in background -- caller should fallback."""
