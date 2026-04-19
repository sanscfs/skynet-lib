"""Exception hierarchy for skynet-impulse.

Keep errors shallow: the library is called from tick loops that prefer to log
and move on. Callers that wrap the engine in retry logic want a common base
class (``ImpulseError``) to catch; specific subclasses exist only when the
caller could reasonably distinguish them.
"""

from __future__ import annotations


class ImpulseError(Exception):
    """Base class for all impulse-engine errors."""


class ConfigError(ImpulseError):
    """Bad ``EngineConfig`` -- unknown drive referenced, negative decay, etc.

    Raised at engine construction; callers should surface this as a boot-time
    failure rather than a soft tick error.
    """


class OptionalDependencyError(ImpulseError):
    """A detector needs an optional extra that was not installed.

    For example, ``PoissonRepeatDetector`` imports ``scipy.stats`` lazily and
    raises this when scipy is missing, so users who only need
    ``CentroidNoveltyDetector`` can ship without scipy.
    """
