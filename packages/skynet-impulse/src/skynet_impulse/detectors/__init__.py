"""Detector implementations -- each emits ``Signal`` batches.

Per §3 of ``docs/domain-curiosity-agents-research.md`` the MVP detector set
is three algorithms that run on the domain's consumption data:

- :class:`CentroidNoveltyDetector` -- "something off-manifold happened"
- :class:`PoissonRepeatDetector` -- "this is being repeated abnormally often"
- :class:`UncertaintySamplingDetector` -- "the model is torn; ask the user"

Each detector is independent, stateless-ish (state is passed in or via
update methods), and emits zero or more ``Signal`` objects that the engine
consumes via the shared Redis stream. None of them speak to the user
directly -- that's the engine's job.
"""

from __future__ import annotations

from .novelty import CentroidNoveltyDetector
from .repeat_intensity import PoissonRepeatDetector
from .uncertainty import UncertaintySamplingDetector

__all__ = [
    "CentroidNoveltyDetector",
    "PoissonRepeatDetector",
    "UncertaintySamplingDetector",
]
