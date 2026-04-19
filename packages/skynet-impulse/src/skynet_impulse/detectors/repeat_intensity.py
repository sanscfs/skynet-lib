"""Poisson repeat-intensity detector.

Detector #2 from the research doc. Given a per-anchor baseline rate
``lambda_baseline`` (events per unit time) and observed counts, emit a
signal when the observed rate is anomalously high -- i.e. the user is
suddenly obsessing over one artist / movie / anchor.

Uses the Poisson tail probability ``P(X >= k | lambda)``: if it's below
``p_threshold``, we declare an anomaly. Scipy is an optional dependency
(``skynet-impulse[scipy]``) -- without it, we fall back to a 2-sigma rule
on the Gaussian approximation ``N(lambda, lambda)``, which is good enough
for large lambda but underestimates the tail when lambda is small.

Callers provide the observed-rate state externally (detection is purely
over "have N events already happened in this window?") rather than the
detector maintaining an internal window: Skynet's consumption streams are
already queryable in Postgres, so letting the caller count is both simpler
and avoids double-counting across detector restarts.
"""

from __future__ import annotations

import math
from typing import Optional

from ..exceptions import OptionalDependencyError
from ..signals import Signal


def _poisson_cdf_tail(k: int, lam: float) -> float:
    """P(X >= k) for Poisson(lam). Tries scipy first, falls back to Gaussian approx.

    Handles the ``lam == 0`` edge case (any observation is "impossible" -> tail=1.0)
    and ``k == 0`` (tail = 1.0 by definition). Clips the approximation into
    [0, 1] so callers can compare to ``p_threshold`` without re-checking.
    """
    if lam <= 0:
        return 1.0 if k > 0 else 1.0
    if k <= 0:
        return 1.0
    try:
        from scipy import stats  # type: ignore[import-not-found]

        # sf(k-1) = P(X >= k)
        return float(stats.poisson.sf(k - 1, lam))
    except ImportError:
        # Gaussian approximation: z-score of k against N(lam, lam).
        z = (k - lam) / math.sqrt(lam)
        # Two-tailed approximation of the upper tail via erfc.
        return 0.5 * math.erfc(z / math.sqrt(2))


class PoissonRepeatDetector:
    """Signal anomaly when per-anchor event rate exceeds its Poisson baseline.

    Parameters
    ----------
    lambda_baseline:
        The steady-state ``events / window`` rate for this anchor. Should be
        an exponentially-smoothed moving average computed by the caller (or
        a Postgres ``AVG`` over the last N windows); the detector itself
        only compares observations against this reference.
    p_threshold:
        Fire if ``P(X >= k | lambda_baseline) < p_threshold``. 0.05 matches
        the research's recommendation; 0.01 is a stricter "really obsessing"
        gate useful for domains with many anchors.
    anchor_prefix:
        Prepended to ``{anchor_name}`` when emitting signals. Conventional:
        ``"music:artist:"``, ``"movies:title:"``.
    min_historical_events:
        When the anchor has fewer than this many lifetime events, the rate
        estimate is noisy -- defer to the novelty detector instead. We return
        empty.
    """

    domain = "repeat_intensity"

    def __init__(
        self,
        lambda_baseline: float,
        *,
        p_threshold: float = 0.05,
        anchor_prefix: str = "",
        min_historical_events: int = 5,
        source: str = "analyzer",
        require_scipy: bool = False,
    ):
        if lambda_baseline < 0:
            raise ValueError("lambda_baseline must be >= 0")
        if not 0.0 < p_threshold < 1.0:
            raise ValueError("p_threshold must be in (0, 1)")
        self._lambda = float(lambda_baseline)
        self._p_threshold = p_threshold
        self._anchor_prefix = anchor_prefix
        self._min_events = min_historical_events
        self._source = source
        if require_scipy:
            # Eager import -- lets callers catch missing-dep errors at boot.
            try:
                from scipy import stats  # noqa: F401
            except ImportError as e:
                raise OptionalDependencyError(
                    "PoissonRepeatDetector(require_scipy=True) needs scipy; install 'skynet-impulse[scipy]'."
                ) from e

    @property
    def anchor_prefix(self) -> str:
        return self._anchor_prefix

    @property
    def lambda_baseline(self) -> float:
        return self._lambda

    def update_baseline(self, new_lambda: float) -> None:
        if new_lambda < 0:
            raise ValueError("new_lambda must be >= 0")
        self._lambda = float(new_lambda)

    async def detect(
        self,
        *,
        anchor_name: str,
        observed_count: int,
        historical_count: int,
        payload: Optional[dict] = None,
        source: Optional[str] = None,
    ) -> list[Signal]:
        """Return 0 or 1 signals given this anchor's counts.

        - ``observed_count``: events attributed to this anchor in the current
          evaluation window.
        - ``historical_count``: lifetime events for this anchor. Gates the
          detector off until ``>= min_historical_events``.
        """
        if historical_count < self._min_events:
            return []
        if observed_count <= 0:
            return []
        p = _poisson_cdf_tail(observed_count, self._lambda)
        if p >= self._p_threshold:
            return []
        salience = max(0.0, min(1.0, 1.0 - p))
        anchor = f"{self._anchor_prefix}{anchor_name}"
        sig = Signal(
            kind="novelty",  # research doc: reuse `novelty` with anchor convention
            source=source or self._source,
            salience=salience,
            anchor=anchor,
            payload={
                "detector": "repeat_intensity",
                "observed": observed_count,
                "lambda": round(self._lambda, 4),
                "p_tail": round(p, 6),
                "anchor_name": anchor_name,
                **(payload or {}),
            },
        )
        return [sig]


__all__ = ["PoissonRepeatDetector"]
