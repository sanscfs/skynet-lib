"""Two-speed EMA adaptive half-life calibration (per-observation).

Replaces the fixed ``refractory_cap_ticks`` magic number with a
self-calibrating half-life derived from observed anchor-repeat intervals
on the signal stream. The per-anchor penalty then decays exponentially
with that learned half-life, yielding a continuous cooldown rather than
a hard integer counter.

Design: per-observation EMA, no batching
----------------------------------------

Every anchor-repeat gap feeds both EMAs immediately. There is no rolling
buffer, no median filter, no periodic update schedule. The smoothing
inherent to a low-alpha EMA is the only noise-rejection mechanism and
it is sufficient for our signal rates.

Why two speeds?
~~~~~~~~~~~~~~~

A single EMA has one exposed knob (alpha) that trades stability (low
alpha) against responsiveness (high alpha). The two-speed trick runs
both simultaneously:

* ``h_fast`` (alpha=0.30, effective window ~10 observations) reacts
  quickly to regime shifts in anchor-repeat intervals.
* ``h_slow`` (alpha=0.05, effective window ~50 observations) is a
  stable long-run estimate that anchors the calibration against noise.

The effective half-life blends them by their disagreement:

* When ``h_fast`` and ``h_slow`` agree, the blend = ``h_slow`` -> stable.
* When they disagree (regime change), the blend shifts towards
  ``h_fast`` -> adaptive.

Formally ``weight_fast = min(|h_fast - h_slow| / max(h_slow, 1), 1)``.

Single-outlier transient
~~~~~~~~~~~~~~~~~~~~~~~~

A single extreme observation (e.g. gap=500 in a stream of ~75) briefly
spikes ``h_fast`` while barely moving ``h_slow``; the disagreement
ratio goes near 1.0 and the effective half-life tracks ``h_fast`` for a
few observations until ``h_fast`` reconverges with ``h_slow``. This is
acceptable — the downside of a single transient high half-life is
slightly slower penalty decay for a couple of signal events, not a
missed fire.

Constants are code-level conventions, NOT env vars
--------------------------------------------------

``HALF_LIFE_ALPHA_FAST`` / ``HALF_LIFE_ALPHA_SLOW`` /
``HALF_LIFE_MIN`` / ``HALF_LIFE_MAX`` / ``HALF_LIFE_PRIOR_FRACTION`` /
``PENALTY_NOISE_FLOOR`` are well-grounded convention values. They are
intentionally NOT exposed as env vars on ``EngineConfig``. If you feel
the need to tune them, the right answer is almost always "gather more
data and trust the EMA to adapt" rather than "knob the EMA".
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


# ---- Well-grounded convention constants ----------------------------------

HALF_LIFE_ALPHA_FAST = 0.30
"""EMA smoothing for the fast estimate. Higher = tracks shifts quicker.

0.30 gives an effective window of ~10 observations (1/alpha).
"""

HALF_LIFE_ALPHA_SLOW = 0.05
"""EMA smoothing for the slow estimate. Lower = more stable long-run.

0.05 gives an effective window of ~50 observations (1/alpha).
"""

HALF_LIFE_MIN = 5
"""Safety clamp: effective half-life never drops below this many signals."""

HALF_LIFE_MAX = 500
"""Safety clamp: effective half-life never exceeds this many signals."""

HALF_LIFE_PRIOR_FRACTION = 0.05
"""Prior h_fast/h_slow = baseline_window * this. 0.05 * 672 ≈ 33."""

PENALTY_NOISE_FLOOR = 0.1
"""Below this, the penalty is treated as zero (gate allows fire)."""

PENALTY_EVICTION_FLOOR = 0.001
"""Below this, the penalty is evicted entirely to bound memory."""

# Persistence cadence: save calibration state every N signals processed.
# Cheap pod-restart safety without hammering Redis on every event.
PERSIST_EVERY_N_SIGNALS = 10


# ---- State container -----------------------------------------------------


@dataclass
class CalibrationState:
    """In-memory calibration state for one domain.

    Persisted to Redis via :class:`CalibrationPersistence` so a pod
    restart doesn't throw away the learned half-life.
    """

    # Two-speed EMA estimates.
    h_fast: float = 33.0
    h_slow: float = 33.0

    # Monotonic counter in signal-event units (logical time).
    global_signal_count: int = 0

    # Last global_signal_count at which we saw each anchor, used to
    # compute the gap on the NEXT sighting.
    anchor_last_seen: dict[str, int] = field(default_factory=dict)

    # Continuous per-anchor penalty [0, 1]. Replaces the ticks-remaining
    # integer countdown. A penalty of 1.0 = "just fired, fully blocked";
    # below ``PENALTY_NOISE_FLOOR`` = "free to fire".
    penalties: dict[str, float] = field(default_factory=dict)

    # Total number of anchor-repeat observations (gaps) fed into the
    # EMAs. Diagnostic-only; not used by the math. Persisted to keep
    # /vibe/status honest across restarts.
    observations: int = 0


# ---- Calibrator ----------------------------------------------------------


class HalfLifeCalibrator:
    """Pure calibration math. Redis I/O lives in the persistence layer.

    Isolation rationale: the math here is trivial to unit-test without
    Redis, and the engine's hot path calls only a handful of these
    methods per signal. Keeping I/O separate means tests can just
    construct a calibrator and hit it with observations.
    """

    def __init__(
        self,
        *,
        prior: float = 33.0,
        alpha_fast: float = HALF_LIFE_ALPHA_FAST,
        alpha_slow: float = HALF_LIFE_ALPHA_SLOW,
        min_half_life: float = HALF_LIFE_MIN,
        max_half_life: float = HALF_LIFE_MAX,
        state: Optional[CalibrationState] = None,
    ):
        self._alpha_fast = alpha_fast
        self._alpha_slow = alpha_slow
        self._min_hl = min_half_life
        self._max_hl = max_half_life
        if state is None:
            state = CalibrationState(h_fast=prior, h_slow=prior)
        self._state = state

    # ---- Accessors -------------------------------------------------------

    @property
    def state(self) -> CalibrationState:
        return self._state

    @property
    def h_fast(self) -> float:
        return self._state.h_fast

    @property
    def h_slow(self) -> float:
        return self._state.h_slow

    @property
    def global_signal_count(self) -> int:
        return self._state.global_signal_count

    @property
    def observations(self) -> int:
        return self._state.observations

    # ---- Core operations -------------------------------------------------

    def observe_signal(self, anchor: Optional[str]) -> Optional[int]:
        """Record a signal event and — if the anchor was seen before —
        feed the gap (in signals) since the last sighting into both EMAs.

        Returns the observed gap (for diagnostics/tests) or ``None`` when
        no gap was observed (first sighting of the anchor, or no anchor).

        Signals without an anchor (``anchor is None``) still increment
        ``global_signal_count`` — the counter is the unit for all
        logical-time bookkeeping, not just per-anchor gaps.
        """
        s = self._state
        s.global_signal_count += 1
        if not anchor:
            return None
        last = s.anchor_last_seen.get(anchor)
        gap: Optional[int] = None
        if last is not None:
            raw_gap = s.global_signal_count - last
            # Non-positive gaps would indicate clock rollback / state
            # corruption; swallow them so the EMAs can't be poisoned.
            if raw_gap > 0:
                gap = raw_gap
                self._apply_observation(raw_gap)
        s.anchor_last_seen[anchor] = s.global_signal_count
        return gap

    def _apply_observation(self, gap: int) -> None:
        """Feed one gap observation into both EMAs immediately."""
        s = self._state
        obs = float(gap)
        s.h_fast = self._alpha_fast * obs + (1.0 - self._alpha_fast) * s.h_fast
        s.h_slow = self._alpha_slow * obs + (1.0 - self._alpha_slow) * s.h_slow
        s.observations += 1

    def effective_half_life(self) -> float:
        """Blend of fast + slow, clamped to [HALF_LIFE_MIN, HALF_LIFE_MAX].

        Weight on h_fast = disagreement ratio capped at 1.0. See the
        module docstring for the trust-region reasoning.
        """
        s = self._state
        disagreement = abs(s.h_fast - s.h_slow) / max(s.h_slow, 1.0)
        weight_fast = min(disagreement, 1.0)
        blend = weight_fast * s.h_fast + (1.0 - weight_fast) * s.h_slow
        return max(self._min_hl, min(self._max_hl, blend))

    def disagreement(self) -> float:
        s = self._state
        return abs(s.h_fast - s.h_slow) / max(s.h_slow, 1.0)

    # ---- Penalty operations ---------------------------------------------

    def decay_penalties(self) -> None:
        """Exponentially decay every active penalty by one signal-event step.

        Uses the current effective half-life so regime changes propagate
        automatically into the cooldown behavior. Penalties below
        ``PENALTY_EVICTION_FLOOR`` are evicted to keep the hash bounded.
        """
        s = self._state
        hl = self.effective_half_life()
        # Decay factor per signal-event: 2^(-1/hl). Equivalent to
        # exp(-ln(2)/hl); both forms give the same number.
        decay_factor = math.exp(-math.log(2.0) / hl)
        for anchor in list(s.penalties):
            s.penalties[anchor] *= decay_factor
            if s.penalties[anchor] < PENALTY_EVICTION_FLOOR:
                del s.penalties[anchor]

    def is_under_penalty(self, anchor: Optional[str]) -> bool:
        """True when the anchor's current penalty is above the noise floor."""
        if not anchor:
            return False
        return self._state.penalties.get(anchor, 0.0) >= PENALTY_NOISE_FLOOR

    def penalty_for(self, anchor: Optional[str]) -> float:
        if not anchor:
            return 0.0
        return self._state.penalties.get(anchor, 0.0)

    def assign_full_penalty(self, anchor: str) -> None:
        """Called after a successful fire: anchor is now fully blocked."""
        self._state.penalties[anchor] = 1.0

    # ---- Diagnostics -----------------------------------------------------

    def diagnostics(self) -> dict:
        """Snapshot of calibration state for ``engine.state()``."""
        s = self._state
        hl = self.effective_half_life()
        return {
            "h_fast": round(s.h_fast, 2),
            "h_slow": round(s.h_slow, 2),
            "effective": round(hl, 2),
            "disagreement": round(self.disagreement(), 3),
            "observations": s.observations,
            "global_signal_count": s.global_signal_count,
        }

    def human_status(self) -> str:
        """One-line human-readable summary for /vibe/status and bots."""
        s = self._state
        hl = self.effective_half_life()
        return (
            f"half-life: {hl:.0f} signals "
            f"(fast={s.h_fast:.0f}, slow={s.h_slow:.0f}, "
            f"blend={int(round(self.disagreement() * 100))}%)"
        )


# ---- Persistence layer (Redis) ------------------------------------------


class CalibrationPersistence:
    """Redis save/load for :class:`CalibrationState`.

    Schema (Redis key suffixes, under the domain's ``key_prefix``):

    * ``:calibration`` — hash with scalar EMA state + counters.
    * ``:last_seen``   — hash mapping anchor -> global_signal_count.
    * ``:penalties``   — hash mapping anchor -> current penalty float.

    Three keys total, all cheap. Loading on start + saving every
    ``PERSIST_EVERY_N_SIGNALS`` signals keeps pod-restart loss bounded.

    Persistence failures are logged and swallowed — they must NEVER
    block signal processing.
    """

    def __init__(self, prefix: str):
        self._prefix = prefix

    # ---- Key helpers -----------------------------------------------------

    @property
    def calibration_key(self) -> str:
        return f"skynet:impulse:calibration:{self._domain()}"

    @property
    def last_seen_key(self) -> str:
        return f"skynet:impulse:last_seen:{self._domain()}"

    @property
    def penalties_key(self) -> str:
        return f"skynet:impulse:penalties:{self._domain()}"

    def _domain(self) -> str:
        """Extract the domain from the prefix.

        Prefix is ``skynet:impulses:{domain}``; we want just the domain
        piece so our key namespace reads as
        ``skynet:impulse:calibration:{domain}`` per the spec.
        """
        if ":" in self._prefix:
            return self._prefix.rsplit(":", 1)[-1]
        return self._prefix

    # ---- Persistence -----------------------------------------------------

    def load(self, redis_client, prior: float) -> CalibrationState:
        """Rehydrate calibration state from Redis; fall back to defaults."""
        state = CalibrationState(h_fast=prior, h_slow=prior)
        try:
            hashed = redis_client.hgetall(self.calibration_key) or {}
            hashed = _decode_hash(hashed)
            if "h_fast" in hashed:
                state.h_fast = float(hashed.get("h_fast", prior))
            if "h_slow" in hashed:
                state.h_slow = float(hashed.get("h_slow", prior))
            if "global_signal_count" in hashed:
                state.global_signal_count = int(float(hashed["global_signal_count"]))
            if "observations" in hashed:
                state.observations = int(float(hashed["observations"]))

            last_seen = redis_client.hgetall(self.last_seen_key) or {}
            last_seen = _decode_hash(last_seen)
            for anchor, val in last_seen.items():
                try:
                    state.anchor_last_seen[anchor] = int(float(val))
                except (TypeError, ValueError):
                    continue

            penalties = redis_client.hgetall(self.penalties_key) or {}
            penalties = _decode_hash(penalties)
            for anchor, val in penalties.items():
                try:
                    state.penalties[anchor] = float(val)
                except (TypeError, ValueError):
                    continue
        except Exception as e:  # noqa: BLE001
            log.debug("calibration load failed (%s); using defaults", e)
        return state

    def save(self, redis_client, state: CalibrationState) -> None:
        """Persist scalars + hashes. Errors are logged and swallowed."""
        try:
            redis_client.hset(
                self.calibration_key,
                mapping={
                    "h_fast": f"{state.h_fast:.6f}",
                    "h_slow": f"{state.h_slow:.6f}",
                    "global_signal_count": str(state.global_signal_count),
                    "observations": str(state.observations),
                },
            )
        except Exception as e:  # noqa: BLE001
            log.debug("calibration save (scalar) failed: %s", e)

        # Rewrite the last_seen hash wholesale: bounded by the number of
        # distinct anchors (typically small). Prefer per-anchor hset so
        # the API works for both real redis and the FakeRedis test shim.
        try:
            for anchor, gsc in state.anchor_last_seen.items():
                redis_client.hset(self.last_seen_key, anchor, str(gsc))
        except Exception as e:  # noqa: BLE001
            log.debug("calibration save (last_seen) failed: %s", e)

        # Penalties: wholesale rewrite so evicted anchors also disappear
        # from Redis. hdel stale keys, then hset the active map.
        try:
            existing = redis_client.hgetall(self.penalties_key) or {}
            existing = _decode_hash(existing)
            stale = [a for a in existing if a not in state.penalties]
            if stale:
                redis_client.hdel(self.penalties_key, *stale)
            if state.penalties:
                redis_client.hset(
                    self.penalties_key,
                    mapping={a: f"{v:.6f}" for a, v in state.penalties.items()},
                )
        except Exception as e:  # noqa: BLE001
            log.debug("calibration save (penalties) failed: %s", e)


def _decode_hash(raw: dict) -> dict:
    """Normalise a redis-py hash return value to {str: str}."""
    if not raw:
        return {}
    first_key = next(iter(raw))
    if isinstance(first_key, bytes):
        return {k.decode(): (v.decode() if isinstance(v, bytes) else v) for k, v in raw.items()}
    return dict(raw)


__all__ = [
    "HALF_LIFE_ALPHA_FAST",
    "HALF_LIFE_ALPHA_SLOW",
    "HALF_LIFE_MIN",
    "HALF_LIFE_MAX",
    "HALF_LIFE_PRIOR_FRACTION",
    "PENALTY_NOISE_FLOOR",
    "PENALTY_EVICTION_FLOOR",
    "PERSIST_EVERY_N_SIGNALS",
    "CalibrationState",
    "HalfLifeCalibrator",
    "CalibrationPersistence",
]
