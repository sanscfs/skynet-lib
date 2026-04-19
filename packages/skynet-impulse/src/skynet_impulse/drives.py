"""Drive + DriveState + SignalToDrive config dataclasses.

A *drive* is a scalar homeostatic variable in [0, 1] that grows in response
to matching signals and decays passively each tick. The agent's 4-drive setup
(boredom / curiosity / concern / need_to_share) is a specific instantiation;
the library treats drives as an arbitrary named set.

``SignalToDrive`` is the wiring between the signal-bus vocabulary (kind
strings like ``"novelty"``) and per-domain drives. One signal may push
several drives simultaneously (see the music domain mapping in the research
doc -- novelty bumps curiosity AND need_to_share).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Drive:
    """Definition of a single homeostatic drive.

    ``decay_rate`` is a geometric multiplier applied each tick when no signals
    push the drive up (``value *= decay_rate``). Use values in ``(0, 1]`` for
    a classic exponential decay; values ``> 1`` are rejected as they would
    blow up the state unboundedly.

    ``growth_per_tick`` is added every tick unconditionally (the boredom
    drive's "time alone makes me bored" mechanic). Most drives leave this at
    0; only the "idleness" drive of a domain uses it.
    """

    name: str
    decay_rate: float = 0.85
    initial: float = 0.0
    growth_per_tick: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Drive.name must be non-empty")
        if not 0.0 < self.decay_rate <= 1.0:
            raise ValueError(
                f"Drive.decay_rate must be in (0, 1]; got {self.decay_rate!r}"
            )
        if self.growth_per_tick < 0:
            raise ValueError(
                f"Drive.growth_per_tick must be >= 0; got {self.growth_per_tick!r}"
            )
        if not 0.0 <= self.initial <= 1.0:
            raise ValueError(
                f"Drive.initial must be in [0, 1]; got {self.initial!r}"
            )


@dataclass(frozen=True)
class SignalToDrive:
    """Wiring: when ``signal_kind`` arrives, add ``salience * multiplier`` to ``drive_name``.

    ``multiplier`` may be negative to dampen a drive on a "resolution" signal
    (e.g. ``resolution`` -> concern with multiplier -0.30). Values are
    otherwise unconstrained; calibration is the caller's problem.

    ``dampen_multiply`` is an alternative wiring that multiplies the drive
    rather than adding to it -- used by the agent's ``spoke`` self-feedback
    to damp ``need_to_share`` after speaking. When set (non-None), it
    overrides the additive behavior and scales the drive by ``dampen_multiply``
    regardless of salience.
    """

    signal_kind: str
    drive_name: str
    multiplier: float = 1.0
    dampen_multiply: float | None = None


@dataclass
class DriveState:
    """Mutable snapshot of all drive values at a single tick.

    Stored in Redis as a flat hash ``{drive_name: value_as_string}`` so humans
    can ``HGETALL`` and eyeball what the agent is feeling. Not thread-safe;
    the engine runs one tick at a time per domain.
    """

    values: dict[str, float] = field(default_factory=dict)

    def clip(self) -> None:
        """Pin all drive values into ``[0, 1]``."""
        for k, v in self.values.items():
            self.values[k] = max(0.0, min(1.0, v))

    def get(self, name: str) -> float:
        return self.values.get(name, 0.0)

    def set(self, name: str, value: float) -> None:
        self.values[name] = value

    def add(self, name: str, delta: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + delta

    def multiply(self, name: str, factor: float) -> None:
        self.values[name] = self.values.get(name, 0.0) * factor

    def dominant(self, exclude: set[str] | None = None) -> tuple[str | None, float]:
        """Return the ``(name, value)`` of the largest drive.

        ``exclude`` lets callers hide drives they consider ineligible for
        triggering (the agent excludes "boredom" when no signals arrived, so
        pure loneliness doesn't ping the user). Returns ``(None, 0.0)`` if
        every drive is excluded or the state is empty.
        """
        exclude = exclude or set()
        candidates = {k: v for k, v in self.values.items() if k not in exclude}
        if not candidates:
            return None, 0.0
        return max(candidates.items(), key=lambda kv: kv[1])

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.values.items()}

    def to_redis_mapping(self) -> dict[str, str]:
        return {k: f"{v:.4f}" for k, v in self.values.items()}

    @classmethod
    def from_redis_mapping(cls, mapping: dict, drives: list[Drive]) -> "DriveState":
        """Rehydrate state from a Redis HGETALL, defaulting missing drives to ``initial``.

        Silent coercion: garbage floats land as the drive's ``initial`` rather
        than raising, because a corrupt tick should not crash the loop.
        """
        values: dict[str, float] = {}
        for drive in drives:
            raw = mapping.get(drive.name) if mapping else None
            try:
                values[drive.name] = float(raw) if raw is not None else drive.initial
            except (TypeError, ValueError):
                values[drive.name] = drive.initial
        return cls(values=values)


__all__ = ["Drive", "SignalToDrive", "DriveState"]
