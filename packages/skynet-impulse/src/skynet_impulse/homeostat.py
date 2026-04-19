"""Homeostat -- N-drive accumulator with decay and passive growth.

Generalized from ``skynet_agent.modules.impulse.homeostat`` which hard-coded
4 named drives (boredom/curiosity/concern/need_to_share). Here we take a
``list[Drive]`` at construction and a ``list[SignalToDrive]`` wiring table,
so each domain can pick its own drive set and signal mappings.

Per-tick flow (owned by the engine, not this class):

    1. ``load_state`` -> ``DriveState`` from Redis
    2. for each incoming Signal: ``apply_signals`` bumps mapped drives
    3. ``apply_decay`` decays all drives that did NOT receive a push this tick
    4. ``state.clip()`` pins everything into [0, 1]
    5. ``save_state`` writes back to Redis

This module is synchronous even when drained signals come from an async
bus, because all it does is CPU-bound arithmetic on a small dict.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from .drives import Drive, DriveState, SignalToDrive
from .signals import Signal

log = logging.getLogger(__name__)


class Homeostat:
    """Stateless helper: turns signal batches + decay rules into drive updates.

    The Homeostat itself holds only configuration (drives, signal mappings,
    Redis key prefix). The runtime state lives in a ``DriveState`` that is
    loaded at the start of each tick and saved at the end -- so the same
    Homeostat instance can be reused across ticks without stale state.
    """

    def __init__(
        self,
        *,
        drives: list[Drive],
        signal_to_drive: list[SignalToDrive],
        state_key: str,
    ):
        if not drives:
            raise ValueError("Homeostat requires at least one drive")
        self._drives = drives
        self._drives_by_name = {d.name: d for d in drives}
        # Validate every wiring entry references a known drive -- catches
        # config typos at construction rather than silently dropping signals.
        for wire in signal_to_drive:
            if wire.drive_name not in self._drives_by_name:
                raise ValueError(
                    f"SignalToDrive references unknown drive {wire.drive_name!r}; "
                    f"known drives: {sorted(self._drives_by_name)}"
                )
        self._wirings = signal_to_drive
        self._state_key = state_key

    @property
    def drives(self) -> list[Drive]:
        return list(self._drives)

    @property
    def state_key(self) -> str:
        return self._state_key

    # ---- Persistence ------------------------------------------------------

    def load_state(self, redis_client) -> DriveState:
        """Load drive values from Redis hash. Missing drives get their ``initial``."""
        raw = redis_client.hgetall(self._state_key) or {}
        # Redis-py returns bytes when decode_responses=False; normalize here so
        # callers don't have to care about the client setting.
        if raw and isinstance(next(iter(raw)), bytes):
            raw = {k.decode(): v.decode() for k, v in raw.items()}
        return DriveState.from_redis_mapping(raw, self._drives)

    def save_state(self, redis_client, state: DriveState) -> None:
        state.clip()
        mapping = state.to_redis_mapping()
        if mapping:
            redis_client.hset(self._state_key, mapping=mapping)

    # ---- Tick operations --------------------------------------------------

    def apply_signals(
        self,
        state: DriveState,
        signals: Iterable[Signal],
    ) -> tuple[Optional[str], set[str]]:
        """Mutate ``state`` by the wiring table for each incoming signal.

        Returns
        -------
        top_anchor : str | None
            Anchor of the highest-salience signal in the batch, for
            per-anchor refractory lookup.
        pushed_drives : set[str]
            Names of drives that received a non-trivial push (by either an
            additive multiplier or a dampen). Used by ``apply_decay`` so
            drives that just got pushed don't immediately decay in the same
            tick.
        """
        top_anchor: Optional[str] = None
        top_sal = -1.0
        pushed: set[str] = set()
        # Pre-index wirings by kind so the per-signal loop is O(matches) not O(all).
        wirings_by_kind: dict[str, list[SignalToDrive]] = {}
        for w in self._wirings:
            wirings_by_kind.setdefault(w.signal_kind, []).append(w)

        for sig in signals:
            s = max(0.0, min(1.0, float(sig.salience)))
            for wire in wirings_by_kind.get(sig.kind, ()):
                if wire.dampen_multiply is not None:
                    state.multiply(wire.drive_name, wire.dampen_multiply)
                else:
                    state.add(wire.drive_name, s * wire.multiplier)
                pushed.add(wire.drive_name)
            if s > top_sal:
                top_sal = s
                top_anchor = sig.anchor or None
        state.clip()
        return top_anchor, pushed

    def apply_decay(
        self,
        state: DriveState,
        *,
        pushed_drives: set[str] | None = None,
    ) -> None:
        """Decay drives that weren't pushed this tick; add ``growth_per_tick`` always.

        If a drive received a push this tick, it is NOT decayed in the same
        tick -- otherwise a signal followed by decay would be near-zero net.
        Growth (boredom-style) is applied unconditionally; the agent's
        ``boredom`` drive grows even on active ticks because waiting is a
        boredom-accumulating activity regardless of what else happened.
        """
        pushed_drives = pushed_drives or set()
        for drive in self._drives:
            if drive.name not in pushed_drives:
                state.multiply(drive.name, drive.decay_rate)
            if drive.growth_per_tick:
                state.add(drive.name, drive.growth_per_tick)
        state.clip()


__all__ = ["Homeostat"]
