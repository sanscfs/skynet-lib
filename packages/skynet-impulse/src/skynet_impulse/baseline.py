"""Adaptive p-percentile baseline + per-anchor refractory cooldown.

Direct port of ``skynet_agent.modules.impulse.baseline``, parameterised so
each domain carries its own history list and refractory hash. The only
behavioural change: the Redis key prefix is injected by the engine rather
than hard-coded to ``agent:impulse:*``.

How it works:

- Each tick the engine appends the dominant drive's *value* to a rolling
  list of the last ``window`` values. The p75 of that list is the adaptive
  threshold. Below that, the agent stays silent.
- The first ``min_history`` ticks fall back to ``cold_start_threshold`` so a
  freshly-booted domain doesn't freeze waiting for its own baseline.
- After speaking, the engine calls ``bump_refractory(anchor)``. The cooldown
  starts at ``min(mentions_for_anchor, refractory_cap_ticks)`` and ticks down
  each subsequent tick. An anchor that gets hit many times caps at the same
  ceiling -- an obsessive spam anchor can't lock itself in refractory forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BaselineConfig:
    """Keys + knobs for baseline + refractory.

    All Redis keys are derived from a ``prefix`` like ``music:curiosity`` so
    each domain has its own history/refractory without stepping on siblings.
    """

    prefix: str
    window: int = 672
    percentile: float = 75.0
    cold_start_threshold: float = 0.35
    min_history: int = 30
    refractory_cap_ticks: int = 4
    mentions_cap: int = 32

    @property
    def history_key(self) -> str:
        return f"{self.prefix}:history"

    @property
    def refractory_key(self) -> str:
        return f"{self.prefix}:refractory"

    @property
    def mentions_key(self) -> str:
        return f"{self.prefix}:mentions"


class AdaptiveBaseline:
    """Wraps baseline + refractory Redis ops so the engine stays small.

    This class holds no in-process state -- every method round-trips to Redis
    so two engine replicas could share a domain without split-brain. Rolling
    window trim uses ``LTRIM`` which is atomic server-side.
    """

    def __init__(self, config: BaselineConfig):
        self._cfg = config

    # ---- baseline / history ----------------------------------------------

    def append_history(self, redis_client, dominant_value: float) -> None:
        cfg = self._cfg
        redis_client.rpush(cfg.history_key, f"{dominant_value:.4f}")
        redis_client.ltrim(cfg.history_key, -cfg.window, -1)

    def p75(self, redis_client) -> float:
        """The adaptive threshold. Falls back to ``cold_start_threshold`` early."""
        cfg = self._cfg
        raw = redis_client.lrange(cfg.history_key, 0, -1) or []
        if len(raw) < cfg.min_history:
            return cfg.cold_start_threshold
        values = sorted(_coerce_float(v) for v in raw)
        # Rank interpolation is overkill; nearest-rank is what the agent shipped.
        idx = int(len(values) * cfg.percentile / 100.0)
        idx = min(idx, len(values) - 1)
        return values[idx]

    def history_len(self, redis_client) -> int:
        return int(redis_client.llen(self._cfg.history_key) or 0)

    # ---- refractory ------------------------------------------------------

    def remaining_refractory(self, redis_client, anchor: Optional[str]) -> int:
        if not anchor:
            return 0
        raw = redis_client.hget(self._cfg.refractory_key, anchor)
        return _coerce_int(raw)

    def bump_refractory(self, redis_client, anchor: str) -> None:
        """Increment per-anchor mentions counter and set cooldown = min(mentions, cap)."""
        cfg = self._cfg
        mentions = int(redis_client.hincrby(cfg.mentions_key, anchor, 1))
        if mentions > cfg.mentions_cap:
            redis_client.hset(cfg.mentions_key, anchor, cfg.mentions_cap)
            mentions = cfg.mentions_cap
        cooldown = min(mentions, cfg.refractory_cap_ticks)
        redis_client.hset(cfg.refractory_key, anchor, cooldown)

    def tick_refractories(self, redis_client) -> None:
        """Decrement every live refractory; evict when it hits zero."""
        all_anchors = redis_client.hgetall(self._cfg.refractory_key) or {}
        # Redis-py returns bytes when decode_responses=False.
        if all_anchors and isinstance(next(iter(all_anchors)), bytes):
            all_anchors = {k.decode(): v.decode() for k, v in all_anchors.items()}
        for anchor, remaining in all_anchors.items():
            r = _coerce_int(remaining) - 1
            if r <= 0:
                redis_client.hdel(self._cfg.refractory_key, anchor)
            else:
                redis_client.hset(self._cfg.refractory_key, anchor, r)

    def list_active_refractories(self, redis_client) -> list[tuple[str, int]]:
        """Diagnostic: every (anchor, ticks_remaining) currently in cooldown."""
        raw = redis_client.hgetall(self._cfg.refractory_key) or {}
        if raw and isinstance(next(iter(raw)), bytes):
            raw = {k.decode(): v.decode() for k, v in raw.items()}
        return [(anchor, _coerce_int(remaining)) for anchor, remaining in raw.items()]


def _coerce_float(raw) -> float:
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _coerce_int(raw) -> int:
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


__all__ = ["BaselineConfig", "AdaptiveBaseline"]
