"""ImpulseEngine -- ties drives, baseline, gate, compose, bandit together.

Port + generalisation of ``skynet_agent.modules.impulse.loop.run_tick``. The
original was synchronous and hard-coded for a single domain ("agent"); this
version is async, domain-parameterised via ``EngineConfig``, and emits a
``TickResult`` every tick so callers can log / persist without re-deriving
state.

**Tick vs heartbeat (signal-driven loop, 2026-04-20)**

Historically the engine was driven by a wall-clock polling loop that
called :meth:`tick` every ``tick_interval_seconds`` seconds. That is now
optional: :meth:`run_forever` blocks on ``XREADGROUP BLOCK`` against the
shared impulse stream and wakes immediately when a signal arrives. A
slow heartbeat (default 1h) still fires on timeout to run decay +
refractory housekeeping so long silences don't freeze drive state
forever. This matches the project's ``decay = logical time`` rule --
refractories tick on signal events, not wall-clock seconds.

Tick sequence (matches the original agent logic, extended):

    1. Drain signals from the domain's Redis consumer group (async).
    2. Apply drive wirings + decay.
    3. Append dominant-drive value to the baseline rolling window.
    4. Gate checks:
       - boredom-alone skip (no signals and boredom dominates)
       - staleness skip (no inbound signal in ``staleness_threshold_hours``)
       - rate-limit skip (>= ``rate_limit_per_day`` fires in last 24h)
       - refractory skip (per-anchor cooldown still ticking)
       - baseline skip (dominant < p75)
    5. Epsilon-greedy override: on baseline skip, occasionally fire anyway.
    6. LLM gate: final yes/no.
    7. Sample archetype from the bandit; compose the message.
    8. Return ``TickResult`` -- the CALLER is responsible for actually
       sending the message + emitting the self-feedback ``spoke`` signal +
       bumping refractory. Keeping delivery out of the engine means the
       same engine works for Matrix, Telegram, stdout-for-tests.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from .archetypes import Archetype, ArchetypeBandit
from .baseline import AdaptiveBaseline, BaselineConfig
from .compose import ComposeClient
from .drives import Drive, DriveState, SignalToDrive
from .exceptions import ConfigError
from .gate import GateClient
from .homeostat import Homeostat
from .signals import (
    DEFAULT_CONSUMER_GROUP,
    STREAM_NAME,
    Signal,
)

log = logging.getLogger(__name__)


# --- Config + result types -------------------------------------------------


@dataclass
class EngineConfig:
    """Complete engine config -- one instance per domain.

    The ``domain`` string scopes every Redis key (``{domain}:impulse:*``)
    and every consumer-group name, so music / movies / agent run fully
    isolated on the same signal bus.
    """

    domain: str
    drives: list[Drive]
    signal_to_drive: list[SignalToDrive]
    # Baseline
    baseline_window: int = 672
    baseline_percentile: int = 75
    cold_start_threshold: float = 0.35
    baseline_min_history: int = 30
    # Refractory
    refractory_cap_ticks: int = 4
    mentions_cap: int = 32
    # Firing rate
    rate_limit_per_day: int = 3
    rate_limit_window_seconds: int = 86400
    # Exploration / behavior
    epsilon_greedy: float = 0.05
    # DEPRECATED: wall-clock tick polling is retired in favour of
    # signal-driven ``run_forever``. Kept only so existing callers that
    # read ``cfg.tick_interval_seconds`` still work; when
    # ``heartbeat_seconds`` is left at its default, the value below is
    # copied over as a best-effort back-compat fallback.
    tick_interval_seconds: int = 900
    # Heartbeat: maximum time between housekeeping passes when no signals
    # are arriving. 3600s matches the project's wiki guidance of "decay
    # runs when something new arrives OR once an hour, whichever first".
    heartbeat_seconds: int = 3600
    # Max signals to drain per wake-up. Larger = fewer round-trips,
    # smaller = lower tail latency on bursts. 10 is a sane middle.
    batch_size: int = 10
    staleness_threshold_hours: int = 24
    # Consumer
    consumer_group: str = DEFAULT_CONSUMER_GROUP
    consumer_name: str = "engine-primary"
    stream_name: str = STREAM_NAME
    # Dominant-drive exclude (agent excludes "boredom" from solo-triggering)
    exclude_from_trigger: list[str] = field(default_factory=list)
    # Voice hint prepended to LLM prompts (domain personality nudge)
    voice_hint: str = ""

    @property
    def key_prefix(self) -> str:
        return f"skynet:impulses:{self.domain}"

    @property
    def state_key(self) -> str:
        return f"{self.key_prefix}:state"

    @property
    def fires_key(self) -> str:
        return f"{self.key_prefix}:fires"

    @property
    def last_signal_key(self) -> str:
        return f"{self.key_prefix}:last_signal_ts"

    def validate(self) -> None:
        if not self.domain:
            raise ConfigError("EngineConfig.domain must be non-empty")
        if not self.drives:
            raise ConfigError("EngineConfig.drives must contain at least one Drive")
        if not 0.0 <= self.epsilon_greedy <= 1.0:
            raise ConfigError(f"epsilon_greedy must be in [0, 1]; got {self.epsilon_greedy!r}")
        if self.rate_limit_per_day < 0:
            raise ConfigError("rate_limit_per_day must be >= 0")
        if self.heartbeat_seconds <= 0:
            raise ConfigError("heartbeat_seconds must be > 0")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be > 0")


@dataclass
class TickResult:
    """Everything the caller needs to know about a single tick."""

    timestamp: datetime
    drained_signals: int
    drive_updates: dict[str, float]
    dominant_drive: Optional[str]
    dominant_value: float
    baseline_p75: float
    fired: bool
    gate_reason: Optional[str] = None
    anchor: Optional[str] = None
    archetype: Optional[Archetype] = None
    message: Optional[str] = None
    rec_id: Optional[str] = None
    skip_reason: Optional[str] = None


@dataclass
class EngineState:
    """Diagnostic snapshot of the engine -- safe to expose on /healthz."""

    drives: dict[str, float]
    baseline_p75: float
    history_len: int
    rate_limit_remaining: int
    last_fire: Optional[datetime]
    active_refractories: list[tuple[str, int]]


# --- The engine ------------------------------------------------------------


class ImpulseEngine:
    """Async orchestrator for one domain."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        bus,
        redis,
        gate_llm: GateClient,
        compose_llm: ComposeClient,
        archetype_bandit: ArchetypeBandit | None = None,
        rng: random.Random | None = None,
    ):
        config.validate()
        self._cfg = config
        self._bus = bus
        self._redis = redis
        self._gate = gate_llm
        self._compose = compose_llm
        self._bandit = archetype_bandit
        self._rng = rng or random.Random()

        self._homeostat = Homeostat(
            drives=config.drives,
            signal_to_drive=config.signal_to_drive,
            state_key=config.state_key,
        )
        self._baseline = AdaptiveBaseline(
            BaselineConfig(
                prefix=config.key_prefix,
                window=config.baseline_window,
                percentile=float(config.baseline_percentile),
                cold_start_threshold=config.cold_start_threshold,
                min_history=config.baseline_min_history,
                refractory_cap_ticks=config.refractory_cap_ticks,
                mentions_cap=config.mentions_cap,
            )
        )
        self._history: list[TickResult] = []
        self._history_cap = 100
        self._started = False

    # ---- Lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Ensure consumer group exists + load the baseline into Redis-visible state.

        Idempotent; calling twice is a no-op.
        """
        if self._started:
            return
        # Consumer-group creation lives on the bus module -- forward the call
        # without binding to a specific impl so tests can pass a mock bus.
        try:
            await _maybe_await(
                self._bus.ensure_consumer_group,
                group=self._cfg.consumer_group,
                redis_client=self._redis,
                stream=self._cfg.stream_name,
            )
        except Exception as e:  # noqa: BLE001
            # Most common: "BUSYGROUP" already exists -- harmless. Log and
            # continue; if it's something worse the first drain_signals will
            # surface it.
            log.debug("start(): ensure_consumer_group said: %s", e)
        self._started = True

    async def stop(self) -> None:
        """Currently a no-op (state is all in Redis). Here for symmetry."""
        self._started = False

    # ---- Main tick --------------------------------------------------------

    async def tick(self) -> TickResult:
        """One full tick (legacy non-blocking drain).

        Retained for test harnesses, manual triggers, and back-compat with
        callers that still drive the engine on a wall-clock cadence. New
        code should prefer :meth:`run_forever` which is signal-driven.

        Pure side-effect on Redis; never raises.
        """
        now = datetime.now(timezone.utc)
        try:
            signals = await self._drain()
        except Exception as e:  # noqa: BLE001
            log.warning("impulse[%s] drain failed: %s", self._cfg.domain, e)
            result = TickResult(
                timestamp=now,
                drained_signals=0,
                drive_updates={},
                dominant_drive=None,
                dominant_value=0.0,
                baseline_p75=self._cfg.cold_start_threshold,
                fired=False,
                skip_reason=f"drain_error:{e}",
            )
            self._record(result)
            return result
        return await self._process_signals(signals, now=now)

    # ---- Signal-driven run loop ------------------------------------------

    async def run_forever(
        self,
        *,
        heartbeat_seconds: Optional[int] = None,
        batch_size: Optional[int] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Signal-driven main loop.

        Blocks on ``XREADGROUP BLOCK`` for up to ``heartbeat_seconds``;
        whenever one or more signals arrive they are processed via the
        same ``_process_signals`` pipeline that :meth:`tick` uses. On the
        block-timeout path :meth:`_heartbeat` runs instead — it only
        does time-independent housekeeping (drive decay + refractory
        expiry + staleness bookkeeping), never the gate/compose LLM
        calls. That means long silences don't burn LLM budget.

        Parameters
        ----------
        heartbeat_seconds:
            Override for ``cfg.heartbeat_seconds``. Mostly useful in
            tests that want a shorter safety-net timeout.
        batch_size:
            Override for ``cfg.batch_size``.
        stop_event:
            Optional asyncio event. When set, the loop exits cleanly
            after the next iteration (current block is interrupted by
            the event's wake semantics — we race between the drain and
            the event).
        """
        cfg = self._cfg
        hb = heartbeat_seconds if heartbeat_seconds is not None else cfg.heartbeat_seconds
        batch = batch_size if batch_size is not None else cfg.batch_size
        block_ms = max(1, int(hb * 1000))

        log.info(
            "impulse[%s] engine waiting for signals on stream=%s group=%s "
            "consumer=%s heartbeat_seconds=%d batch_size=%d",
            cfg.domain,
            cfg.stream_name,
            cfg.consumer_group,
            cfg.consumer_name,
            hb,
            batch,
        )

        while True:
            if stop_event is not None and stop_event.is_set():
                log.info("impulse[%s] stop_event set, exiting run_forever", cfg.domain)
                return
            try:
                signals = await self._drain(block_ms=block_ms, count=batch)
            except asyncio.CancelledError:
                log.info("impulse[%s] run_forever cancelled", cfg.domain)
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("impulse[%s] drain error in run_forever: %s", cfg.domain, e)
                # Brief backoff so a broken Redis doesn't hot-spin.
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    raise
                continue

            now = datetime.now(timezone.utc)
            try:
                if signals:
                    await self._process_signals(signals, now=now)
                else:
                    # Block-timeout: safety-net housekeeping only.
                    self._heartbeat(now=now)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("impulse[%s] process loop error", cfg.domain)
                # Never exit the loop on per-iteration errors.
                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    raise

    # ---- Process one batch of signals ------------------------------------

    async def _process_signals(
        self,
        signals: list[Signal],
        *,
        now: Optional[datetime] = None,
    ) -> TickResult:
        """Apply signals -> drives -> baseline -> gate -> compose.

        This is the shared body for both :meth:`tick` (legacy wall-clock)
        and :meth:`run_forever` (signal-driven). Refractory countdown
        happens here, so refractories are counted in *signal events* not
        wall-clock ticks -- matching the project's logical-time decay
        convention (see ``memory/feedback_decay_logical_time.md``).
        """
        now = now or datetime.now(timezone.utc)

        # Update + persist drive state.
        state = self._homeostat.load_state(self._redis)
        top_anchor, pushed = self._homeostat.apply_signals(state, signals)
        self._homeostat.apply_decay(state, pushed_drives=pushed)
        self._baseline.tick_refractories(self._redis)

        # Track the freshest inbound signal so staleness checks work even
        # across process restarts (state lives in Redis).
        if signals:
            try:
                self._redis.set(self._cfg.last_signal_key, int(now.timestamp()))
            except Exception as e:  # noqa: BLE001
                log.debug("impulse[%s] last_signal_ts persist failed: %s", self._cfg.domain, e)

        # Dominant is computed over ALL drives so skip-reason distinguishes
        # "no drive active" from "only an excluded drive is active"; the
        # exclude list is enforced in ``_evaluate_skip`` below.
        dominant_drive, dominant_value = state.dominant()
        self._baseline.append_history(self._redis, dominant_value)
        baseline_p75 = self._baseline.p75(self._redis)

        result = TickResult(
            timestamp=now,
            drained_signals=len(signals),
            drive_updates=state.to_dict(),
            dominant_drive=dominant_drive,
            dominant_value=dominant_value,
            baseline_p75=baseline_p75,
            fired=False,
            anchor=top_anchor,
        )

        # Early exits -- each saves state and returns a TickResult.
        skip_reason = await self._evaluate_skip(
            signals, state, dominant_drive, dominant_value, baseline_p75, top_anchor, now
        )
        if skip_reason is not None:
            result.skip_reason = skip_reason
            self._homeostat.save_state(self._redis, state)
            self._record(result)
            return result

        # LLM gate.
        recent_context = ""  # caller enriches via context_fetcher in a later iteration
        try:
            fire, gate_reason = await self._gate.should_fire(
                domain=self._cfg.domain,
                dominant_drive=dominant_drive or "",
                dominant_value=dominant_value,
                baseline_p75=baseline_p75,
                anchor=top_anchor,
                recent_context=recent_context,
                drives=state.to_dict(),
                signals=signals,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("impulse[%s] gate raised: %s", self._cfg.domain, e)
            fire, gate_reason = False, f"gate_error:{e}"
        result.gate_reason = gate_reason
        if not fire:
            result.skip_reason = "gate_no"
            self._homeostat.save_state(self._redis, state)
            self._record(result)
            return result

        # Archetype + compose.
        archetype = self._sample_archetype(signals)
        result.archetype = archetype
        try:
            message = await self._compose.compose(
                domain=self._cfg.domain,
                dominant_drive=dominant_drive or "",
                anchor=top_anchor,
                archetype=archetype,
                context={
                    "drives": state.to_dict(),
                    "signals": signals,
                    "reason": gate_reason,
                },
            )
        except Exception as e:  # noqa: BLE001
            log.warning("impulse[%s] compose raised: %s", self._cfg.domain, e)
            message = ""
        if not message:
            result.skip_reason = "compose_failed"
            self._homeostat.save_state(self._redis, state)
            self._record(result)
            return result

        # Bookkeeping for successful fire.
        result.fired = True
        result.message = message
        result.rec_id = uuid.uuid4().hex[:12]
        self._record_fire(now)

        self._homeostat.save_state(self._redis, state)
        self._record(result)
        return result

    # ---- Diagnostics ------------------------------------------------------

    def state(self) -> EngineState:
        drive_state = self._homeostat.load_state(self._redis)
        p75 = self._baseline.p75(self._redis)
        remaining = self._cfg.rate_limit_per_day - self._recent_fire_count()
        last = self._latest_fire_time()
        refractories = self._baseline.list_active_refractories(self._redis)
        return EngineState(
            drives=drive_state.to_dict(),
            baseline_p75=p75,
            history_len=self._baseline.history_len(self._redis),
            rate_limit_remaining=max(0, remaining),
            last_fire=last,
            active_refractories=refractories,
        )

    def history(self, n: int = 20) -> list[TickResult]:
        return self._history[-n:]

    # ---- Internals --------------------------------------------------------

    async def _drain(
        self,
        *,
        block_ms: Optional[int] = None,
        count: Optional[int] = None,
    ) -> list[Signal]:
        """Pull pending signals from the bus for our consumer group.

        ``block_ms`` and ``count`` are optional forwards to the bus
        adapter's ``drain_signals``. Older adapters that don't accept
        these kwargs are tolerated via a ``TypeError`` fallback so
        callers on older code paths (e.g. legacy :meth:`tick`) keep
        working unchanged.
        """
        kwargs = {
            "group": self._cfg.consumer_group,
            "async_redis": self._redis,
            "stream": self._cfg.stream_name,
        }
        if block_ms is not None:
            kwargs["block_ms"] = block_ms
        if count is not None:
            kwargs["count"] = count
        try:
            pairs = await _maybe_await(
                self._bus.drain_signals,
                self._cfg.consumer_name,
                **kwargs,
            )
        except TypeError:
            # Older adapter: retry without the new kwargs. Accept both
            # signal-driven and legacy adapters on the same code path.
            kwargs.pop("block_ms", None)
            kwargs.pop("count", None)
            pairs = await _maybe_await(
                self._bus.drain_signals,
                self._cfg.consumer_name,
                **kwargs,
            )
        sigs: list[Signal] = []
        ids: list[str] = []
        for entry_id, sig in pairs or []:
            sigs.append(sig)
            ids.append(entry_id)
        if ids:
            try:
                await _maybe_await(
                    self._bus.ack_signals,
                    ids,
                    group=self._cfg.consumer_group,
                    async_redis=self._redis,
                    stream=self._cfg.stream_name,
                )
            except Exception as e:  # noqa: BLE001
                log.debug("impulse[%s] ack failed: %s", self._cfg.domain, e)
        return sigs

    # ---- Heartbeat (slow housekeeping) -----------------------------------

    def _heartbeat(self, *, now: Optional[datetime] = None) -> None:
        """Safety-net housekeeping fired on block-timeout in run_forever.

        Does only time-independent work:

        * Apply ``growth_per_tick`` for drives that grow passively (e.g.
          the agent's ``boredom`` drive).
        * Apply one decay step to drives that didn't just get bumped.
        * Persist drive state.

        Explicitly does NOT run the LLM gate, compose, or touch
        refractories: refractories are "number of signal events since
        last bump" per the logical-time convention. Drive state,
        however, does need to evolve across very long quiet periods or
        dominant-drive values become misleading artefacts of the last
        signal burst.
        """
        _ = now or datetime.now(timezone.utc)
        try:
            state = self._homeostat.load_state(self._redis)
            # pushed_drives=set() means every drive decays this heartbeat.
            self._homeostat.apply_decay(state, pushed_drives=set())
            self._homeostat.save_state(self._redis, state)
        except Exception as e:  # noqa: BLE001
            log.debug("impulse[%s] heartbeat decay failed: %s", self._cfg.domain, e)

    async def _evaluate_skip(
        self,
        signals: list[Signal],
        state: DriveState,
        dominant_drive: Optional[str],
        dominant_value: float,
        baseline_p75: float,
        top_anchor: Optional[str],
        now: datetime,
    ) -> Optional[str]:
        """Return a skip reason or ``None`` to proceed to the gate."""
        cfg = self._cfg

        # Boredom-alone: dominant drive is in the exclude list AND no fresh signals.
        if dominant_drive is None:
            return "no_active_drive"
        if dominant_drive in cfg.exclude_from_trigger and not signals:
            return "excluded_drive_alone"

        # Staleness: the domain hasn't seen a signal in ``staleness_threshold_hours``.
        if self._is_stale(now):
            return "stale_domain"

        # Rate limit.
        if self._recent_fire_count() >= cfg.rate_limit_per_day:
            return "rate_limit"

        # Refractory on the top-anchor.
        if top_anchor:
            remaining = self._baseline.remaining_refractory(self._redis, top_anchor)
            if remaining > 0:
                return f"refractory:{remaining}"

        # Baseline threshold -- with epsilon-greedy exploration override.
        if dominant_value < baseline_p75:
            if self._rng.random() < cfg.epsilon_greedy:
                # Fire anyway; skip returns None so the gate still runs.
                log.debug("impulse[%s] epsilon-greedy override below baseline", cfg.domain)
                return None
            return "below_threshold"

        return None

    def _sample_archetype(self, signals: list[Signal]) -> Archetype:
        # Derive trigger_kind from the top-salience signal if we have one;
        # defaults to "*" so the bandit treats it as an unclassified trigger.
        trigger_kind: Optional[str] = None
        if signals:
            top = max(signals, key=lambda s: s.salience)
            trigger_kind = top.kind
        if self._bandit is None:
            # Deterministic fallback so the engine still composes without a bandit.
            return Archetype(
                trigger_kind or "*",
                tone="reflective",
                length="short",
            )
        return self._bandit.sample(trigger_kind=trigger_kind)

    # ---- Rate limit helpers ----------------------------------------------

    def _record_fire(self, now: datetime) -> None:
        ts = int(now.timestamp())
        try:
            self._redis.zadd(self._cfg.fires_key, {str(ts): ts})
            # Trim old entries eagerly -- keeps the ZSET small.
            cutoff = ts - self._cfg.rate_limit_window_seconds
            self._redis.zremrangebyscore(self._cfg.fires_key, 0, cutoff)
        except Exception as e:  # noqa: BLE001
            log.debug("impulse[%s] record_fire failed: %s", self._cfg.domain, e)

    def _recent_fire_count(self) -> int:
        now_ts = int(datetime.now(timezone.utc).timestamp())
        cutoff = now_ts - self._cfg.rate_limit_window_seconds
        try:
            return int(self._redis.zcount(self._cfg.fires_key, cutoff, "+inf") or 0)
        except Exception:  # noqa: BLE001
            return 0

    def _latest_fire_time(self) -> Optional[datetime]:
        try:
            raw = self._redis.zrevrange(self._cfg.fires_key, 0, 0, withscores=True)
        except Exception:
            return None
        if not raw:
            return None
        _member, score = raw[0]
        return datetime.fromtimestamp(float(score), tz=timezone.utc)

    def _is_stale(self, now: datetime) -> bool:
        try:
            raw = self._redis.get(self._cfg.last_signal_key)
        except Exception:
            return False
        if raw is None:
            # Cold start: don't call the domain stale if we simply have no history.
            return False
        try:
            last_ts = int(raw if not isinstance(raw, bytes) else raw.decode())
        except (TypeError, ValueError):
            return False
        delta = now - datetime.fromtimestamp(last_ts, tz=timezone.utc)
        return delta > timedelta(hours=self._cfg.staleness_threshold_hours)

    def _record(self, result: TickResult) -> None:
        self._history.append(result)
        if len(self._history) > self._history_cap:
            self._history = self._history[-self._history_cap :]


async def _maybe_await(fn, *args, **kwargs):
    """Call ``fn`` allowing either sync or async implementations.

    The bus module ships async helpers (``drain_signals``), but callers who
    inject a mock bus for tests usually provide plain ``MagicMock`` returns.
    This shim lets both kinds work without branching at every call site.
    """
    result = fn(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result


__all__ = [
    "EngineConfig",
    "TickResult",
    "EngineState",
    "ImpulseEngine",
]
