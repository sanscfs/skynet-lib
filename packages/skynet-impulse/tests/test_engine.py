"""ImpulseEngine tick orchestration: happy path + every skip reason."""

from __future__ import annotations

import asyncio
import random
from unittest.mock import AsyncMock

import pytest
from _fake_redis import FakeRedis
from skynet_impulse.archetypes import default_archetypes
from skynet_impulse.drives import Drive, SignalToDrive
from skynet_impulse.engine import EngineConfig, ImpulseEngine
from skynet_impulse.signals import Signal


class FakeBus:
    """Minimal async bus -- drains whatever the test seeded.

    Supports ``block_ms`` kwarg to mimic the real ``XREADGROUP BLOCK``:
    when the queue is empty and ``block_ms > 0`` the drain sleeps for
    that long and returns an empty list (emulating the timeout). That
    lets ``run_forever`` tests exercise the heartbeat path deterministically
    by seeding a batch, letting it drain instantly, then letting the
    subsequent call time out.
    """

    def __init__(self, signals: list[Signal] | None = None):
        self._signals = list(signals or [])
        self._drain_calls = 0

    async def ensure_consumer_group(self, **_kw):
        return None

    async def drain_signals(self, *args, block_ms: int | None = None, **kwargs):
        self._drain_calls += 1
        if self._signals:
            out = [(f"id-{i}", s) for i, s in enumerate(self._signals)]
            self._signals = []
            return out
        if block_ms:
            # Mimic Redis blocking drain: sleep for block_ms then return [].
            await asyncio.sleep(block_ms / 1000.0)
        return []

    async def ack_signals(self, *args, **kwargs):
        return 0

    def seed(self, signals: list[Signal]) -> None:
        self._signals = list(signals)


def _make_engine(
    *,
    bus: FakeBus,
    redis,
    gate_fire: bool = True,
    compose_text: str = "Думав про той трек.",
    rng_seed: int = 0,
    rate_limit: int = 3,
    epsilon: float = 0.0,
    staleness_hours: int = 24,
):
    cfg = EngineConfig(
        domain="test",
        drives=[
            Drive("boredom", 0.9, growth_per_tick=0.08),
            Drive("curiosity", 0.85),
            Drive("concern", 0.8),
            Drive("need_to_share", 0.9),
        ],
        signal_to_drive=[
            SignalToDrive("novelty", "curiosity", 0.30),
            SignalToDrive("concern", "concern", 0.40),
        ],
        baseline_window=100,
        baseline_percentile=75,
        baseline_min_history=5,
        cold_start_threshold=0.05,
        rate_limit_per_day=rate_limit,
        epsilon_greedy=epsilon,
        staleness_threshold_hours=staleness_hours,
        exclude_from_trigger=["boredom"],
    )

    gate = AsyncMock()
    gate.should_fire.return_value = (gate_fire, "test reason")

    compose = AsyncMock()
    compose.compose.return_value = compose_text

    bandit = ArchetypeBandit_fixed()
    engine = ImpulseEngine(
        cfg,
        bus=bus,
        redis=redis,
        gate_llm=gate,
        compose_llm=compose,
        archetype_bandit=bandit,
        rng=random.Random(rng_seed),
    )
    return engine, gate, compose


def ArchetypeBandit_fixed():
    """Deterministic bandit whose .sample always returns a fixed archetype."""
    from skynet_impulse.archetypes import ArchetypeBandit

    return ArchetypeBandit(default_archetypes(), rng=random.Random(0))


@pytest.mark.asyncio
async def test_tick_happy_path_fires():
    bus = FakeBus([Signal(kind="novelty", source="analyzer", salience=0.9, anchor="music:x")])
    redis = FakeRedis()
    engine, gate, compose = _make_engine(bus=bus, redis=redis)
    await engine.start()
    # Prime baseline history so we're past cold start.
    for v in [0.1] * 10:
        engine._baseline.append_history(redis, v)
    result = await engine.tick()
    assert result.fired is True
    assert result.message == "Думав про той трек."
    assert result.rec_id is not None
    assert gate.should_fire.await_count == 1
    assert compose.compose.await_count == 1


@pytest.mark.asyncio
async def test_tick_skips_when_no_signals_and_only_boredom():
    bus = FakeBus([])
    redis = FakeRedis()
    engine, gate, compose = _make_engine(bus=bus, redis=redis)
    await engine.start()
    result = await engine.tick()
    # Boredom grows on every tick via growth_per_tick; with no signals it'd be dominant.
    assert result.fired is False
    assert result.skip_reason == "excluded_drive_alone"
    assert gate.should_fire.await_count == 0


@pytest.mark.asyncio
async def test_tick_rate_limit_skips():
    bus = FakeBus([Signal(kind="novelty", source="analyzer", salience=0.9, anchor="a")])
    redis = FakeRedis()
    engine, gate, compose = _make_engine(bus=bus, redis=redis, rate_limit=0)
    await engine.start()
    result = await engine.tick()
    assert result.fired is False
    assert result.skip_reason == "rate_limit"


@pytest.mark.asyncio
async def test_tick_below_threshold_without_epsilon():
    # Prime the baseline at 0.8 so our weak signal is below.
    bus = FakeBus([Signal(kind="novelty", source="analyzer", salience=0.1, anchor="a")])
    redis = FakeRedis()
    engine, _gate, _compose = _make_engine(bus=bus, redis=redis, epsilon=0.0)
    await engine.start()
    for v in [0.9] * 50:
        engine._baseline.append_history(redis, v)
    result = await engine.tick()
    assert result.fired is False
    assert result.skip_reason == "below_threshold"


@pytest.mark.asyncio
async def test_tick_epsilon_greedy_overrides_threshold():
    bus = FakeBus([Signal(kind="novelty", source="analyzer", salience=0.1, anchor="a")])
    redis = FakeRedis()
    engine, _gate, _compose = _make_engine(bus=bus, redis=redis, epsilon=1.0)
    await engine.start()
    for v in [0.9] * 50:
        engine._baseline.append_history(redis, v)
    result = await engine.tick()
    # epsilon=1.0 guarantees override; gate will fire (default).
    assert result.fired is True


@pytest.mark.asyncio
async def test_tick_gate_no_does_not_fire():
    bus = FakeBus([Signal(kind="novelty", source="analyzer", salience=0.9, anchor="a")])
    redis = FakeRedis()
    engine, _gate, _compose = _make_engine(bus=bus, redis=redis, gate_fire=False)
    await engine.start()
    for v in [0.1] * 10:
        engine._baseline.append_history(redis, v)
    result = await engine.tick()
    assert result.fired is False
    assert result.skip_reason == "gate_no"


@pytest.mark.asyncio
async def test_tick_refractory_blocks():
    bus = FakeBus([Signal(kind="novelty", source="analyzer", salience=0.9, anchor="A")])
    redis = FakeRedis()
    engine, _gate, _compose = _make_engine(bus=bus, redis=redis)
    await engine.start()
    for v in [0.1] * 10:
        engine._baseline.append_history(redis, v)
    # Bump twice so after the tick's own ``tick_refractories`` decrement the
    # cooldown is still > 0 when the skip-evaluator inspects it.
    engine._baseline.bump_refractory(redis, "A")
    engine._baseline.bump_refractory(redis, "A")
    result = await engine.tick()
    assert result.fired is False
    assert result.skip_reason.startswith("refractory")


@pytest.mark.asyncio
async def test_tick_compose_empty_does_not_fire():
    bus = FakeBus([Signal(kind="novelty", source="analyzer", salience=0.9, anchor="a")])
    redis = FakeRedis()
    engine, _gate, _compose = _make_engine(bus=bus, redis=redis, compose_text="")
    await engine.start()
    for v in [0.1] * 10:
        engine._baseline.append_history(redis, v)
    result = await engine.tick()
    assert result.fired is False
    assert result.skip_reason == "compose_failed"


@pytest.mark.asyncio
async def test_engine_config_validation():
    from skynet_impulse.exceptions import ConfigError

    with pytest.raises(ConfigError):
        EngineConfig(domain="", drives=[Drive("x")], signal_to_drive=[]).validate()

    with pytest.raises(ConfigError):
        EngineConfig(domain="d", drives=[], signal_to_drive=[]).validate()

    with pytest.raises(ConfigError):
        EngineConfig(
            domain="d",
            drives=[Drive("x")],
            signal_to_drive=[],
            epsilon_greedy=1.5,
        ).validate()


@pytest.mark.asyncio
async def test_state_reports_diagnostics():
    bus = FakeBus([])
    redis = FakeRedis()
    engine, _, _ = _make_engine(bus=bus, redis=redis)
    await engine.start()
    state = engine.state()
    assert "boredom" in state.drives
    assert state.rate_limit_remaining == 3


@pytest.mark.asyncio
async def test_tick_history_ring_buffer():
    bus = FakeBus([])
    redis = FakeRedis()
    engine, _, _ = _make_engine(bus=bus, redis=redis)
    await engine.start()
    for _ in range(5):
        await engine.tick()
    hist = engine.history(n=3)
    assert len(hist) == 3


@pytest.mark.asyncio
async def test_tick_records_fire_for_rate_limit():
    """After firing, the rate-limit counter should reflect the new fire."""
    bus = FakeBus([Signal(kind="novelty", source="analyzer", salience=0.9, anchor="A")])
    redis = FakeRedis()
    engine, _, _ = _make_engine(bus=bus, redis=redis, rate_limit=2)
    await engine.start()
    for v in [0.1] * 10:
        engine._baseline.append_history(redis, v)
    await engine.tick()
    remaining_after = engine.state().rate_limit_remaining
    assert remaining_after == 1


# ---- Signal-driven run_forever + heartbeat ------------------------------


@pytest.mark.asyncio
async def test_run_forever_wakes_on_signal_and_processes():
    """A seeded signal should drive a full process + gate + compose."""
    bus = FakeBus([Signal(kind="novelty", source="analyzer", salience=0.9, anchor="X")])
    redis = FakeRedis()
    engine, gate, compose = _make_engine(bus=bus, redis=redis)
    await engine.start()
    for v in [0.1] * 10:
        engine._baseline.append_history(redis, v)

    stop = asyncio.Event()

    async def _stop_after_process():
        # Wait briefly for the first batch to be drained + processed.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if gate.should_fire.await_count > 0:
                break
        stop.set()

    task = asyncio.create_task(engine.run_forever(heartbeat_seconds=5, batch_size=5, stop_event=stop))
    await _stop_after_process()
    try:
        await asyncio.wait_for(task, timeout=2)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    assert gate.should_fire.await_count == 1
    assert compose.compose.await_count == 1


@pytest.mark.asyncio
async def test_run_forever_heartbeat_on_timeout_does_not_call_gate():
    """With no signals, run_forever must hit the heartbeat path, not the gate."""
    bus = FakeBus([])  # empty
    redis = FakeRedis()
    engine, gate, compose = _make_engine(bus=bus, redis=redis)
    await engine.start()

    stop = asyncio.Event()

    async def _cancel_after_two_heartbeats():
        # Wait long enough for at least one block-timeout-then-heartbeat cycle.
        # block_ms = heartbeat_seconds*1000, so keep heartbeat small.
        await asyncio.sleep(0.15)
        stop.set()

    task = asyncio.create_task(engine.run_forever(heartbeat_seconds=0.05, batch_size=5, stop_event=stop))
    await _cancel_after_two_heartbeats()
    try:
        await asyncio.wait_for(task, timeout=1)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    # No signals = no gate/compose calls.
    assert gate.should_fire.await_count == 0
    assert compose.compose.await_count == 0
    # But the bus should have been drained at least once.
    assert bus._drain_calls >= 1


@pytest.mark.asyncio
async def test_heartbeat_decays_drives_without_running_gate():
    """Direct unit test: _heartbeat decays drive state and nothing else."""
    bus = FakeBus([])
    redis = FakeRedis()
    engine, gate, compose = _make_engine(bus=bus, redis=redis)
    await engine.start()

    # Seed a drive above 0 so decay can be observed.
    state = engine._homeostat.load_state(redis)
    state.set("curiosity", 0.8)
    engine._homeostat.save_state(redis, state)

    engine._heartbeat()

    after = engine._homeostat.load_state(redis)
    assert after.get("curiosity") < 0.8  # decayed
    assert gate.should_fire.await_count == 0
    assert compose.compose.await_count == 0


@pytest.mark.asyncio
async def test_refractory_counts_signal_events_not_wall_clock():
    """Refractory should count down on each signal-processing call, not time.

    This is the logical-time property: two ``_process_signals`` calls
    decrement refractory twice; wall-clock waits alone do nothing.
    """
    bus = FakeBus([])
    redis = FakeRedis()
    engine, _, _ = _make_engine(bus=bus, redis=redis)
    await engine.start()

    # Set up a refractory at cap (=4 after 4 bumps).
    for _ in range(4):
        engine._baseline.bump_refractory(redis, "A")
    before = engine._baseline.remaining_refractory(redis, "A")

    # Heartbeat must NOT touch refractories.
    engine._heartbeat()
    assert engine._baseline.remaining_refractory(redis, "A") == before

    # Each _process_signals decrements refractories by 1.
    await engine._process_signals([])
    after_one = engine._baseline.remaining_refractory(redis, "A")
    assert after_one == before - 1

    await engine._process_signals([])
    after_two = engine._baseline.remaining_refractory(redis, "A")
    assert after_two == before - 2


@pytest.mark.asyncio
async def test_run_forever_processes_multiple_signal_bursts():
    """Seed two bursts; both must end up processed through the gate."""
    bus = FakeBus([Signal(kind="novelty", source="analyzer", salience=0.9, anchor="A")])
    redis = FakeRedis()
    engine, gate, compose = _make_engine(bus=bus, redis=redis, rate_limit=5)
    await engine.start()
    for v in [0.1] * 10:
        engine._baseline.append_history(redis, v)

    stop = asyncio.Event()

    async def _feed_second_burst_then_stop():
        # Wait for first burst to clear.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if gate.should_fire.await_count >= 1:
                break
        bus.seed([Signal(kind="novelty", source="analyzer", salience=0.9, anchor="B")])
        for _ in range(50):
            await asyncio.sleep(0.01)
            if gate.should_fire.await_count >= 2:
                break
        stop.set()

    task = asyncio.create_task(engine.run_forever(heartbeat_seconds=0.05, batch_size=5, stop_event=stop))
    await _feed_second_burst_then_stop()
    try:
        await asyncio.wait_for(task, timeout=2)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    assert gate.should_fire.await_count >= 2


@pytest.mark.asyncio
async def test_engine_config_back_compat_heartbeat_default():
    """An EngineConfig not setting heartbeat_seconds still validates."""
    cfg = EngineConfig(
        domain="bc",
        drives=[Drive("x")],
        signal_to_drive=[SignalToDrive("novelty", "x", 0.3)],
    )
    cfg.validate()
    assert cfg.heartbeat_seconds == 3600
    assert cfg.batch_size == 10


@pytest.mark.asyncio
async def test_engine_config_rejects_zero_heartbeat():
    """heartbeat_seconds must be strictly positive."""
    from skynet_impulse.exceptions import ConfigError

    cfg = EngineConfig(
        domain="bc",
        drives=[Drive("x")],
        signal_to_drive=[SignalToDrive("novelty", "x", 0.3)],
        heartbeat_seconds=0,
    )
    with pytest.raises(ConfigError):
        cfg.validate()
