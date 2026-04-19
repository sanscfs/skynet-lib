"""ImpulseEngine tick orchestration: happy path + every skip reason."""

from __future__ import annotations

import random
from unittest.mock import AsyncMock

import pytest
from _fake_redis import FakeRedis
from skynet_impulse.archetypes import default_archetypes
from skynet_impulse.drives import Drive, SignalToDrive
from skynet_impulse.engine import EngineConfig, ImpulseEngine
from skynet_impulse.signals import Signal


class FakeBus:
    """Minimal async bus -- drains whatever the test seeded."""

    def __init__(self, signals: list[Signal] | None = None):
        self._signals = list(signals or [])

    async def ensure_consumer_group(self, **_kw):
        return None

    async def drain_signals(self, *args, **kwargs):
        out = [(f"id-{i}", s) for i, s in enumerate(self._signals)]
        self._signals = []
        return out

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
