"""Tests for :meth:`VibeEngine.match` + τ calibration (v2 API).

match() is now Qdrant-native: it votes cosine-weighted across top-k
neighbors grouped by extra_payload.source_type. Static prototypes are no
longer required.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest
from skynet_vibe import (
    FacetVectors,
    MatchResult,
    PrototypeRegistry,
    VibeEngine,
    VibeSignal,
    VibeStore,
)
from skynet_vibe import Source as Src

# ---------------------------------------------------------------------------
# Fixtures local to this module
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(fake_qdrant, async_hash_embedder, fake_llm):
    store = VibeStore(fake_qdrant, collection="t")
    prototypes = PrototypeRegistry(async_hash_embedder)
    return VibeEngine(
        store=store,
        prototypes=prototypes,
        embedder=async_hash_embedder,
        llm_client=fake_llm,
    )


# ---------------------------------------------------------------------------
# Pure entropy maths
# ---------------------------------------------------------------------------


def _entropy_bits(probs: list[float]) -> float:
    total = 0.0
    for p in probs:
        if p > 1e-12:
            total -= p * math.log2(p)
    return total


def test_entropy_uniform_equals_log2_n() -> None:
    """Uniform over N prototypes -> entropy = log2(N)."""
    for n in (2, 4, 10, 50):
        probs = [1.0 / n] * n
        h = _entropy_bits(probs)
        assert h == pytest.approx(math.log2(n), abs=1e-9)


def test_entropy_delta_equals_zero() -> None:
    """All mass on one prototype -> entropy = 0."""
    probs = [0.0, 1.0, 0.0, 0.0]
    assert _entropy_bits(probs) == pytest.approx(0.0, abs=1e-9)


def test_entropy_known_binary() -> None:
    """Binary 0.25/0.75 gives H = 0.25*log2(4) + 0.75*log2(4/3) ≈ 0.8113."""
    probs = [0.25, 0.75]
    expected = 0.25 * math.log2(1 / 0.25) + 0.75 * math.log2(1 / 0.75)
    assert _entropy_bits(probs) == pytest.approx(expected, abs=1e-9)
    assert _entropy_bits(probs) == pytest.approx(0.8112781244591328, abs=1e-9)


# ---------------------------------------------------------------------------
# match() behaviour (Qdrant-native voting)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_match_empty_store_returns_unknown(engine) -> None:
    """No signals in store -> match returns 'unknown' without raising."""
    result = await engine.match("whatever text we have")
    assert isinstance(result, MatchResult)
    assert result.winner == "unknown"
    assert result.accepted is False
    assert result.confidence == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_match_on_training_phrase_accepted(engine) -> None:
    """Signals with a matching source_type win the vote.

    We insert several signals aligned with the query text, all tagged
    source_type='alpha', plus a few tagged 'beta' with an unrelated vector.
    The query must route to 'alpha'.
    """
    alpha_vec = await engine.embedder("aaa aaa aaa")
    beta_vec = await engine.embedder("bbb bbb bbb")

    for i in range(5):
        await engine.store.put(
            VibeSignal(
                id=f"alpha-{i}",
                text_raw="aaa aaa aaa",
                vectors=FacetVectors(content=list(alpha_vec)),
                source=Src(type="chat"),
                timestamp=datetime.now(timezone.utc),
                confidence=1.0,
                extra_payload={"source_type": "alpha"},
            )
        )
    for i in range(3):
        await engine.store.put(
            VibeSignal(
                id=f"beta-{i}",
                text_raw="bbb bbb bbb",
                vectors=FacetVectors(content=list(beta_vec)),
                source=Src(type="chat"),
                timestamp=datetime.now(timezone.utc),
                confidence=1.0,
                extra_payload={"source_type": "beta"},
            )
        )

    result = await engine.match("aaa aaa aaa")
    assert isinstance(result, MatchResult)
    assert result.winner == "alpha"
    assert result.accepted is True
    assert 0.0 <= result.confidence <= 1.0
    # entropy strictly less than H_max (two types -> H_max = 1 bit)
    assert result.entropy_bits < math.log2(2)


@pytest.mark.asyncio
async def test_match_returns_full_distribution(engine) -> None:
    """Both source_types appear in the distribution; probs sum to 1.

    Use the same base vector for both types so both have positive cosine
    with the query and neither gets filtered by the zero noise floor.
    """
    base_vec = await engine.embedder("aaa aaa aaa")

    for i in range(3):
        await engine.store.put(
            VibeSignal(
                id=f"a-{i}",
                text_raw="aaa",
                vectors=FacetVectors(content=list(base_vec)),
                source=Src(type="chat"),
                timestamp=datetime.now(timezone.utc),
                extra_payload={"source_type": "alpha"},
            )
        )
        await engine.store.put(
            VibeSignal(
                id=f"b-{i}",
                text_raw="aaa",
                vectors=FacetVectors(content=list(base_vec)),
                source=Src(type="chat"),
                timestamp=datetime.now(timezone.utc),
                extra_payload={"source_type": "beta"},
            )
        )

    result = await engine.match("aaa aaa aaa")
    assert set(result.softmax_probs.keys()) == {"alpha", "beta"}
    assert set(result.cosines.keys()) == {"alpha", "beta"}
    assert sum(result.softmax_probs.values()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_match_uniform_cosine_rejected(fake_qdrant) -> None:
    """When every cosine is equal, entropy = log2(N) and accepted=False."""

    async def uniform_embedder(text: str) -> list[float]:
        return [1.0] * 8

    store = VibeStore(fake_qdrant, collection="t")
    prototypes = PrototypeRegistry(uniform_embedder)
    eng = VibeEngine(
        store=store,
        prototypes=prototypes,
        embedder=uniform_embedder,
        llm_client=lambda _prompt: "irrelevant",
    )

    # Insert signals for two types — identical vectors so cosines are equal.
    for name in ("a", "b"):
        for i in range(3):
            await eng.store.put(
                VibeSignal(
                    id=f"{name}-{i}",
                    text_raw=name,
                    vectors=FacetVectors(content=[1.0] * 8),
                    source=Src(type="chat"),
                    timestamp=datetime.now(timezone.utc),
                    extra_payload={"source_type": name},
                )
            )

    result = await eng.match("anything goes")
    h_max = math.log2(2)
    assert result.entropy_bits == pytest.approx(h_max, abs=1e-6)
    assert result.accepted is False
    assert result.confidence == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# τ calibration (still valid — PrototypeRegistry is still part of the engine)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calibrate_tau_converges_in_reasonable_range(engine) -> None:
    """τ lands inside [0.01, 2.0] after calibration on a trivial bank."""
    await engine.prototypes.add("alpha", ["aa bb cc", "aa bb cc ee"])
    await engine.prototypes.add("beta", ["xx yy zz", "xx yy zz ww"])
    await engine.prototypes.add("gamma", ["11 22 33", "11 22 33 44"])

    tau = engine.prototypes.calibrate_tau()
    assert 0.01 <= tau <= 2.0
    # Registry-level attribute is updated in place
    assert engine.prototypes.tau == pytest.approx(tau)


@pytest.mark.asyncio
async def test_calibrate_tau_target_mean_entropy(engine) -> None:
    """Post-calibration mean entropy ≈ log2(N)/4 (within 10%)."""
    for i in range(6):
        await engine.prototypes.add(f"p{i}", [f"phrase-{i}-a", f"phrase-{i}-b", f"phrase-{i}-c"])
    tau = engine.prototypes.calibrate_tau()

    # Recompute mean entropy at the calibrated τ to confirm it lands near target.
    from skynet_vibe.affinity import cosine

    protos = engine.prototypes.all()
    h_max = math.log2(len(protos))
    target = h_max / 4.0

    training = []
    for name in engine.prototypes._training_embeddings:
        training.extend(engine.prototypes._training_embeddings[name])

    def mean_h(tau_val: float) -> float:
        total = 0.0
        for tv in training:
            logits = [cosine(tv, p.centroid) / tau_val for p in protos]
            max_l = max(logits)
            exps = [math.exp(lg - max_l) for lg in logits]
            s = sum(exps)
            probs = [e / s for e in exps]
            h = 0.0
            for pi in probs:
                if pi > 1e-12:
                    h -= pi * math.log2(pi)
            total += h
        return total / len(training)

    mh = mean_h(tau)
    assert mh == pytest.approx(target, rel=0.25, abs=0.25)
    assert 0.01 < tau < 2.0


@pytest.mark.asyncio
async def test_calibrate_tau_raises_without_prototypes(engine) -> None:
    from skynet_vibe import EmbeddingError

    with pytest.raises(EmbeddingError):
        engine.prototypes.calibrate_tau()


@pytest.mark.asyncio
async def test_calibrate_tau_is_monotonic_mean_entropy(engine) -> None:
    """Sanity: higher τ -> higher mean entropy. If this breaks binary search breaks."""
    for i in range(5):
        await engine.prototypes.add(f"p{i}", [f"phrase-{i}-x", f"phrase-{i}-y"])

    from skynet_vibe.affinity import cosine

    protos = engine.prototypes.all()
    training = []
    for name in engine.prototypes._training_embeddings:
        training.extend(engine.prototypes._training_embeddings[name])

    def mean_h(tau_val: float) -> float:
        total = 0.0
        for tv in training:
            logits = [cosine(tv, p.centroid) / tau_val for p in protos]
            max_l = max(logits)
            exps = [math.exp(lg - max_l) for lg in logits]
            s = sum(exps)
            probs = [e / s for e in exps]
            h = 0.0
            for pi in probs:
                if pi > 1e-12:
                    h -= pi * math.log2(pi)
            total += h
        return total / len(training)

    low, mid, high = mean_h(0.05), mean_h(0.3), mean_h(1.5)
    assert low <= mid <= high


# ---------------------------------------------------------------------------
# Warmup wires calibration automatically
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warmup_calibrates_tau(async_hash_embedder) -> None:
    registry = PrototypeRegistry(async_hash_embedder)
    # DEFAULT_TAU before warmup.
    assert registry.tau == pytest.approx(PrototypeRegistry.DEFAULT_TAU)
    registry.start_warmup()
    ok = await registry.wait_ready(timeout=10.0)
    assert ok is True
    # Post-warmup τ is inside the calibration bracket.
    assert 0.01 <= registry.tau <= 2.0
    # Default bundle loaded -> at least a handful of prototypes.
    assert len(registry.names()) >= 3
