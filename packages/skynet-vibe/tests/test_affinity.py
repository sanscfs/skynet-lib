"""Tests for cosine / signal_weight.

Logical-time decay itself lives in ``skynet_scoring`` and is covered
there; ``skynet_vibe`` only consumes the pre-computed ``decay_factor``
so our tests verify that the multiplicative composition of the other
terms still holds for any decay value the caller passes in.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest
from skynet_vibe import FacetVectors, Source, VibeSignal, cosine, signal_weight
from skynet_vibe.affinity import SOURCE_TRUST


def test_cosine_known() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_dimension_mismatch_returns_zero() -> None:
    assert cosine([1.0, 0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_zero_norm_returns_zero() -> None:
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def _sig(source_type: str = "chat", confidence: float = 1.0) -> VibeSignal:
    return VibeSignal(
        id="x",
        text_raw="t",
        vectors=FacetVectors(content=[1.0, 0.0]),
        source=Source(type=source_type),
        timestamp=datetime(2026, 4, 19, tzinfo=timezone.utc),
        confidence=confidence,
    )


def test_signal_weight_composite() -> None:
    sig = _sig(confidence=0.8)
    proto = [1.0, 0.0]  # cosine 1
    context = [1.0, 0.0]  # cosine 1 -> (1 + 0.5 * 1) = 1.5
    w = signal_weight(
        sig,
        prototype_centroid=proto,
        context_vector=context,
        decay_factor=0.5,
        context_alpha=0.5,
    )
    # 0.8 * 1.0 (chat) * 0.5 (decay) * 1.0 (proto) * 1.5 (context) = 0.6
    assert w == pytest.approx(0.6, rel=1e-6)


def test_signal_weight_default_decay_is_one() -> None:
    """Omitting decay_factor means 'no decay' -- matches logical-time
    behaviour for a signal that has had no missed opportunities yet."""
    sig = _sig()
    w = signal_weight(sig, prototype_centroid=None, context_vector=None)
    # 1.0 conf * 1.0 (chat trust) * 1.0 (default decay) = 1.0
    assert w == pytest.approx(1.0)


def test_signal_weight_unknown_source_falls_back() -> None:
    sig = _sig(source_type="mystery")
    w = signal_weight(sig, prototype_centroid=None, context_vector=None)
    # 1.0 conf * 0.5 default trust = 0.5
    assert w == pytest.approx(0.5)


def test_signal_weight_no_prototype_or_context() -> None:
    sig = _sig(confidence=0.5)
    w = signal_weight(sig, prototype_centroid=None, context_vector=None)
    assert w == pytest.approx(0.5 * SOURCE_TRUST["chat"])


def test_signal_weight_negative_cosine_clipped() -> None:
    sig = _sig()
    w = signal_weight(sig, prototype_centroid=[-1.0, 0.0], context_vector=None)
    # negative cosine is clipped to 0 -> overall weight 0
    assert w == 0.0


def test_signal_weight_negative_decay_clipped() -> None:
    """Decay factor clipped at 0 -- caller bugs shouldn't produce negative weights."""
    sig = _sig()
    w = signal_weight(
        sig,
        prototype_centroid=None,
        context_vector=None,
        decay_factor=-0.5,
    )
    assert w == 0.0


def test_source_trust_table_complete() -> None:
    expected_keys = {"chat", "reaction", "wiki", "consumption", "dag", "implicit"}
    assert expected_keys == set(SOURCE_TRUST.keys())
    for v in SOURCE_TRUST.values():
        assert 0.0 < v <= 1.0
    # avoid unused-math flake
    assert math.isfinite(1.0)
