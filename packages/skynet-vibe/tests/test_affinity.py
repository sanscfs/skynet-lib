"""Tests for cosine / time_decay / signal_weight."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from skynet_vibe import FacetVectors, Source, VibeSignal, cosine, signal_weight, time_decay
from skynet_vibe.affinity import SOURCE_TRUST


def test_cosine_known() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_dimension_mismatch_returns_zero() -> None:
    assert cosine([1.0, 0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_zero_norm_returns_zero() -> None:
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_time_decay_half_life() -> None:
    now = datetime(2026, 4, 19, tzinfo=timezone.utc)
    ts = now - timedelta(days=45)
    assert time_decay(ts, now, 45.0) == pytest.approx(0.5, rel=1e-6)


def test_time_decay_double_half_life() -> None:
    now = datetime(2026, 4, 19, tzinfo=timezone.utc)
    ts = now - timedelta(days=90)
    assert time_decay(ts, now, 45.0) == pytest.approx(0.25, rel=1e-6)


def test_time_decay_now_is_one() -> None:
    now = datetime(2026, 4, 19, tzinfo=timezone.utc)
    assert time_decay(now, now, 45.0) == 1.0


def test_time_decay_zero_half_life() -> None:
    now = datetime(2026, 4, 19, tzinfo=timezone.utc)
    ts = now - timedelta(days=10)
    assert time_decay(ts, now, 0) == 1.0


def test_signal_weight_composite() -> None:
    now = datetime(2026, 4, 19, tzinfo=timezone.utc)
    ts = now - timedelta(days=45)  # half life -> decay 0.5
    vec = [1.0, 0.0]
    proto = [1.0, 0.0]  # cosine 1
    context = [1.0, 0.0]  # cosine 1 -> (1 + 0.5 * 1) = 1.5
    sig = VibeSignal(
        id="x",
        text_raw="t",
        vectors=FacetVectors(content=vec),
        source=Source(type="chat"),
        timestamp=ts,
        confidence=0.8,
    )
    w = signal_weight(
        sig,
        prototype_centroid=proto,
        context_vector=context,
        now=now,
        half_life_days=45.0,
        context_alpha=0.5,
    )
    # 0.8 * 1.0 (chat) * 0.5 (decay) * 1.0 (proto) * 1.5 (context) = 0.6
    assert w == pytest.approx(0.6, rel=1e-6)


def test_signal_weight_unknown_source_falls_back() -> None:
    now = datetime(2026, 4, 19, tzinfo=timezone.utc)
    sig = VibeSignal(
        id="x",
        text_raw="t",
        vectors=FacetVectors(content=[1.0, 0.0]),
        source=Source(type="mystery"),
        timestamp=now,
    )
    w = signal_weight(sig, prototype_centroid=None, context_vector=None, now=now)
    # no decay (ts == now), no proto/context -> 1.0 conf * 0.5 default trust
    assert w == pytest.approx(0.5)


def test_signal_weight_no_prototype_or_context() -> None:
    now = datetime(2026, 4, 19, tzinfo=timezone.utc)
    sig = VibeSignal(
        id="x",
        text_raw="t",
        vectors=FacetVectors(content=[1.0, 0.0]),
        source=Source(type="chat"),
        timestamp=now,
        confidence=0.5,
    )
    w = signal_weight(sig, prototype_centroid=None, context_vector=None, now=now)
    assert w == pytest.approx(0.5 * SOURCE_TRUST["chat"])


def test_signal_weight_negative_cosine_clipped() -> None:
    now = datetime(2026, 4, 19, tzinfo=timezone.utc)
    sig = VibeSignal(
        id="x",
        text_raw="t",
        vectors=FacetVectors(content=[1.0, 0.0]),
        source=Source(type="chat"),
        timestamp=now,
    )
    w = signal_weight(
        sig, prototype_centroid=[-1.0, 0.0], context_vector=None, now=now
    )
    # negative cosine is clipped to 0 -> overall weight 0
    assert w == 0.0


def test_source_trust_table_complete() -> None:
    expected_keys = {"chat", "reaction", "wiki", "consumption", "dag", "implicit"}
    assert expected_keys == set(SOURCE_TRUST.keys())
    for v in SOURCE_TRUST.values():
        assert 0.0 < v <= 1.0
    # avoid unused-math flake
    assert math.isfinite(1.0)
