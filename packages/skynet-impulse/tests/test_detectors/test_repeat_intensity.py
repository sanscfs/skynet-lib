"""Poisson repeat-intensity detector."""

from __future__ import annotations

import pytest
from skynet_impulse.detectors import PoissonRepeatDetector


@pytest.mark.asyncio
async def test_below_min_history_never_fires():
    det = PoissonRepeatDetector(lambda_baseline=2.0, min_historical_events=10)
    sigs = await det.detect(anchor_name="Autechre", observed_count=50, historical_count=2)
    assert sigs == []


@pytest.mark.asyncio
async def test_high_observed_count_fires():
    det = PoissonRepeatDetector(
        lambda_baseline=1.0,
        p_threshold=0.05,
        anchor_prefix="music:artist:",
        min_historical_events=1,
    )
    sigs = await det.detect(anchor_name="X", observed_count=10, historical_count=100)
    assert len(sigs) == 1
    assert sigs[0].anchor == "music:artist:X"
    assert sigs[0].kind == "novelty"
    assert sigs[0].payload["detector"] == "repeat_intensity"
    assert 0.0 <= sigs[0].salience <= 1.0


@pytest.mark.asyncio
async def test_baseline_observation_does_not_fire():
    det = PoissonRepeatDetector(lambda_baseline=5.0, p_threshold=0.05, min_historical_events=1)
    sigs = await det.detect(anchor_name="x", observed_count=5, historical_count=50)
    # Observed = expected; tail >> 0.05.
    assert sigs == []


@pytest.mark.asyncio
async def test_zero_observed_skips():
    det = PoissonRepeatDetector(lambda_baseline=1.0, min_historical_events=1)
    sigs = await det.detect(anchor_name="x", observed_count=0, historical_count=50)
    assert sigs == []


def test_init_validates():
    with pytest.raises(ValueError):
        PoissonRepeatDetector(lambda_baseline=-1.0)
    with pytest.raises(ValueError):
        PoissonRepeatDetector(lambda_baseline=1.0, p_threshold=0.0)
    with pytest.raises(ValueError):
        PoissonRepeatDetector(lambda_baseline=1.0, p_threshold=1.5)


def test_update_baseline():
    det = PoissonRepeatDetector(lambda_baseline=1.0)
    det.update_baseline(3.7)
    assert det.lambda_baseline == 3.7
    with pytest.raises(ValueError):
        det.update_baseline(-1)
