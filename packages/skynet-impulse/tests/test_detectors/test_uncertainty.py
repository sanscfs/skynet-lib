"""Uncertainty-sampling detector (entropy-based classifier query)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from skynet_impulse.detectors import UncertaintySamplingDetector


class FakeClassifier:
    def __init__(self, probas):
        self._probas = np.asarray(probas, dtype=float)

    def predict_proba(self, X):
        return self._probas


def test_requires_predict_proba():
    class Bad:
        pass

    with pytest.raises(TypeError):
        UncertaintySamplingDetector(Bad())


def test_threshold_range():
    cli = FakeClassifier([[0.5, 0.5]])
    with pytest.raises(ValueError):
        UncertaintySamplingDetector(cli, uncertainty_threshold=0.0)
    with pytest.raises(ValueError):
        UncertaintySamplingDetector(cli, uncertainty_threshold=1.5)


@pytest.mark.asyncio
async def test_high_entropy_fires():
    # 50/50 -> entropy = 1.0 bit = max. Normalized = 1.0.
    cli = FakeClassifier([[0.5, 0.5]])
    det = UncertaintySamplingDetector(cli, uncertainty_threshold=0.6)
    sigs = await det.detect(X=[[0, 0, 0]], signal_ids=["row-0"])
    assert len(sigs) == 1
    assert sigs[0].payload["detector"] == "uncertainty"
    assert sigs[0].payload["entropy_bits"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_low_entropy_skips():
    # 95/5 -> entropy very low.
    cli = FakeClassifier([[0.95, 0.05]])
    det = UncertaintySamplingDetector(cli, uncertainty_threshold=0.6)
    sigs = await det.detect(X=[[0]], signal_ids=["r0"])
    assert sigs == []


@pytest.mark.asyncio
async def test_anchor_prefix_applied():
    cli = FakeClassifier([[0.5, 0.5]])
    det = UncertaintySamplingDetector(
        cli, uncertainty_threshold=0.6, anchor_prefix="movies:title:",
    )
    sigs = await det.detect(X=[[0]], signal_ids=["Oppenheimer"])
    assert sigs[0].anchor == "movies:title:Oppenheimer"


@pytest.mark.asyncio
async def test_misaligned_signal_ids_raises():
    cli = FakeClassifier([[0.5, 0.5], [0.5, 0.5]])
    det = UncertaintySamplingDetector(cli, uncertainty_threshold=0.6)
    with pytest.raises(ValueError, match="signal_ids"):
        await det.detect(X=[[0], [0]], signal_ids=["only-one"])


@pytest.mark.asyncio
async def test_multiclass_entropy():
    # Uniform 3-class -> entropy = log2(3) ~ 1.585 bits, max = 1.585 -> normalized=1.0.
    cli = FakeClassifier([[1/3, 1/3, 1/3]])
    det = UncertaintySamplingDetector(cli, uncertainty_threshold=0.6)
    sigs = await det.detect(X=[[0]], signal_ids=["x"])
    assert sigs[0].payload["entropy_bits"] == pytest.approx(math.log(3) / math.log(2), abs=1e-3)


@pytest.mark.asyncio
async def test_requires_2d_output():
    cli = FakeClassifier([0.5, 0.5])  # 1-D
    det = UncertaintySamplingDetector(cli, uncertainty_threshold=0.6)
    with pytest.raises(ValueError, match="2-D"):
        await det.detect(X=[0], signal_ids=["x"])
