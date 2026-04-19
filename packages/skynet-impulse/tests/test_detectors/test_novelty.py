"""CentroidNoveltyDetector math + cold-start behavior."""

from __future__ import annotations

import pytest
from skynet_impulse.detectors import CentroidNoveltyDetector


def test_rejects_zero_centroid():
    with pytest.raises(ValueError, match="zero norm"):
        CentroidNoveltyDetector(centroid=[0.0, 0.0, 0.0])


def test_rejects_non_1d_centroid():
    with pytest.raises(ValueError, match="1-D"):
        CentroidNoveltyDetector(centroid=[])


@pytest.mark.asyncio
async def test_detects_vector_far_from_centroid():
    det = CentroidNoveltyDetector(
        centroid=[1.0, 0.0, 0.0],
        threshold_cos=0.5,
        anchor_prefix="music:artist:",
        min_events_for_stable=1,
    )
    # Vector orthogonal -> cos_sim = 0 -> below 0.5 threshold.
    sigs = await det.detect(vector=[0.0, 1.0, 0.0], signal_id="Autechre")
    assert len(sigs) == 1
    assert sigs[0].anchor == "music:artist:Autechre"
    assert sigs[0].kind == "novelty"
    assert sigs[0].salience == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_skips_vector_close_to_centroid():
    det = CentroidNoveltyDetector(
        centroid=[1.0, 0.0],
        threshold_cos=0.5,
        min_events_for_stable=1,
    )
    sigs = await det.detect(vector=[0.9, 0.1], signal_id="x")
    assert sigs == []


@pytest.mark.asyncio
async def test_cold_start_damps_salience():
    det = CentroidNoveltyDetector(
        centroid=[1.0, 0.0, 0.0],
        threshold_cos=0.9,
        min_events_for_stable=100,
        cold_start_salience_factor=0.3,
    )
    # Completely new direction -> nov = 1.0, dampened to 0.3.
    sigs = await det.detect(vector=[0.0, 0.0, 1.0], signal_id="x")
    assert sigs[0].salience == pytest.approx(0.3, rel=1e-3)


@pytest.mark.asyncio
async def test_update_centroid_enforces_shape():
    det = CentroidNoveltyDetector(centroid=[1.0, 0.0])
    with pytest.raises(ValueError, match="shape"):
        det.update_centroid([1.0, 0.0, 0.0])


def test_novelty_is_one_minus_cosine():
    det = CentroidNoveltyDetector(centroid=[1.0, 0.0])
    # Identical vector -> novelty = 0.
    assert det.novelty([2.0, 0.0]) == pytest.approx(0.0)
    # Opposite vector -> novelty = 2.
    assert det.novelty([-1.0, 0.0]) == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_seen_events_counter_grows():
    det = CentroidNoveltyDetector(
        centroid=[1.0, 0.0],
        threshold_cos=0.9,
        min_events_for_stable=3,
        cold_start_salience_factor=0.1,
    )
    # After 3 events we're out of cold start and full salience applies.
    sigs = []
    for i in range(5):
        sigs.extend(await det.detect(vector=[0.0, 1.0], signal_id=f"id-{i}"))
    # First 2 detections are still within cold start (seen<min), 3rd+ are full.
    dampened = sigs[0].salience
    full = sigs[-1].salience
    assert full > dampened
