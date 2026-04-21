"""Pure math helpers — no I/O."""

from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def weighted_centroid(vecs: list[list[float]], weights: list[float]) -> list[float]:
    """L1-normalised weighted mean of ``vecs``."""
    if not vecs or not weights:
        return []
    total_w = sum(weights)
    if total_w <= 0.0:
        return []
    dim = len(vecs[0])
    acc = [0.0] * dim
    for vec, w in zip(vecs, weights):
        if len(vec) != dim or w <= 0.0:
            continue
        for i, x in enumerate(vec):
            acc[i] += w * x
    return [x / total_w for x in acc]


def cosine_alignment(vecs: list[list[float]], centroid: list[float], weights: list[float]) -> float:
    """Weighted mean cosine of each vec to the centroid — cluster tightness proxy."""
    if not vecs or not centroid:
        return 0.0
    total_w = sum(weights)
    if total_w <= 0.0:
        return 0.0
    return sum(w * cosine(v, centroid) for v, w in zip(vecs, weights)) / total_w
