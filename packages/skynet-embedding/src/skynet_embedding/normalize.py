"""Vector normalization and Matryoshka truncation utilities."""

from __future__ import annotations

import math


def l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector. Returns zero vector if norm is 0."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def truncate_and_normalize(vec: list[float], dim: int) -> list[float]:
    """Truncate to `dim` dimensions (Matryoshka) and L2-normalize.

    Used with models like nomic-embed-text that support Matryoshka
    representations -- you can truncate the full embedding to a
    smaller dimension and still get good similarity search results.
    """
    if len(vec) < dim:
        vec = vec + [0.0] * (dim - len(vec))
    return l2_normalize(vec[:dim])
