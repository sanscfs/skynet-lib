"""Classifier-uncertainty detector.

Detector #3 from the research doc. Given a binary / multiclass classifier
that predicts something interesting (will the user like this artist?
Is this movie "slow cinema" or "thriller"?), emit a signal on samples
with high entropy -- the model is torn, so the user's reply would teach
the model a lot.

Classifier is caller-provided -- any duck-type with a
``predict_proba(X) -> np.ndarray of shape (n_samples, n_classes)`` works.
Sklearn is the natural source but anything that quacks fits. The only hard
dep here is numpy (for the entropy calc).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from ..signals import Signal


class _ProbaClassifier(Protocol):
    """Any object exposing ``predict_proba``."""

    def predict_proba(self, X: Any) -> Any: ...


def _shannon_entropy(probs: Sequence[float]) -> float:
    """Base-2 entropy of a discrete distribution. Returns 0 on degenerate inputs.

    Uses natural log internally then converts; handles zero-probability
    entries with the convention ``0 * log(0) = 0``.
    """
    total = 0.0
    arr = np.asarray(probs, dtype=float)
    if arr.size == 0:
        return 0.0
    # Normalize if numerics drift (caller may pass un-normalized logits after softmax).
    s = float(arr.sum())
    if s <= 0:
        return 0.0
    arr = arr / s
    for p in arr:
        if p > 0:
            total -= float(p) * math.log(float(p))
    # Convert nats -> bits.
    return total / math.log(2)


def _max_entropy(n_classes: int) -> float:
    """Entropy of the uniform distribution over ``n_classes`` (in bits)."""
    if n_classes <= 1:
        return 0.0
    return math.log(n_classes) / math.log(2)


class UncertaintySamplingDetector:
    """Emit a signal when the classifier's entropy exceeds a threshold.

    Parameters
    ----------
    classifier:
        Any object with ``predict_proba(X) -> array-like [n, k]``.
    uncertainty_threshold:
        Fire when ``entropy / max_entropy(n_classes) > threshold``. Values
        in [0, 1]; 0.6 is the research default -- roughly "the model isn't
        >= 70% confident in its top class".
    anchor_prefix:
        Optional prefix for emitted anchors, e.g. ``"music:artist:"``.
    """

    domain = "uncertainty"

    def __init__(
        self,
        classifier: _ProbaClassifier,
        *,
        uncertainty_threshold: float = 0.6,
        anchor_prefix: str = "",
        source: str = "analyzer",
    ):
        if not hasattr(classifier, "predict_proba"):
            raise TypeError(
                "UncertaintySamplingDetector requires a classifier with "
                ".predict_proba(); got "
                f"{type(classifier).__name__}"
            )
        if not 0.0 < uncertainty_threshold <= 1.0:
            raise ValueError("uncertainty_threshold must be in (0, 1]")
        self._classifier = classifier
        self._threshold = uncertainty_threshold
        self._anchor_prefix = anchor_prefix
        self._source = source

    @property
    def anchor_prefix(self) -> str:
        return self._anchor_prefix

    async def detect(
        self,
        *,
        X: Any,
        signal_ids: Sequence[str],
        payload: Optional[dict] = None,
        source: Optional[str] = None,
    ) -> list[Signal]:
        """Classify ``X`` then emit one signal per sample above the threshold.

        ``signal_ids`` aligns 1:1 with rows of ``X`` -- used to build the
        anchor. Raise on mismatch rather than silently truncating, because
        mixed-up IDs would mis-anchor every signal.
        """
        probas = np.asarray(self._classifier.predict_proba(X), dtype=float)
        if probas.ndim != 2:
            raise ValueError(
                f"classifier.predict_proba must return 2-D; got shape {probas.shape}"
            )
        if probas.shape[0] != len(signal_ids):
            raise ValueError(
                f"signal_ids length {len(signal_ids)} != predict_proba rows {probas.shape[0]}"
            )
        n_classes = probas.shape[1]
        max_h = _max_entropy(n_classes)
        if max_h == 0.0:
            return []

        signals: list[Signal] = []
        for row_idx, row in enumerate(probas):
            h = _shannon_entropy(row)
            normalized = h / max_h
            if normalized <= self._threshold:
                continue
            sid = signal_ids[row_idx]
            anchor = f"{self._anchor_prefix}{sid}"
            signals.append(
                Signal(
                    kind="novelty",
                    source=source or self._source,
                    salience=min(1.0, normalized),
                    anchor=anchor,
                    payload={
                        "detector": "uncertainty",
                        "entropy_bits": round(h, 4),
                        "max_entropy_bits": round(max_h, 4),
                        "probs": [round(float(p), 4) for p in row],
                        "signal_id": sid,
                        **(payload or {}),
                    },
                )
            )
        return signals


__all__ = ["UncertaintySamplingDetector"]
