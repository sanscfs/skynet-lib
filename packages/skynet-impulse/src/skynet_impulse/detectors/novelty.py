"""Centroid-distance novelty detector.

Detector #1 from the research doc. Maintains a running centroid of the
domain's user_profile_raw vectors; emits a ``novelty`` Signal whenever a
new vector's cosine similarity to the centroid falls below a threshold.

Typical usage::

    detector = CentroidNoveltyDetector(
        centroid=np.array([...]),              # from skynet-profile-synthesis
        threshold_cos=0.4,
        anchor_prefix="music:artist:",
    )
    for new_artist_vec, artist_name in recent_discoveries:
        signals = await detector.detect(
            vector=new_artist_vec,
            signal_id=artist_name,
            source="collectors",
        )
        for sig in signals:
            emit_signal(sig.kind, sig.source, sig.salience,
                        anchor=sig.anchor, payload=sig.payload)

The detector does NOT call ``emit_signal`` itself -- it returns the Signal
list so the caller can filter, batch, or route. This matches the design
note in the research that detectors should be deterministic pure functions
over data + config, nothing more.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..signals import Signal


class CentroidNoveltyDetector:
    """Outlier-from-centroid novelty detector.

    Parameters
    ----------
    centroid:
        The reference vector to compare against. Typically the mean of the
        domain's user_profile_raw embeddings (recomputed nightly by a DAG).
    threshold_cos:
        Signal fires when ``cosine_similarity(vec, centroid) < threshold_cos``.
        Default 0.4 empirically matches the research's "first few discoveries
        are already interesting enough" note; tune per-domain.
    anchor_prefix:
        Prepended to each emitted signal's anchor, e.g. ``"music:artist:"``.
    min_events_for_stable:
        Below this, boost salience by ``cold_start_factor`` instead of
        dampening -- the first discoveries ARE interesting, and research
        says they should fire sparingly rather than be suppressed entirely.
    """

    domain = "novelty"

    def __init__(
        self,
        centroid: Sequence[float],
        *,
        threshold_cos: float = 0.4,
        anchor_prefix: str = "",
        min_events_for_stable: int = 50,
        cold_start_salience_factor: float = 0.3,
        source: str = "analyzer",
    ):
        arr = np.asarray(centroid, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("centroid must be a non-empty 1-D vector")
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            raise ValueError("centroid has zero norm; cannot compute cosine similarity")
        self._centroid_unit = arr / norm
        self._threshold = threshold_cos
        self._anchor_prefix = anchor_prefix
        self._min_events = min_events_for_stable
        self._cold_start_factor = cold_start_salience_factor
        self._source = source
        self._seen_events = 0

    @property
    def anchor_prefix(self) -> str:
        return self._anchor_prefix

    def update_centroid(self, new_centroid: Sequence[float]) -> None:
        """Swap in a freshly-computed centroid. Used by nightly refresh jobs."""
        arr = np.asarray(new_centroid, dtype=float)
        if arr.ndim != 1 or arr.size != self._centroid_unit.size:
            raise ValueError("new centroid has wrong shape")
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            raise ValueError("new centroid has zero norm")
        self._centroid_unit = arr / norm

    def mark_seen(self, n: int = 1) -> None:
        """Increment the seen-events counter; callers use this to grow out of cold-start."""
        self._seen_events += max(0, int(n))

    def novelty(self, vector: Sequence[float]) -> float:
        """Cosine novelty = ``1 - cos_sim``. Exposed for debugging / tests."""
        arr = np.asarray(vector, dtype=float)
        if arr.ndim != 1 or arr.size != self._centroid_unit.size:
            raise ValueError("vector dims mismatch centroid")
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return 0.0
        cos_sim = float(np.dot(arr / norm, self._centroid_unit))
        return 1.0 - cos_sim

    async def detect(
        self,
        *,
        vector: Sequence[float],
        signal_id: str,
        source: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> list[Signal]:
        """Return 0 or 1 signals depending on the cosine distance."""
        nov = self.novelty(vector)
        cos_sim = 1.0 - nov
        self._seen_events += 1
        if cos_sim >= self._threshold:
            return []
        # Novelty in [0, 2]; clip to [0, 1] for the Signal contract.
        salience = max(0.0, min(1.0, nov))
        if self._seen_events < self._min_events:
            salience *= self._cold_start_factor
        anchor = f"{self._anchor_prefix}{signal_id}"
        sig = Signal(
            kind="novelty",
            source=source or self._source,
            salience=salience,
            anchor=anchor,
            payload={
                "cos_sim": round(cos_sim, 4),
                "threshold": self._threshold,
                "signal_id": signal_id,
                **(payload or {}),
            },
        )
        return [sig]


__all__ = ["CentroidNoveltyDetector"]
