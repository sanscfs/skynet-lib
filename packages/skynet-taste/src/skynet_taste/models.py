from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MoodResult:
    """Decay-weighted centroid of recent vibe signals.

    ``vector`` — weighted mean embedding (empty list if no signals found).
    ``confidence`` — mean cosine alignment of contributing signals to the
    centroid. 1.0 = tight cluster (clear mood); 0.0 = diffuse.
    ``age`` — weighted mean missed_opportunities across contributing signals.
    Low = fresh mood (recent signals dominate). High = mood hasn't been
    reinforced recently.
    ``signal_count`` — number of signals that contributed.
    """

    vector: list[float]
    confidence: float
    age: float
    signal_count: int = 0


@dataclass
class TasteScore:
    """Combined mood + preference score for a single candidate."""

    candidate: dict
    mood_score: float
    pref_score: float
    combined: float
