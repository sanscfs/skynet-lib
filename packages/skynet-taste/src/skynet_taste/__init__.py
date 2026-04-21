"""Skynet Taste — cross-domain mood + preference scoring.

Usage::

    from skynet_taste import get_mood_vector, get_preference_vector, rank_candidates

    mood = await get_mood_vector(store, embedder, query="something intense")
    pref = await get_preference_vector(store, embedder, source_type="music_review")
    ranked = await rank_candidates(candidates, mood.vector, embedder, pref, beta=0.5)
"""

from skynet_taste.absorb import absorb_correction
from skynet_taste.models import MoodResult, TasteScore
from skynet_taste.mood import get_mood_vector
from skynet_taste.preference import get_preference_vector
from skynet_taste.scorer import rank_candidates, score_candidate

__all__ = [
    "MoodResult",
    "TasteScore",
    "get_mood_vector",
    "get_preference_vector",
    "score_candidate",
    "rank_candidates",
    "absorb_correction",
]
