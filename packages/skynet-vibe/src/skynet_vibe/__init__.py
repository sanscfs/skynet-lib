"""Skynet Vibe -- universal gradient signal system for preference capture and retrieval.

Core idea: signals do NOT carry a domain label. Domain membership is computed at
retrieval time as cosine similarity to domain prototype centroids. One signal can
legitimately contribute to many domains with naturally different weights.

Public entry point: :class:`VibeEngine`.
"""

from __future__ import annotations

__version__ = "2026.4.19"
