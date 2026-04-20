"""skynet-capture — shared ingest helpers.

Public API:

- ``MusicCapture`` — atomic write path for music listening events.
- ``MoviesCapture`` — atomic write path for movie watching events.
- ``looks_like_integrity_error`` — shared uniqueness-violation detector.
- ``extract_consumption`` — LLM-backed multi-item extraction.
- ``PoolLike`` — Protocol type accepted by all write helpers.
"""

from skynet_capture.common.consumption_extractor import extract_consumption
from skynet_capture.common.errors import looks_like_integrity_error
from skynet_capture.common.pg import PoolLike
from skynet_capture.movies import MoviesCapture
from skynet_capture.music import MusicCapture

__all__ = [
    "MusicCapture",
    "MoviesCapture",
    "extract_consumption",
    "looks_like_integrity_error",
    "PoolLike",
]
