"""skynet-capture — shared ingest helpers.

Public API:

- ``MusicCapture`` — atomic write path for music listening events.
- ``MoviesCapture`` — atomic write path for movie watching events.
- ``looks_like_integrity_error`` — shared uniqueness-violation detector.
- ``extract_consumption`` — LLM-backed multi-item extraction.
- ``PoolLike`` — Protocol type accepted by all write helpers.
- ``ActivityIdentifier`` / ``IdentifyResult`` — LLM-scored entity identification.
- ``ActivitySource`` / ``Candidate`` — search protocol + result type.
- ``MusicBrainzSource`` — MusicBrainz search adapter.
- ``PendingCapture`` / ``PendingStore`` / ``timeout_watcher`` — pending state machine.
- ``resolve_candidate_reply`` — light-LLM thread-reply resolver.
"""

from skynet_capture.common.consumption_extractor import extract_consumption
from skynet_capture.common.errors import looks_like_integrity_error
from skynet_capture.common.pg import PoolLike
from skynet_capture.identifier import ActivityIdentifier, IdentifyResult, LLMCaller
from skynet_capture.movies import MoviesCapture
from skynet_capture.music import MusicCapture
from skynet_capture.pending import PendingCapture, PendingStore, timeout_watcher
from skynet_capture.resolver import resolve_candidate_reply
from skynet_capture.sources.base import ActivitySource, Candidate
from skynet_capture.sources.musicbrainz import MusicBrainzSource

__all__ = [
    "ActivityIdentifier",
    "ActivitySource",
    "Candidate",
    "IdentifyResult",
    "LLMCaller",
    "MusicBrainzSource",
    "MusicCapture",
    "MoviesCapture",
    "PendingCapture",
    "PendingStore",
    "extract_consumption",
    "looks_like_integrity_error",
    "PoolLike",
    "resolve_candidate_reply",
    "timeout_watcher",
]
