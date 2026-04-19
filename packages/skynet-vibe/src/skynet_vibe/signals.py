"""Data schema for vibe signals.

Key design decision: no ``domain`` field. Domain membership is computed
run-time via cosine similarity to prototype centroids (see
:mod:`skynet_vibe.prototypes`). A single signal can simultaneously
contribute to ``music``, ``movies``, ``photography`` etc. with naturally
different weights.

Facet vectors (``content``, ``context``, ``user_state``) decouple the
*what* from the *when/where/how* — the core content vector is required;
context and user-state vectors are optional and used to shape weighting
at retrieval time.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class FacetVectors:
    """Optional multi-facet embedding bundle attached to a signal.

    * ``content`` is the embedding of ``text_raw`` (required, typically 512-d
      Matryoshka-truncated).
    * ``context`` is the embedding of the surrounding conversation / wiki page /
      batch context that produced the signal (optional).
    * ``user_state`` is the embedding of a short description of the user's
      current state when the signal was captured -- mood, time-of-day, location,
      physiological cue (optional).
    """

    content: list[float]
    context: list[float] | None = None
    user_state: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": list(self.content),
            "context": list(self.context) if self.context is not None else None,
            "user_state": list(self.user_state) if self.user_state is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FacetVectors:
        return cls(
            content=list(data["content"]),
            context=list(data["context"]) if data.get("context") is not None else None,
            user_state=list(data["user_state"]) if data.get("user_state") is not None else None,
        )


@dataclass
class Source:
    """Provenance of a signal.

    ``type`` is one of ``chat``, ``reaction``, ``wiki``, ``consumption``,
    ``dag``, ``implicit``. Additional fields pin the exact origin but are
    optional depending on the source type.
    """

    type: str
    room_id: str | None = None
    path: str | None = None
    dag_id: str | None = None
    agent_inferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Source:
        return cls(
            type=data["type"],
            room_id=data.get("room_id"),
            path=data.get("path"),
            dag_id=data.get("dag_id"),
            agent_inferred=bool(data.get("agent_inferred", False)),
        )


@dataclass
class VibeSignal:
    """A persistent preference/opinion/context signal.

    Deliberately *no* ``domain`` field -- see module docstring.
    ``confidence`` is the capture-time certainty of the extractor.
    ``linked_rec_id`` ties a feedback signal back to the recommendation that
    elicited it, letting the engine close the loop.
    """

    id: str
    text_raw: str
    vectors: FacetVectors
    source: Source
    timestamp: datetime
    confidence: float = 1.0
    linked_rec_id: str | None = None
    extra_payload: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text_raw": self.text_raw,
            "vectors": self.vectors.to_dict(),
            "source": self.source.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "confidence": float(self.confidence),
            "linked_rec_id": self.linked_rec_id,
            "extra_payload": dict(self.extra_payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VibeSignal:
        ts_raw = data["timestamp"]
        if isinstance(ts_raw, datetime):
            ts = ts_raw
        else:
            ts = datetime.fromisoformat(ts_raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return cls(
            id=data["id"],
            text_raw=data["text_raw"],
            vectors=FacetVectors.from_dict(data["vectors"]),
            source=Source.from_dict(data["source"]),
            timestamp=ts,
            confidence=float(data.get("confidence", 1.0)),
            linked_rec_id=data.get("linked_rec_id"),
            extra_payload=dict(data.get("extra_payload", {})),
        )
