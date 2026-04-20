from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass
class Candidate:
    source_id: str
    title: str
    subtitle: str  # artist name / director / author
    year: int | None
    type: Literal["artist", "album", "track", "movie", "book"]
    url: str
    cover_url: str | None = None
    raw: dict = field(default_factory=dict, repr=False)


class ActivitySource(Protocol):
    async def search(self, query: str, *, year: int | None = None) -> list[Candidate]: ...


__all__ = ["ActivitySource", "Candidate"]
