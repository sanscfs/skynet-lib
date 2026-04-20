"""MusicBrainz search adapter.

Queries /ws/2/release-group and /ws/2/artist, merges and deduplicates
results, returns up to ``limit`` Candidate objects sorted by MB score.

Rate limit: 1 req/s without auth. We track the last request timestamp
and sleep the delta before each call to stay compliant.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from skynet_capture.sources.base import Candidate

log = logging.getLogger("skynet_capture.sources.musicbrainz")

_BASE = "https://musicbrainz.org/ws/2"
_UA = "skynet-music/1.0 (sanscfs@gmail.com)"
_MIN_INTERVAL = 1.1  # seconds between requests


class MusicBrainzSource:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None
        self._last_req: float = 0.0

    async def __aenter__(self) -> "MusicBrainzSource":
        if self._owns_client:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": _UA, "Accept": "application/json"},
                timeout=10.0,
            )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, endpoint: str, params: dict) -> dict:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - self._last_req)
        if wait > 0:
            await asyncio.sleep(wait)
        assert self._client is not None
        resp = await self._client.get(f"{_BASE}/{endpoint}", params={**params, "fmt": "json"})
        self._last_req = time.monotonic()
        resp.raise_for_status()
        return resp.json()

    async def fetch_tracklist(self, release_group_mbid: str) -> list[str]:
        """Return track titles for the primary release of a release group.

        Performs two MB API calls (rate-limited):
          1. ``release-group/{mbid}?inc=releases`` → pick first release
          2. ``release/{release_mbid}?inc=recordings`` → extract track titles
        Returns an empty list on any error.
        """
        try:
            rg_data = await self._get(f"release-group/{release_group_mbid}", {"inc": "releases"})
        except Exception as exc:
            log.debug("fetch_tracklist: release-group lookup failed for %s: %s", release_group_mbid, exc)
            return []

        releases = rg_data.get("releases") or []
        if not releases:
            return []
        release_id = releases[0].get("id") or ""
        if not release_id:
            return []

        try:
            rel_data = await self._get(f"release/{release_id}", {"inc": "recordings"})
        except Exception as exc:
            log.debug("fetch_tracklist: release lookup failed for %s: %s", release_id, exc)
            return []

        titles: list[str] = []
        for medium in rel_data.get("media") or []:
            for track in medium.get("tracks") or []:
                title = (track.get("recording") or {}).get("title") or track.get("title") or ""
                if title:
                    titles.append(title)
        return titles

    async def search(self, query: str, *, year: int | None = None, limit: int = 5) -> list[Candidate]:
        q = query.strip()
        if not q:
            return []

        results: list[Candidate] = []
        seen: set[str] = set()

        async def _add(endpoint: str, key: str, to_candidate: Any) -> None:
            try:
                data = await self._get(endpoint, {"query": q, "limit": limit})
            except Exception as exc:
                log.debug("MB %s search failed: %s", endpoint, exc)
                return
            for item in data.get(key, []):
                score = int(item.get("score", 0))
                if score < 20:
                    continue
                try:
                    cand = to_candidate(item, score)
                except Exception:
                    continue
                if cand.source_id not in seen:
                    seen.add(cand.source_id)
                    results.append(cand)

        await _add("artist", "artists", _artist_to_candidate)
        await _add("release-group", "release-groups", _release_to_candidate)

        results.sort(key=lambda c: int(c.raw.get("score", 0)), reverse=True)
        return results[:limit]


def _artist_to_candidate(item: dict, score: int) -> Candidate:
    mbid = item["id"]
    name = item.get("name") or item.get("sort-name") or mbid
    area = item.get("area", {}).get("name") or ""
    return Candidate(
        source_id=mbid,
        title=name,
        subtitle=area,
        year=None,
        type="artist",
        url=f"https://musicbrainz.org/artist/{mbid}",
        raw={**item, "score": score},
    )


def _release_to_candidate(item: dict, score: int) -> Candidate:
    mbid = item["id"]
    title = item.get("title") or mbid
    credits = item.get("artist-credit") or []
    artist = credits[0].get("name") or "" if credits else ""
    date = item.get("first-release-date") or ""
    year: int | None = None
    if date:
        try:
            year = int(date[:4])
        except ValueError:
            pass
    return Candidate(
        source_id=mbid,
        title=title,
        subtitle=artist,
        year=year,
        type="album",
        url=f"https://musicbrainz.org/release-group/{mbid}",
        raw={**item, "score": score},
    )


__all__ = ["MusicBrainzSource"]
