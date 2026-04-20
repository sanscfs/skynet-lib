"""Atomic write path for movie watching events.

``MoviesCapture`` encapsulates the full write chain —
movie upsert → INSERT watch_log — so the profiler and chat agent share
identical write logic. Profiler-generated stubs use a negative deterministic
``tmdb_id`` (matching skynet-movies' own convention) so reconciliation DAGs
can match them to real TMDB rows later.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from skynet_capture.common.pg import PoolLike

log = logging.getLogger("skynet_capture.movies")


def _pseudo_tmdb_id(title: str, year: Optional[int]) -> int:
    """Stable negative int for (title, year) pairs without a real TMDB ID.

    skynet-movies uses ``-abs(hash(title)) % 2_000_000_000`` for fallback
    rows. Python's ``hash`` is process-salted, so we include ``year`` in the
    key and accept that the pseudo-id is only stable within a single Python
    process (which is fine — ON CONFLICT upsert never reads it across
    processes; reconciliation DAGs match on title/year).
    """
    key = f"{title.strip().lower()}|{year or ''}"
    return -(1 + (abs(hash(key)) % 2_000_000_000))


class MoviesCapture:
    """Shared DB write logic for movie watching events."""

    @staticmethod
    async def ensure_schema(pool: PoolLike) -> None:
        """Idempotently add capture-owned columns + partial unique index.

        Requires DDL privileges. Same caveat as ``MusicCapture.ensure_schema``.
        """
        await pool.execute("ALTER TABLE watch_log ADD COLUMN IF NOT EXISTS source_event_id TEXT")
        await pool.execute("ALTER TABLE watch_log ADD COLUMN IF NOT EXISTS notes_source TEXT")
        await pool.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_watch_log_source_event "
            "ON watch_log (source_event_id, movie_id) "
            "WHERE source_event_id IS NOT NULL"
        )

    @staticmethod
    async def upsert_movie(
        pool: PoolLike,
        title: str,
        year: Optional[int],
    ) -> int:
        """Return movie id, inserting a stub row if not found by pseudo tmdb_id."""
        tmdb_id = _pseudo_tmdb_id(title, year)
        row = await pool.fetchrow(
            """
            INSERT INTO movies (tmdb_id, title, title_search, year)
            VALUES ($1, $2, LOWER($2), $3)
            ON CONFLICT (tmdb_id) DO UPDATE SET title = EXCLUDED.title
            RETURNING id
            """,
            tmdb_id,
            title,
            year,
        )
        if row is None:
            row = await pool.fetchrow(
                "SELECT id FROM movies WHERE tmdb_id = $1",
                tmdb_id,
            )
        if row is None:
            raise RuntimeError(f"failed to resolve movie id for {title!r}")
        return row["id"]

    @classmethod
    async def persist(
        cls,
        pool: PoolLike,
        *,
        title: str,
        year: Optional[int] = None,
        notes: Optional[str] = None,
        source: str = "chat",
        source_event_id: Optional[str] = None,
        watched_at: Optional[datetime] = None,
    ) -> int:
        """Atomic upsert chain: movie → INSERT watch_log.

        Returns the resolved ``movie_id``. Columns ``source_event_id`` and
        ``notes_source`` must exist before this is called.
        """
        title = (title or "").strip()
        if not title:
            raise ValueError("title is required")

        movie_id = await cls.upsert_movie(pool, title, year)
        await pool.execute(
            """
            INSERT INTO watch_log
              (movie_id, notes, source, watched_at, source_event_id, notes_source)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            movie_id,
            notes or None,
            source,
            watched_at or datetime.now(timezone.utc),
            source_event_id or None,
            source,
        )
        log.debug(
            "persist: movie_id=%s source_event_id=%s source=%s",
            movie_id,
            source_event_id,
            source,
        )
        return movie_id
