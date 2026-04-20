"""Atomic write path for music listening events.

``MusicCapture`` encapsulates the full upsert chain —
artist → album → track → listen — so callers cannot accidentally skip a step
(which is how the ``track_id NOT NULL`` bug happened in the chat agent).

Design notes:

* All public methods are static/classmethod — no per-instance state.
* ``PoolLike`` is satisfied by both raw asyncpg pools and
  ``skynet_postgres.AsyncPool``; callers pass what they have.
* ``ensure_schema()`` runs idempotent DDL to add the profiler-owned columns
  (``notes``, ``source_event_id``, ``notes_source``) and the partial unique
  index. Long-running consumers (the profiler) call it once on first write.
  Short-lived consumers (chat-agent MCP tools) rely on the columns being
  pre-created by either the init SQL script or a prior profiler run.
* ``persist()`` never calls ``ensure_schema()`` automatically — that would
  fail if the Vault dynamic user lacks DDL privileges.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from skynet_capture.common.pg import PoolLike

log = logging.getLogger("skynet_capture.music")


def _pseudo_yt_id(track_title: str) -> str:
    """Stable pseudo YouTube ID for profiler-owned tracks.

    Real YT IDs are 11-char base64url; the ``profiler:`` prefix makes these
    clearly distinguishable. SHA-1 truncated to 16 hex chars is
    collision-free at skynet-music scale.
    """
    digest = hashlib.sha1(track_title.strip().lower().encode("utf-8")).hexdigest()
    return f"profiler:{digest[:16]}"


class MusicCapture:
    """Shared DB write logic for music listening events."""

    @staticmethod
    async def ensure_schema(pool: PoolLike) -> None:
        """Idempotently add capture-owned columns + partial unique index.

        Requires DDL privileges (table owner or superuser). Callers without
        DDL rights (e.g. Vault dynamic users in skynet-music) must NOT call
        this; they rely on the schema being pre-applied.
        """
        await pool.execute("ALTER TABLE listens ADD COLUMN IF NOT EXISTS notes TEXT")
        await pool.execute("ALTER TABLE listens ADD COLUMN IF NOT EXISTS source_event_id TEXT")
        await pool.execute("ALTER TABLE listens ADD COLUMN IF NOT EXISTS notes_source TEXT")
        await pool.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_listens_source_event "
            "ON listens (source_event_id, track_id) "
            "WHERE source_event_id IS NOT NULL"
        )

    @staticmethod
    async def upsert_artist(pool: PoolLike, name: str) -> int:
        """Return artist id, inserting if missing (case-insensitive lookup)."""
        row = await pool.fetchrow(
            "SELECT id FROM artists WHERE LOWER(name) = LOWER($1) LIMIT 1",
            name,
        )
        if row is not None:
            return row["id"]
        row = await pool.fetchrow(
            "INSERT INTO artists (name) VALUES ($1) RETURNING id",
            name,
        )
        return row["id"]

    @staticmethod
    async def upsert_album(
        pool: PoolLike,
        title: str,
        year: Optional[int],
    ) -> int:
        """Return album id keyed on title (case-insensitive), inserting if missing."""
        row = await pool.fetchrow(
            "SELECT id FROM albums WHERE LOWER(title) = LOWER($1) LIMIT 1",
            title,
        )
        if row is not None:
            return row["id"]
        row = await pool.fetchrow(
            "INSERT INTO albums (title, year) VALUES ($1, $2) RETURNING id",
            title,
            year,
        )
        return row["id"]

    @staticmethod
    async def upsert_track(
        pool: PoolLike,
        title: str,
        album_id: int,
    ) -> int:
        """Return track id keyed on pseudo yt_video_id, inserting if missing."""
        pseudo = _pseudo_yt_id(title)
        row = await pool.fetchrow(
            "SELECT id FROM tracks WHERE yt_video_id = $1",
            pseudo,
        )
        if row is not None:
            return row["id"]
        row = await pool.fetchrow(
            """
            INSERT INTO tracks (title, yt_video_id, album_id)
            VALUES ($1, $2, $3)
            ON CONFLICT (yt_video_id) DO UPDATE SET title = EXCLUDED.title
            RETURNING id
            """,
            title,
            pseudo,
            album_id,
        )
        return row["id"]

    @staticmethod
    async def link_track_artist(
        pool: PoolLike,
        track_id: int,
        artist_id: int,
    ) -> None:
        """Link track to artist in ``track_artists`` (idempotent via ON CONFLICT)."""
        await pool.execute(
            """
            INSERT INTO track_artists (track_id, artist_id, role)
            VALUES ($1, $2, 'primary')
            ON CONFLICT DO NOTHING
            """,
            track_id,
            artist_id,
        )

    @classmethod
    async def persist(
        cls,
        pool: PoolLike,
        *,
        artist: str,
        track: str,
        year: Optional[int] = None,
        notes: Optional[str] = None,
        source: str = "chat",
        source_event_id: Optional[str] = None,
        listened_at: Optional[datetime] = None,
    ) -> int:
        """Atomic upsert chain: artist → album → track → INSERT listen.

        Returns the resolved ``track_id``. The LLM cannot skip steps because
        this method resolves all FKs before the INSERT — there is no path to
        ``INSERT INTO listens`` without a valid ``track_id``.

        Columns ``notes``, ``source_event_id``, and ``notes_source`` must exist
        before this is called; either add them via ``ensure_schema()`` or include
        them in the DB init script.
        """
        track_name = (track or "").strip()
        if not track_name:
            raise ValueError("track name is required")
        artist_name = (artist or "").strip()

        artist_id: Optional[int] = None
        if artist_name:
            artist_id = await cls.upsert_artist(pool, artist_name)

        album_id = await cls.upsert_album(pool, track_name, year)
        track_id = await cls.upsert_track(pool, track_name, album_id)
        if artist_id is not None:
            await cls.link_track_artist(pool, track_id, artist_id)

        await pool.execute(
            """
            INSERT INTO listens
              (track_id, listened_at, source, notes, source_event_id, notes_source)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            track_id,
            listened_at or datetime.now(timezone.utc),
            source,
            notes or None,
            source_event_id or None,
            source,
        )
        log.debug(
            "persist: track_id=%s source_event_id=%s source=%s",
            track_id,
            source_event_id,
            source,
        )
        return track_id
