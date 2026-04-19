"""Chronicle mirror hook — XADD every Matrix message onto a Redis stream.

Why this exists
---------------
Chronicle is Skynet's "raw text, nothing touched by an LLM" record-keeping
layer. Every inbound message a bot receives and every reply a bot sends
should land in Parquet on MinIO via the Chronicle ingester, at full
fidelity, with zero intermediate transformation by an LLM / embedder /
analyzer.

This module is the single emission point. The contract is::

    from skynet_matrix.chronicle_mirror import mirror_message

    mirror_message(
        redis_client,                # sync redis.Redis
        direction="in" | "out",
        room_id="!abc:matrix...",
        sender="@sanscfs:matrix...",
        body="hello",
        event_id="$evt:matrix...",
        ts=time.time(),              # seconds since epoch, float
        user="sanscfs" | "skynet-<component>",
    )

The call is wrapped in try/except and NEVER raises back to the caller —
mirroring is opt-in observability, not a critical path. A broken Redis
must not take down a Matrix bot.

Security / zero-LLM guarantee
-----------------------------
The ONLY external things this file touches are:

* ``redis`` (TCP to our in-cluster Redis) — the transport.
* ``json`` + ``logging`` + ``time`` + ``os`` — stdlib.

There is deliberately NO import of any LLM / embedder / http client
that could leak the raw message text to a third party before it reaches
Chronicle's own WAL + Parquet files on MinIO. Reviewers should keep
this module's imports tightly controlled when extending it.

Opt-in / opt-out
----------------
The helper is a no-op unless the caller provides a live Redis client.
``CommandBot`` pulls the URL from ``CHRONICLE_REDIS_URL`` (env) once at
startup and caches a sync client; when the env var is unset the bot
simply skips the mirror call. Existing bots keep working without any
config change.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

#: Environment variable that opts a bot into Chronicle mirroring. Unset
#: => no mirror client is constructed and ``mirror_message`` is a no-op.
CHRONICLE_REDIS_URL_ENV = "CHRONICLE_REDIS_URL"

#: Default destination stream read by the Chronicle ingester
#: (``_consume_redis_streams`` in ``skynet_chronicle_service.ingester``).
#: Configurable via ``CHRONICLE_MATRIX_STREAM`` for tests / staging.
DEFAULT_STREAM = "matrix:events"

#: Trim the stream at roughly this many entries. The Chronicle ingester
#: is the durable store; this stream is only an in-flight buffer.
DEFAULT_MAXLEN = 100_000


Direction = Literal["in", "out"]


def _build_envelope(
    *,
    direction: Direction,
    room_id: str,
    sender: str,
    body: str,
    event_id: str,
    ts: float,
    user: str,
) -> dict[str, Any]:
    """Build a Chronicle-native ``EventEnvelope`` dict.

    Field meanings match ``skynet_chronicle.EventEnvelope`` exactly so
    that the ingester can hand the dict straight to ``coerce_envelope``
    without extra remapping. See
    ``skynet_chronicle_service.ingester._envelope_from_stream_fields``
    for the preferred ``{"envelope": "<json>"}`` wire shape we use.

    ``user`` preserves the identity distinction from
    ``project_chat_identity_fix`` — user-originated messages land under
    ``user="sanscfs"`` (or whichever MXID localpart sent them) and
    bot-originated replies land under ``user="skynet-<component>"``. The
    Chronicle query layer relies on this to separate the two identities
    in later retrieval.
    """
    # ts is seconds-since-epoch; chronicle accepts ISO-8601 strings via
    # coerce_envelope and fills tzinfo when missing.
    iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + f".{int((ts % 1) * 1_000_000):06d}+00:00"

    labels: dict[str, str] = {
        "room": room_id,
        "sender": sender,
        "user": user,
        "direction": direction,
    }
    payload: dict[str, Any] = {
        "event_id": event_id,
        "room_id": room_id,
        "sender": sender,
        "body": body,
        "direction": direction,
    }
    return {
        "ts": iso,
        "source": "matrix",
        "kind": "message",
        "subject": room_id,
        "text": body,
        "labels": labels,
        "payload": payload,
    }


def mirror_message(
    redis_client: Any,
    *,
    direction: Direction,
    room_id: str,
    sender: str,
    body: str,
    event_id: str,
    ts: float,
    user: str,
    stream: Optional[str] = None,
    maxlen: int = DEFAULT_MAXLEN,
) -> None:
    """XADD a Matrix message envelope onto the Chronicle mirror stream.

    Parameters
    ----------
    redis_client:
        A ``redis.Redis`` (sync) client. The bot is responsible for
        constructing this from ``CHRONICLE_REDIS_URL``. If ``None``, this
        call is a silent no-op — that lets bots keep running in
        environments where Chronicle mirroring is deliberately disabled.
    direction:
        ``"in"`` for messages the bot received, ``"out"`` for replies
        the bot sent. This is critical for preserving the user identity
        distinction: inbound messages belong to the human sender, outbound
        messages belong to the bot and must not feed back into user
        profiling (see ``project_chat_identity_fix``).
    room_id, sender, body, event_id, ts, user:
        Forwarded into the ``EventEnvelope`` built by
        :func:`_build_envelope`. ``ts`` is seconds-since-epoch (float).
    stream:
        Destination Redis stream. Defaults to ``matrix:events`` (or the
        value of ``CHRONICLE_MATRIX_STREAM`` env var if set), which is
        what the Chronicle ingester's consumer group subscribes to.
    maxlen:
        Approximate stream trim bound. The ingester drains this stream
        into durable Parquet almost immediately, so the in-flight buffer
        stays small in steady state.

    Returns
    -------
    None. Never raises — all exceptions are logged at WARNING and
    swallowed. Mirroring is an observability hook, not a correctness
    gate.
    """
    if redis_client is None:
        return

    target_stream = stream or os.environ.get("CHRONICLE_MATRIX_STREAM") or DEFAULT_STREAM

    try:
        envelope = _build_envelope(
            direction=direction,
            room_id=room_id,
            sender=sender,
            body=body,
            event_id=event_id,
            ts=ts,
            user=user,
        )
    except Exception as exc:  # pragma: no cover — paranoia
        logger.warning("chronicle_mirror: envelope build failed: %s", exc)
        return

    fields = {
        # Preferred wire shape per ingester._envelope_from_stream_fields:
        # a single "envelope" field containing the serialized dict.
        "envelope": json.dumps(envelope, ensure_ascii=False),
        # Mirror a few top-level keys as flat fields so stream browsing
        # tools (redis-cli XINFO, Redis Insights) surface them without
        # having to parse the embedded JSON.
        "source": "matrix",
        "kind": "message",
        "subject": room_id,
        "direction": direction,
        "user": user,
    }

    try:
        redis_client.xadd(
            target_stream,
            fields,
            maxlen=maxlen,
            approximate=True,
        )
    except Exception as exc:
        # Never take the bot down because Chronicle mirroring is unhappy.
        logger.warning(
            "chronicle_mirror: XADD stream=%s failed: %s",
            target_stream,
            type(exc).__name__,
        )


def get_mirror_client(url: Optional[str] = None) -> Any | None:
    """Build a sync Redis client from ``CHRONICLE_REDIS_URL``, or return ``None``.

    Centralized here so multiple call-sites (``CommandBot``, direct users of
    :class:`skynet_matrix.MatrixClient` that want mirroring) build the
    client the same way. The caller owns the client lifetime — we do NOT
    cache it as a module-level singleton, because different components
    may want different databases (``/7`` for chat bots today, possibly
    other DBs later).

    Returns ``None`` when:

    * No URL is given and ``CHRONICLE_REDIS_URL`` is unset / empty.
    * The ``redis`` package somehow failed to import.
    * The initial ``ping`` failed — we still return the client so the
      caller sees the misconfig, but we log a warning.

    The client uses ``decode_responses=True`` so the envelope string
    round-trips through ``json.loads`` in the ingester cleanly.
    """
    resolved = url or os.environ.get(CHRONICLE_REDIS_URL_ENV) or ""
    if not resolved:
        return None

    try:
        import redis as redis_sync
    except Exception as exc:  # pragma: no cover — redis is a hard dep
        logger.warning("chronicle_mirror: redis import failed: %s", exc)
        return None

    try:
        client = redis_sync.Redis.from_url(
            resolved,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
    except Exception as exc:
        logger.warning("chronicle_mirror: redis.from_url failed: %s", exc)
        return None

    try:
        client.ping()
    except Exception as exc:
        # Return the client anyway — the mirror call itself logs and
        # swallows errors, so transient unreachability is self-healing
        # as soon as Redis comes back.
        logger.warning("chronicle_mirror: ping failed url=%s: %s", resolved, exc)

    return client


__all__ = [
    "CHRONICLE_REDIS_URL_ENV",
    "DEFAULT_STREAM",
    "DEFAULT_MAXLEN",
    "Direction",
    "get_mirror_client",
    "mirror_message",
]
