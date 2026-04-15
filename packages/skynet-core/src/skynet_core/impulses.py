"""Signal bus primitives for the Skynet autonomous-impulse system.

Redis-stream-backed contract so every Skynet component can nominate
"something worth noticing" without being the component that decides to
speak to the user. Only skynet-agent reads this stream and, based on
its own homeostat state, decides whether to compose a message.

Producers:
    emit_signal(kind="novelty", source="analyzer", salience=0.7, ...)

Consumer (agent-side):
    ensure_consumer_group()                    # once at startup
    sigs = await drain_signals("agent-1")      # each tick
    ...process the batch, update homeostat...
    for entry_id, _ in sigs:
        await ack_signal(entry_id)

Design notes:
  * Stream name and retention are library constants; every caller
    sees the same bus by default. Override only in tests.
  * Salience is the producer's honest self-estimate of importance, in
    [0, 1]. It is NOT a bid for attention -- the agent's homeostat can
    and will damp high-salience signals on topics it just spoke about.
  * Anchor is a short topic key (e.g. "trait:focus", "git:sanscfs/infra",
    "incident:qdrant-oom") used for per-topic refractory dedupe. Keep
    it stable across emissions about the same subject so the consumer
    can track "I just spoke about this."
  * No schema version field: producers and consumer ship in one
    monorepo bump, so there is nothing to migrate between. Add one the
    day a real breaking change is needed -- not before.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

# --- schema ------------------------------------------------------------

SignalKind = Literal[
    "novelty",  # collectors/analyzer detected something new/out-of-distribution
    "concern",  # bad alert / incident / health regression / silence-with-anomaly
    "resolution",  # a prior concern was closed (damps the concern drive)
    "trait_drift",  # profile-synthesis detected a shift in a user trait
    "memory_activation",  # identity surfaced a hot-but-unmentioned point during retrieval
    "unfinished",  # a parked topic / pending question is still open
    "spoke",  # the agent itself just posted (self-feedback, dampens need_to_share)
    "silenced",  # user asked the agent to be quiet (bumps refractory)
]

SignalSource = Literal[
    "analyzer",
    "collectors",
    "identity",
    "profile-synthesis",
    "sre",
    "alert-bridge",
    "self",
    "external",  # one-off DAG / script / manual injection
]

STREAM_NAME = "skynet:impulses"
DEFAULT_MAXLEN = 5000
DEFAULT_CONSUMER_GROUP = "agent-self"


@dataclass(frozen=True, slots=True)
class Signal:
    kind: SignalKind
    source: SignalSource
    salience: float
    anchor: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    ts_logical: int = field(default_factory=lambda: int(time.time()))
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_fields(self) -> dict[str, str]:
        """Flatten to the Redis-streams field map.

        Redis stores field values as strings; JSON-encode the payload
        so nested dicts round-trip cleanly and keep non-ASCII (cyrillic
        anchors etc.) readable in XRANGE output.
        """
        return {
            "kind": self.kind,
            "source": self.source,
            "salience": f"{self.salience:.4f}",
            "anchor": self.anchor or "",
            "payload": json.dumps(self.payload, ensure_ascii=False, default=str),
            "ts_logical": str(self.ts_logical),
            "id": self.id,
        }

    @classmethod
    def from_fields(cls, fields: dict[str, str]) -> "Signal":
        return cls(
            kind=fields["kind"],  # type: ignore[arg-type]
            source=fields["source"],  # type: ignore[arg-type]
            salience=float(fields.get("salience") or 0.0),
            anchor=(fields.get("anchor") or None),
            payload=json.loads(fields.get("payload") or "{}"),
            ts_logical=int(fields.get("ts_logical") or 0),
            id=fields.get("id") or "",
        )


def _validate_salience(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"salience must be in [0, 1]; got {value!r}")
    return value


# --- emit (producers) --------------------------------------------------


def emit_signal(
    kind: SignalKind,
    source: SignalSource,
    salience: float,
    anchor: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
    *,
    redis_client=None,
    stream: str = STREAM_NAME,
    maxlen: int = DEFAULT_MAXLEN,
) -> str:
    """Publish a Signal to the impulse bus (sync).

    Returns the Redis-assigned stream entry id. Producers should call
    this at the moment they notice something -- not batched, not delayed.
    The agent's tick loop is the only component that aggregates.
    """
    from skynet_core.redis import get_redis

    _validate_salience(salience)
    sig = Signal(
        kind=kind,
        source=source,
        salience=salience,
        anchor=anchor,
        payload=payload or {},
    )
    client = redis_client or get_redis()
    return client.xadd(stream, sig.to_fields(), maxlen=maxlen, approximate=True)


async def emit_signal_async(
    kind: SignalKind,
    source: SignalSource,
    salience: float,
    anchor: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
    *,
    async_redis=None,
    stream: str = STREAM_NAME,
    maxlen: int = DEFAULT_MAXLEN,
) -> str:
    """Async variant for FastAPI / asyncio consumers."""
    from skynet_core.redis import get_async_redis

    _validate_salience(salience)
    sig = Signal(
        kind=kind,
        source=source,
        salience=salience,
        anchor=anchor,
        payload=payload or {},
    )
    client = async_redis or get_async_redis()
    return await client.xadd(stream, sig.to_fields(), maxlen=maxlen, approximate=True)


# --- consume (agent) ---------------------------------------------------


def ensure_consumer_group(
    group: str = DEFAULT_CONSUMER_GROUP,
    *,
    redis_client=None,
    stream: str = STREAM_NAME,
    start_id: str = "$",
) -> None:
    """Idempotent create of the agent consumer group on the impulses stream.

    start_id="$" (default) = only signals emitted after the group was
    created are delivered. Pass "0" to replay the entire retained
    history (useful in tests or a cold-start backfill).
    """
    from skynet_core.redis import get_redis

    client = redis_client or get_redis()
    try:
        client.xgroup_create(stream, group, id=start_id, mkstream=True)
    except Exception as exc:
        # BUSYGROUP means the group already exists -- idempotent happy path.
        if "BUSYGROUP" not in str(exc):
            raise


async def drain_signals(
    consumer: str,
    *,
    group: str = DEFAULT_CONSUMER_GROUP,
    async_redis=None,
    stream: str = STREAM_NAME,
    count: int = 200,
    block_ms: int = 100,
) -> list[tuple[str, Signal]]:
    """Pull pending signals for this consumer, one non-blocking read.

    Returns a list of (entry_id, Signal). Caller is responsible for
    calling ack_signal(entry_id) after each is processed; un-acked
    entries live in the PEL and can be reclaimed by another consumer
    if this one dies.

    Malformed entries are ACK-and-dropped to avoid poison-pill loops.
    """
    from skynet_core.redis import get_async_redis

    client = async_redis or get_async_redis()
    # ">" = only entries never delivered to any consumer in this group.
    resp = await client.xreadgroup(
        group,
        consumer,
        streams={stream: ">"},
        count=count,
        block=block_ms,
    )
    out: list[tuple[str, Signal]] = []
    if not resp:
        return out
    for _stream_name, entries in resp:
        for entry_id, fields in entries:
            try:
                out.append((entry_id, Signal.from_fields(fields)))
            except Exception as exc:
                logger.warning(
                    "impulses: dropping malformed entry %s (%s): %s",
                    entry_id,
                    exc,
                    fields,
                )
                try:
                    await client.xack(stream, group, entry_id)
                except Exception:
                    pass
    return out


async def ack_signal(
    entry_id: str,
    *,
    group: str = DEFAULT_CONSUMER_GROUP,
    async_redis=None,
    stream: str = STREAM_NAME,
) -> int:
    """ACK a single entry so Redis removes it from the PEL."""
    from skynet_core.redis import get_async_redis

    client = async_redis or get_async_redis()
    return await client.xack(stream, group, entry_id)


async def ack_signals(
    entry_ids: list[str],
    *,
    group: str = DEFAULT_CONSUMER_GROUP,
    async_redis=None,
    stream: str = STREAM_NAME,
) -> int:
    """Batch ACK. Returns the count of entries Redis actually removed."""
    if not entry_ids:
        return 0
    from skynet_core.redis import get_async_redis

    client = async_redis or get_async_redis()
    return await client.xack(stream, group, *entry_ids)
