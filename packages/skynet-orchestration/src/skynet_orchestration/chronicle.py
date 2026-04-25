"""Per-invocation chronicle stream.

Every AgentEvent produced inside a sub-agent's handler XADDs a row
to ``orchestration:events`` on the Chronicle Redis DB (DB 7 -- the
same one used by the existing ``agent:events`` stream in
skynet-agent). One stream serves the whole tree; consumers filter
by ``root_invocation_id`` to reconstruct a call-graph view.

The fields here are deliberately a superset of what the existing
chronicle ingester already understands (``source``, ``kind``,
``subject``, ``text``, plus flat scalar labels) so the same
ingester can be pointed at this stream without changes.

Failure mode: the chronicle write swallows exceptions with a log
warning. Losing observability never breaks the actual agent loop.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from .envelopes import AgentEvent

log = logging.getLogger("skynet_orchestration.chronicle")

CHRONICLE_STREAM = "orchestration:events"
CHRONICLE_MAXLEN = 200_000  # approximate cap


_chronicle_redis = None
_chronicle_redis_lock = threading.Lock()


def configure_chronicle(redis_client) -> None:
    """Inject the Redis client to use for chronicle writes.

    Pinned to whatever DB the caller wires in. The convention in
    this codebase is DB 7 (Chronicle ingester's DB). Call this once
    at process startup; subsequent emit() calls reuse the client.
    """
    global _chronicle_redis
    with _chronicle_redis_lock:
        _chronicle_redis = redis_client


def emit_event(
    event: AgentEvent,
    *,
    root_invocation_id: str,
    target: str,
    caller: Optional[str] = None,
) -> None:
    """Persist one AgentEvent to the chronicle stream.

    Adds tree-level identifiers (``root_invocation_id``, ``target``,
    ``caller``) so a consumer can reconstruct the full graph
    without joining against another store.
    """
    client = _chronicle_redis
    if client is None:
        # Configured on first call; before that we're silently dropping.
        # That's intentional -- importing this module shouldn't force
        # a Redis connection.
        return
    try:
        ts = datetime.fromtimestamp(event.ts, tz=timezone.utc).isoformat()
        payload = json.dumps(
            {
                "metadata": event.metadata,
                "root_invocation_id": root_invocation_id,
                "target": target,
                "caller": caller or "",
            },
            ensure_ascii=False,
        )
        fields = {
            "source": "orchestration",
            "kind": "agent_event",
            "subject": target,
            "text": (event.content or "")[:500],
            "event_type": event.type,
            "invocation_id": event.invocation_id,
            "root_invocation_id": root_invocation_id,
            "target": target,
            "caller": caller or "",
            "payload": payload,
            "ts": ts,
        }
        client.xadd(
            CHRONICLE_STREAM,
            fields,
            maxlen=CHRONICLE_MAXLEN,
            approximate=True,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("chronicle xadd failed for invocation=%s: %s", event.invocation_id, e)


def emit_call_start(call, *, target_received_at: float) -> None:
    """Convenience: emit a synthetic ``progress`` event marking the
    moment a server actually accepted an incoming call.

    Lets dashboards measure dispatch + queueing latency separately
    from handler latency. ``call`` is an AgentCall (avoid importing
    typed annotation here to keep the Pydantic dep optional in
    consumer-side imports).
    """
    evt = AgentEvent(
        invocation_id=call.invocation_id,
        type="progress",
        content=f"received: {call.caller}→{call.target}",
        metadata={
            "purpose": call.purpose,
            "call_chain": list(call.call_chain),
            "estimate_tokens": call.estimate.tokens_needed,
            "granted_tokens": call.granted.tokens,
        },
        ts=target_received_at,
    )
    emit_event(
        evt,
        root_invocation_id=call.root_invocation_id,
        target=call.target,
        caller=call.caller,
    )
