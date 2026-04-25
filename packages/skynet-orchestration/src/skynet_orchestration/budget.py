"""Shared budget tracking for one invocation tree.

A user-initiated turn allocates a root budget. Every sub-invocation
in the tree decrements from the same Redis hash, so depth + breadth
are bounded by one shared resource pool rather than per-agent caps.
This naturally kills runaway loops -- once the pool drains, no one
gets more work, regardless of topology.

Storage: Redis hash at ``orchestration:budget:<root_invocation_id>``
with fields ``tokens``, ``tool_calls``, ``time_ms`` (each an int
counting *remaining*) and ``extensions_remaining`` (per-invocation
cap honoured by the server). TTL set on first write so dead trees
GC themselves.

All ops use ``HINCRBY`` for atomic decrement; we never read-modify-
write the hash from Python so two concurrent sub-agents in the
same tree can decrement safely.
"""

from __future__ import annotations

from dataclasses import dataclass

from .envelopes import BudgetGrant, WorkActuals

KEY_PREFIX = "orchestration:budget"
ROOT_TTL_SECONDS = 3600  # 1 hour -- a single user turn never legitimately runs longer


def _key(root_invocation_id: str) -> str:
    return f"{KEY_PREFIX}:{root_invocation_id}"


@dataclass
class RootBudget:
    """Per-tree budget cap. Set once when a user turn begins."""

    tokens: int
    tool_calls: int
    time_ms: int


def init_root(redis_client, root_invocation_id: str, cap: RootBudget) -> None:
    """Initialise the shared pool for a new tree.

    Idempotent: if the hash already exists (re-entry from a retry),
    leaves the existing remaining values alone. The TTL is refreshed
    so a long tree doesn't expire mid-flight.
    """
    key = _key(root_invocation_id)
    pipe = redis_client.pipeline()
    pipe.hsetnx(key, "tokens", cap.tokens)
    pipe.hsetnx(key, "tool_calls", cap.tool_calls)
    pipe.hsetnx(key, "time_ms", cap.time_ms)
    pipe.hsetnx(key, "extensions_granted", 0)
    pipe.expire(key, ROOT_TTL_SECONDS)
    pipe.execute()


def remaining(redis_client, root_invocation_id: str) -> RootBudget:
    """Snapshot of the pool right now."""
    key = _key(root_invocation_id)
    raw = redis_client.hgetall(key) or {}
    # Handle bytes-or-str depending on decode_responses setting.
    norm = {
        (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v) for k, v in raw.items()
    }
    return RootBudget(
        tokens=int(norm.get("tokens", 0)),
        tool_calls=int(norm.get("tool_calls", 0)),
        time_ms=int(norm.get("time_ms", 0)),
    )


def try_reserve(
    redis_client,
    root_invocation_id: str,
    grant: BudgetGrant,
) -> bool:
    """Reserve grant from the pool atomically.

    Returns True if the pool had enough; False if the call should be
    rejected with status=budget_exhausted. Uses optimistic decrement
    + rollback on negative -- HINCRBY can't conditionally fail, so we
    decrement, check, and undo if any field went below zero.
    """
    key = _key(root_invocation_id)
    pipe = redis_client.pipeline()
    pipe.hincrby(key, "tokens", -grant.tokens)
    pipe.hincrby(key, "tool_calls", -grant.tool_calls)
    pipe.hincrby(key, "time_ms", -grant.time_ms)
    after = pipe.execute()
    if any(int(v) < 0 for v in after):
        # rollback
        rb = redis_client.pipeline()
        rb.hincrby(key, "tokens", grant.tokens)
        rb.hincrby(key, "tool_calls", grant.tool_calls)
        rb.hincrby(key, "time_ms", grant.time_ms)
        rb.execute()
        return False
    return True


def refund(redis_client, root_invocation_id: str, actuals: WorkActuals, granted: BudgetGrant) -> None:
    """Return unused budget to the pool after a call finishes.

    If the agent used less than was granted, the unused difference
    flows back so siblings/parents can use it. Negative diffs (used
    more than granted -- happens with mid-flight extensions) are
    *not* refunded; the pool only ever gains here.
    """
    key = _key(root_invocation_id)
    diff_tokens = max(0, granted.tokens - actuals.tokens_used)
    diff_calls = max(0, granted.tool_calls - actuals.tool_calls_made)
    diff_ms = max(0, granted.time_ms - actuals.time_ms)
    if diff_tokens or diff_calls or diff_ms:
        pipe = redis_client.pipeline()
        if diff_tokens:
            pipe.hincrby(key, "tokens", diff_tokens)
        if diff_calls:
            pipe.hincrby(key, "tool_calls", diff_calls)
        if diff_ms:
            pipe.hincrby(key, "time_ms", diff_ms)
        pipe.execute()


def grant_extension(
    redis_client,
    root_invocation_id: str,
    additional: BudgetGrant,
    *,
    max_extensions_per_tree: int = 10,
) -> bool:
    """Honour a mid-flight budget_request event.

    Two checks: (1) pool has the headroom, (2) tree-wide extension
    counter hasn't hit its cap. Counter is shared across the tree so
    every sub-agent contributes to the same ceiling -- prevents one
    branch from monopolising extensions.
    """
    key = _key(root_invocation_id)
    extensions = int(redis_client.hincrby(key, "extensions_granted", 1))
    if extensions > max_extensions_per_tree:
        # rollback the counter and refuse
        redis_client.hincrby(key, "extensions_granted", -1)
        return False
    return try_reserve(redis_client, root_invocation_id, additional)
