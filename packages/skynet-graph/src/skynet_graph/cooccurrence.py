"""
cooccurrence.py -- accumulate co-occurrence counts into payload form.

Phase 2d plan: identity writes the top-K point ids of each /context
call into a Redis set `coocur:{point_id}` with a sliding 7-day
expiry. A nightly DAG flushes those sets into payload as
`co_occurs_with: [{id, count}]`, pruning entries with count==0. The
math of "merge a new batch of peer ids into an existing count map" is
common enough to live here rather than in the DAG, so unit tests can
cover the merge semantics without Redis in the loop.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

# Cap on how many co-occurrence partners we store per point. Memory
# graph only uses the top-N anyway; below the floor they're noise.
MAX_COOCCURRENCE_PARTNERS = 32


def merge_cooccurrence(
    existing: Iterable[Mapping[str, Any]] | None,
    new_peer_ids: Iterable[Any],
    *,
    max_partners: int = MAX_COOCCURRENCE_PARTNERS,
) -> list[dict]:
    """Merge a batch of new peer ids into an existing co-occurrence list.

    `existing` is the current `co_occurs_with` payload value (a list
    of `{id, count}` dicts, or empty). `new_peer_ids` is the iterable
    of point ids seen together in the current retrieval batch. Each
    incoming id increments its count by 1; duplicates in a single
    batch count multiple times which matches the Redis-set-cardinality
    semantics the DAG uses downstream.

    Returned list is sorted by count descending then by id ascending,
    truncated to `max_partners` so payload size stays bounded. Entries
    with count <= 0 are pruned (the DAG passes in None for decayed
    partners to signal removal).
    """
    counts: dict[Any, int] = {}
    if existing:
        for entry in existing:
            if not isinstance(entry, Mapping):
                continue
            pid = entry.get("id")
            if pid is None:
                continue
            try:
                counts[pid] = int(entry.get("count", 0) or 0)
            except (TypeError, ValueError):
                counts[pid] = 0

    for pid in new_peer_ids:
        if pid is None:
            continue
        counts[pid] = counts.get(pid, 0) + 1

    cleaned = [(pid, c) for pid, c in counts.items() if c > 0]
    cleaned.sort(key=lambda kv: (-kv[1], str(kv[0])))
    cleaned = cleaned[:max_partners]
    return [{"id": pid, "count": c} for pid, c in cleaned]
