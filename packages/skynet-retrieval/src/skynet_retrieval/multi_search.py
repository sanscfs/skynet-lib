"""Multi-vector search over a backend-agnostic search callable.

`multi_search` is the one retrieval primitive that identity /context,
the agent, and the future graph walker all go through. It generalises
single-vector Qdrant search to N query vectors in one call, with a
pluggable merge strategy.

Design constraints:

1. Stateless and backend-agnostic -- the library does NOT import
   skynet_qdrant. Callers pass a `search_fn(vector, limit, filter)`
   callable that knows how to talk to Qdrant / Milvus / an in-memory
   test fixture. This mirrors how `skynet_graph.build_similarity_edges`
   takes a search function, and keeps the test surface small.

2. Candidate weights are arbitrary positive floats -- the merge
   strategy decides how to interpret them (RRF scales the 1/(rank+k)
   term; max_score multiplies into the cosine score; primary_preferred
   uses them to break ties among non-primary candidates).

3. `limit` is the FINAL output length, not a per-candidate budget. The
   function over-fetches internally (2x the final limit per candidate)
   to give the merge strategy some headroom without burning too much
   Qdrant bandwidth.

4. Filters pass through unchanged -- Qdrant filter dicts are opaque
   here; we just forward them to every search call.

Typical call from identity /context:

    from skynet_retrieval import multi_search, MergeStrategy

    def _search(vec, limit, filt):
        return _qdrant.search(COLLECTION_EPISODIC, vec, limit=limit, filter=filt)

    candidates = [(body_vec, 1.0)]
    if anchor_vec is not None:
        candidates.append((anchor_vec, 0.5))
    if hyde_vec is not None:
        candidates.append((hyde_vec, 0.8))

    results = multi_search(
        _search,
        candidates=candidates,
        limit=LAZY_EPISODIC_LIMIT * 2,
        filter=_active_filter,
        merge_strategy=MergeStrategy.RECIPROCAL_RANK_FUSION,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

from skynet_retrieval.merge import MergeStrategy, merge_candidates

logger = logging.getLogger(__name__)


SearchFn = Callable[[Sequence[float], int, Any], list[dict[str, Any]]]
"""Signature of the user-supplied search callable.

Takes `(vector, limit, filter)` and returns a list of result dicts
with at least `id` and `score`. The `filter` argument is opaque (a
Qdrant filter dict, or None).
"""


def _per_candidate_limit(final_limit: int, num_candidates: int) -> int:
    """Over-fetch per candidate so the merge has ranking headroom.

    Rule of thumb: fetch `2 * final_limit` per candidate, clamped to
    a reasonable floor so small-limit calls still get a useful pool
    for RRF to fuse. This matches what the identity tier-2 budgets
    already do explicitly (`LAZY_EPISODIC_LIMIT * 2`) but centralises
    the logic so single-candidate calls don't double-inflate.
    """
    if num_candidates <= 1:
        return max(final_limit, 1)
    return max(final_limit * 2, 10)


def multi_search(
    search_fn: SearchFn,
    candidates: Sequence[tuple[Sequence[float], float]],
    limit: int,
    filter: Any | None = None,
    merge_strategy: str = MergeStrategy.RECIPROCAL_RANK_FUSION,
) -> list[dict[str, Any]]:
    """Run one search per `(vector, weight)` candidate and merge.

    Parameters
    ----------
    search_fn:
        Caller-supplied search callable. Takes `(vector, limit, filter)`
        and returns a list of result dicts. Typically bound to a
        QdrantClient method with the collection name partially applied.
    candidates:
        List of `(vector, weight)` pairs. The first entry is treated
        as the primary by the `primary_preferred` strategy; weights
        modulate the fused score in `reciprocal_rank_fusion` and
        `max_score`.
    limit:
        Final output length after merging.
    filter:
        Optional filter dict, forwarded to every search call unchanged.
    merge_strategy:
        One of `MergeStrategy.*`. Default is RRF because it's the most
        forgiving when candidates have different absolute score scales.

    Returns
    -------
    list[dict]
        Merged ranking of length <= `limit`, deduplicated by id.
    """
    if not candidates:
        return []

    # Filter out empty vectors up-front -- callers pass optional
    # anchor/HyDE that may be None, and we'd rather treat them as
    # "not provided" than error out.
    usable: list[tuple[Sequence[float], float]] = [
        (vec, weight) for vec, weight in candidates if vec is not None and len(vec) > 0
    ]
    if not usable:
        return []

    per_cand_limit = _per_candidate_limit(limit, len(usable))
    weights = [w for _, w in usable]

    # Single-candidate fast path. No merge strategy can beat the plain
    # search result here, and we avoid a pointless dict walk.
    if len(usable) == 1:
        vec, _ = usable[0]
        try:
            return search_fn(vec, limit, filter)[:limit]
        except Exception as e:
            logger.warning("single-candidate multi_search failed: %s", e)
            return []

    per_results: list[list[dict[str, Any]]] = []
    for i, (vec, weight) in enumerate(usable):
        try:
            hits = search_fn(vec, per_cand_limit, filter)
        except Exception as e:
            # One bad search (e.g. a corrupted HyDE vector) must not
            # kill the whole call -- we log and contribute an empty
            # list so the other candidates still get merged.
            logger.warning("multi_search candidate %d failed: %s", i, e)
            hits = []
        per_results.append(list(hits or []))

    return merge_candidates(per_results, weights, limit=limit, strategy=merge_strategy)
