"""Merge strategies for multi-vector retrieval.

When `multi_search` runs N searches with N different vectors (body,
anchor, HyDE), it ends up with N ranked lists that often overlap on the
same points. The merge strategy decides how those overlapping lists
collapse into a single final ranking.

Three strategies cover the practical use cases:

1. `primary_preferred` -- the first candidate's ranking is the skeleton;
   additional candidates only contribute points that the primary missed.
   Useful when the primary signal is high-trust and the secondaries are
   supplemental (e.g. body is the user's literal question; HyDE is a
   paraphrase that should only help when the primary whiffs).

2. `reciprocal_rank_fusion` (RRF) -- each point's final score is the
   weighted sum of `1/(rank + k)` across every list it appears in. This
   is the classical IR fusion approach and is the one that most closely
   matches the roadmap's phrasing:
       score(doc) = sum_i weight_i * 1/(rank_i(doc) + k)
   where `k=60` is the standard Cormack et al. constant. Ranks are
   1-indexed; points missing from a list contribute zero from that list.

3. `max_score` -- final score is the max of `weight_i * score_i` over
   the lists the point appears in. Cheapest strategy; useful when the
   absolute cosine score is already calibrated (normalised vectors into
   the same embedding space) and you just want "best hit from any
   direction".

All strategies operate on pre-searched candidates, so the merge is pure
Python with no side effects and is easy to unit-test.

Input shape: each candidate is a list of result dicts with at least
`id`, `score`, and optional `payload` / `vector`. This matches what
`skynet_qdrant.QdrantClient.search` returns and what identity already
consumes, so callers do not need to re-shape the data.
"""

from __future__ import annotations

from typing import Any, Sequence


class MergeStrategy:
    """String constants for merge strategies.

    Deliberately plain strings (not an Enum) so values can be passed
    across HTTP boundaries, stored in ConfigMaps, and written to
    metrics labels without a serialization dance.
    """

    PRIMARY_PREFERRED = "primary_preferred"
    RECIPROCAL_RANK_FUSION = "reciprocal_rank_fusion"
    MAX_SCORE = "max_score"


#: Cormack et al. RRF constant -- dampens the contribution of low ranks
#: so the top of each list dominates the fused order. 60 is the
#: classical value that the IR literature converged on and what most
#: modern hybrid-search systems (Elasticsearch, Vespa, Pinecone) ship as
#: default. Exposed as a module constant so tests and the atlas
#: formulas catalog can reference the exact same number.
RRF_K = 60


def _point_id(point: dict[str, Any]) -> Any:
    """Extract the id from a result dict, preferring `id` over `point_id`.

    Some callers have historically used `point_id` as a payload field,
    so we fall back to that to stay tolerant of mixed input shapes.
    Returning `None` would silently drop results, so we raise instead.
    """
    pid = point.get("id")
    if pid is None:
        pid = point.get("point_id")
    if pid is None:
        raise ValueError(f"retrieval result missing id: {point!r}")
    return pid


def _dedupe_keep_first(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate ids from a list, keeping the first occurrence.

    Qdrant shouldn't return duplicates within one call, but merged
    lists from multi_search are built by concatenating N searches and
    can include the same id more than once if we forget to dedupe.
    """
    seen: set[Any] = set()
    out: list[dict[str, Any]] = []
    for p in points:
        pid = _point_id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def _merge_primary_preferred(
    candidate_results: Sequence[Sequence[dict[str, Any]]],
    weights: Sequence[float],
    limit: int,
) -> list[dict[str, Any]]:
    """Primary-preferred merge: first list wins, rest fill gaps.

    The rationale is conservative: the primary vector (usually the
    user's literal query) is the "trusted" ranking. Secondary vectors
    (HyDE, anchor) only get to insert points that the primary missed.
    This preserves the retrieval behaviour that tests expect from the
    pre-Phase-3 code path while still giving the fuzzy signals a way
    in when they have unique hits.
    """
    if not candidate_results:
        return []

    primary = list(candidate_results[0] or [])
    primary_ids = {_point_id(p) for p in primary}

    # Remaining candidates contribute their points in priority order,
    # but only if not already in the primary list. Weight doesn't
    # affect ordering here -- it only gates the contribution budget
    # when `limit` is tight, and higher-weighted candidates get
    # preference by being processed first.
    order = sorted(
        range(1, len(candidate_results)),
        key=lambda i: -float(weights[i] if i < len(weights) else 1.0),
    )
    extras: list[dict[str, Any]] = []
    for i in order:
        for p in candidate_results[i] or []:
            pid = _point_id(p)
            if pid in primary_ids:
                continue
            primary_ids.add(pid)
            extras.append(p)

    merged = primary + extras
    return merged[:limit]


def _merge_rrf(
    candidate_results: Sequence[Sequence[dict[str, Any]]],
    weights: Sequence[float],
    limit: int,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion merge (Cormack et al.).

    `score(doc) = sum_i weight_i * 1/(rank_i(doc) + RRF_K)`

    - Rank is 1-indexed (top hit rank=1, second rank=2, ...).
    - Documents missing from a candidate list contribute 0 from it.
    - Final ordering is by descending fused score.
    - Each document carries its best-seen representative dict (the one
      from the highest-weighted list it appeared in) so downstream
      code still has the payload and vector available.
    """
    fused_score: dict[Any, float] = {}
    best_repr: dict[Any, dict[str, Any]] = {}
    best_weight: dict[Any, float] = {}

    for i, results in enumerate(candidate_results):
        w = float(weights[i] if i < len(weights) else 1.0)
        for rank, point in enumerate(results or [], start=1):
            pid = _point_id(point)
            fused_score[pid] = fused_score.get(pid, 0.0) + w / (rank + RRF_K)
            if w > best_weight.get(pid, float("-inf")):
                best_weight[pid] = w
                best_repr[pid] = point

    ordered = sorted(fused_score.keys(), key=lambda pid: -fused_score[pid])
    out: list[dict[str, Any]] = []
    for pid in ordered[:limit]:
        p = dict(best_repr[pid])
        # Preserve the original Qdrant score so decay/graph scoring in
        # the caller still sees a cosine number; the RRF score is
        # attached as a side field for observability.
        p["rrf_score"] = fused_score[pid]
        out.append(p)
    return out


def _merge_max_score(
    candidate_results: Sequence[Sequence[dict[str, Any]]],
    weights: Sequence[float],
    limit: int,
) -> list[dict[str, Any]]:
    """Max weighted score merge: best `weight_i * cos_i` across lists.

    Cheap and intuitive. Each point's final score is the highest of
    its weighted cosine scores across the candidate lists it appears
    in. The representative dict kept is the one from the list that
    contributed the winning score, so `score` on the output IS the
    fused score (unlike RRF where the original cosine is preserved).
    """
    best_score: dict[Any, float] = {}
    best_repr: dict[Any, dict[str, Any]] = {}

    for i, results in enumerate(candidate_results):
        w = float(weights[i] if i < len(weights) else 1.0)
        for point in results or []:
            pid = _point_id(point)
            cos = float(point.get("score", 0.0) or 0.0)
            weighted = w * cos
            if weighted > best_score.get(pid, float("-inf")):
                best_score[pid] = weighted
                rep = dict(point)
                rep["score"] = weighted
                best_repr[pid] = rep

    ordered = sorted(best_score.keys(), key=lambda pid: -best_score[pid])
    return [best_repr[pid] for pid in ordered[:limit]]


def merge_candidates(
    candidate_results: Sequence[Sequence[dict[str, Any]]],
    weights: Sequence[float],
    limit: int,
    strategy: str = MergeStrategy.RECIPROCAL_RANK_FUSION,
) -> list[dict[str, Any]]:
    """Merge per-candidate result lists into a single ranking.

    `candidate_results[i]` is the result list from the i-th search;
    `weights[i]` is the caller-supplied importance of that candidate.
    `limit` caps the merged output length. Returns a deduplicated,
    strategy-dependent ranking.

    Missing weights default to 1.0. Empty candidate lists are handled
    gracefully (the strategy just sees no contribution from that slot).
    """
    if not candidate_results:
        return []

    if strategy == MergeStrategy.PRIMARY_PREFERRED:
        merged = _merge_primary_preferred(candidate_results, weights, limit)
    elif strategy == MergeStrategy.RECIPROCAL_RANK_FUSION:
        merged = _merge_rrf(candidate_results, weights, limit)
    elif strategy == MergeStrategy.MAX_SCORE:
        merged = _merge_max_score(candidate_results, weights, limit)
    else:
        raise ValueError(f"unknown merge strategy: {strategy!r}")

    return _dedupe_keep_first(merged)
