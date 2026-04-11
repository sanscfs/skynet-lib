"""
traversal.py -- walk the payload-stored memory graph.

Identity's /neighbors endpoint and Phase 5 consolidation both need to
answer "starting from point X, what is reachable within N hops over
these edge kinds?". This module implements a bounded BFS over the
`similar_ids[]` payload field (and its Phase 4 siblings) without
assuming the graph lives in memory.

The caller supplies a `lookup_fn(point_id) -> payload_dict` which we
call at each hop; that lets the traversal hit Qdrant lazily, or read
from a pre-fetched dict in tests, without baking in any specific
backend. Results are returned as a list of `(id, depth, edge_kind,
cos)` tuples so callers can render provenance ("you reached X via a
similar_to edge at depth 2 from Y").

The BFS is capped by `max_depth` and `max_nodes` so a densely-linked
cluster can't hijack a retrieval loop, and by `edge_types` so callers
can request e.g. "only similar_to + mentioned_in" or "only structural
edges (supersedes/elaborates/caused_by)".
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from skynet_graph.similarity import EdgeKind

logger = logging.getLogger(__name__)

# Edge kinds that count as "structural" -- these are the ones LLM
# classification adds in Phase 4 and they carry stronger semantics
# than raw similarity. Useful as a default edge_types filter when a
# caller wants "typed relationships only" rather than neighbourhood.
STRUCTURAL_EDGE_KINDS: frozenset[str] = frozenset(
    {
        EdgeKind.SUPERSEDES,
        EdgeKind.CONTRADICTS,
        EdgeKind.ELABORATES,
        EdgeKind.CAUSED_BY,
        EdgeKind.EXAMPLE_OF,
    }
)


@dataclass(slots=True, frozen=True)
class ReachableNode:
    """One result of a traversal walk.

    `provenance_id` is the id from which the BFS reached this node
    (None for the starting point) and `edge_kind` / `cos` describe
    the edge that was traversed to get here. `depth` is the hop
    count from the start.
    """

    id: Any
    depth: int
    provenance_id: Any | None
    edge_kind: str | None
    cos: float | None


def _read_edges(
    payload: dict,
    *,
    edge_types: frozenset[str] | None,
) -> list[tuple[Any, str, float]]:
    """Extract outbound edges from a payload's `similar_ids` field.

    Accepts two shapes for backward compatibility:
      1. New: list of `{id, cos, kind, collection?}` dicts
      2. Legacy: list of bare ids (no weight, no kind) -- treated as
         SIMILAR_TO with cos=1.0 so they still participate in traversal
    """
    raw = payload.get("similar_ids") or []
    out: list[tuple[Any, str, float]] = []
    for entry in raw:
        if isinstance(entry, dict):
            eid = entry.get("id")
            kind = str(entry.get("kind") or EdgeKind.SIMILAR_TO)
            cos = float(entry.get("cos", 1.0) or 1.0)
        else:
            eid = entry
            kind = EdgeKind.SIMILAR_TO
            cos = 1.0
        if eid is None:
            continue
        if edge_types is not None and kind not in edge_types:
            continue
        out.append((eid, kind, cos))
    return out


def traverse_from(
    start_id: Any,
    lookup_fn: Callable[[Any], dict | None],
    *,
    max_depth: int = 2,
    max_nodes: int = 100,
    edge_types: Iterable[str] | None = None,
    include_start: bool = False,
) -> list[ReachableNode]:
    """BFS from `start_id` over payload-stored edges.

    `lookup_fn(id) -> payload_dict | None` is called at each hop to
    fetch the point's payload. Returning None aborts that branch
    (use for deleted / archived targets).

    `edge_types` accepts any iterable of kind strings; falsy values
    mean "every kind". Duplicates are de-duplicated into a frozenset
    before the walk.

    Returns a list sorted by (depth, -cos) so shallow high-quality
    hits come first. The starting node is only included when
    `include_start=True`.
    """
    edge_type_set: frozenset[str] | None = None
    if edge_types is not None:
        et = frozenset(str(e) for e in edge_types)
        if et:
            edge_type_set = et

    visited: set = {start_id}
    queue: deque = deque([(start_id, 0, None, None, None)])
    results: list[ReachableNode] = []
    if include_start:
        results.append(ReachableNode(id=start_id, depth=0, provenance_id=None, edge_kind=None, cos=None))

    while queue:
        if len(results) >= max_nodes:
            break
        current_id, depth, prov_id, prov_kind, prov_cos = queue.popleft()
        if depth >= max_depth:
            continue

        try:
            payload = lookup_fn(current_id)
        except Exception as e:
            logger.warning("traverse_from lookup_fn(%s) raised %s", current_id, e)
            continue
        if payload is None:
            continue

        for target_id, kind, cos in _read_edges(payload, edge_types=edge_type_set):
            if target_id in visited:
                continue
            visited.add(target_id)
            results.append(
                ReachableNode(
                    id=target_id,
                    depth=depth + 1,
                    provenance_id=current_id,
                    edge_kind=kind,
                    cos=cos,
                )
            )
            if len(results) >= max_nodes:
                break
            queue.append((target_id, depth + 1, current_id, kind, cos))

    # Sort: shallower hops first, then descending cos within each depth.
    results.sort(key=lambda r: (r.depth, -(r.cos or 0.0)))
    return results


def reachable(
    start_id: Any,
    lookup_fn: Callable[[Any], dict | None],
    *,
    max_depth: int = 2,
    max_nodes: int = 100,
    edge_types: Iterable[str] | None = None,
) -> set:
    """Thin wrapper around `traverse_from` that returns just the id set.

    Convenience for callers that don't need provenance -- e.g. the
    Phase 5 consolidation promoter counts reachable clique size.
    """
    return {
        node.id
        for node in traverse_from(
            start_id,
            lookup_fn,
            max_depth=max_depth,
            max_nodes=max_nodes,
            edge_types=edge_types,
            include_start=False,
        )
    }
