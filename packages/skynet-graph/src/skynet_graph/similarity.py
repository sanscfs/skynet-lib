"""
similarity.py -- build top-K similarity edges via a user-supplied
search function.

The main entry point is `build_similarity_edges` which takes:
  1. An iterable of `(point_id, vector)` pairs to act as anchors.
  2. A callable `search_fn(vector, limit) -> list[{id, score}]` that
     performs a vector search against the desired backend. The library
     stays backend-agnostic this way -- the Phase 2 nightly DAG passes
     in `skynet_qdrant.QdrantClient.search`, tests pass in a pure
     Python lambda, and future backends (Milvus, Vespa, in-memory) can
     plug in without a library rewrite.

The helper returns a flat list of `SimilarityEdge` records that
callers typically serialise to the `similar_ids` payload field of the
anchor point as a JSON array of `{id, cos, kind}` objects.

EdgeKind exists to let payload readers distinguish similarity edges
built by this batch job from the LLM-classified edges added later in
Phase 4 (supersedes / contradicts / elaborates / caused_by). All
edges live in the same payload list, differentiated only by `kind`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

logger = logging.getLogger(__name__)


class EdgeKind:
    """String constants for typed memory-graph edges.

    Avoids `Enum` so that edges serialised to Qdrant payload stay
    plain strings and can be read back without importing this module.
    New kinds added in Phase 4 join this list as a drop-in extension.
    """

    SIMILAR_TO = "similar_to"
    MENTIONED_IN = "mentioned_in"
    CO_OCCURS_WITH = "co_occurs_with"
    SUPERSEDES = "supersedes"
    CONTRADICTS = "contradicts"
    ELABORATES = "elaborates"
    CAUSED_BY = "caused_by"
    EXAMPLE_OF = "example_of"


@dataclass(slots=True)
class SimilarityEdge:
    """A directed edge from `source_id` to `target_id` with a cosine weight.

    `cos` is the raw similarity score from the search backend, in
    [0, 1] for normalised vectors. `collection` is optional and lets
    cross-collection edges carry provenance so the traversal code can
    follow them without re-running search.

    Serialises to `{id, cos, kind, collection?}` when written to
    Qdrant payload; callers own the top-level list that holds them.
    """

    source_id: Any
    target_id: Any
    cos: float
    kind: str = EdgeKind.SIMILAR_TO
    collection: str | None = None

    def to_payload(self) -> dict:
        """Return the compact dict form stored in `similar_ids[]`."""
        out: dict = {"id": self.target_id, "cos": round(self.cos, 6), "kind": self.kind}
        if self.collection is not None:
            out["collection"] = self.collection
        return out


class _SearchCallable(Protocol):
    """Protocol for the search function `build_similarity_edges` consumes.

    Must return an iterable of dicts with `id` + `score` keys; extra
    keys (payload, version) are ignored. Typing it as a Protocol keeps
    callers from depending on any specific client class.
    """

    def __call__(self, vector: Sequence[float], limit: int) -> Iterable[dict]: ...


def top_k_neighbours(
    search_fn: _SearchCallable,
    vector: Sequence[float],
    *,
    top_k: int = 10,
    min_cos: float = 0.0,
    exclude_self_id: Any | None = None,
) -> list[tuple[Any, float]]:
    """Return `[(id, cos), ...]` for the top-K hits over `min_cos`.

    Fetches `top_k + 1` internally so that an optional self-hit can be
    filtered without starving the result list. Results are returned
    in descending cos order.
    """
    limit = top_k + 1 if exclude_self_id is not None else top_k
    try:
        hits = list(search_fn(vector, limit))
    except Exception as e:
        logger.warning("top_k_neighbours search_fn raised %s, returning []", e)
        return []

    out: list[tuple[Any, float]] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        hid = h.get("id")
        if hid is None:
            continue
        if exclude_self_id is not None and hid == exclude_self_id:
            continue
        cos = float(h.get("score", 0) or 0.0)
        if cos < min_cos:
            continue
        out.append((hid, cos))
        if len(out) >= top_k:
            break
    return out


def build_similarity_edges(
    anchors: Iterable[tuple[Any, Sequence[float]]],
    search_fn: _SearchCallable,
    *,
    top_k: int = 10,
    min_cos: float = 0.70,
    collection: str | None = None,
) -> list[SimilarityEdge]:
    """Build a flat edge list from a batch of anchor points.

    Each anchor produces up to `top_k` outbound SIMILAR_TO edges. Each
    anchor's own id is filtered from its own neighbour list so points
    never edge to themselves. `min_cos` defaults to 0.70 to match the
    floor used by `skynet-scoring.dynamic_similarity_threshold` at
    high memory density -- anything below that is too weak to carry.

    Errors from the search function are swallowed per anchor so one
    malformed vector doesn't abort the whole batch.
    """
    edges: list[SimilarityEdge] = []
    for aid, vec in anchors:
        neighbours = top_k_neighbours(
            search_fn,
            vec,
            top_k=top_k,
            min_cos=min_cos,
            exclude_self_id=aid,
        )
        for nid, cos in neighbours:
            edges.append(
                SimilarityEdge(
                    source_id=aid,
                    target_id=nid,
                    cos=cos,
                    kind=EdgeKind.SIMILAR_TO,
                    collection=collection,
                )
            )
    return edges


def edges_to_payload(edges: Iterable[SimilarityEdge]) -> list[dict]:
    """Serialise a list of SimilarityEdge into the compact payload form.

    Used by the nightly DAG when writing `similar_ids[]` back to
    Qdrant payload via bulk `set_payload`.
    """
    return [e.to_payload() for e in edges]
