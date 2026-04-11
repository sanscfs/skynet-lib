"""Skynet Graph -- memory graph utilities: similarity, traversal, cliques.

Phase 2 of docs/rag-memory-roadmap.md. Provides the primitives that
the nightly `skynet_build_similarity_graph` DAG, identity's
/neighbors endpoint, and Phase 4 Louvain clustering all build on.

Deliberately stateless and backend-agnostic: the library knows how to
build a graph from a list of Qdrant points (or any mapping with an
`id` + `vector` + `payload`) but does not hold a QdrantClient of its
own. Callers pass in a search function if they want the library to
fetch neighbours itself, or pre-fetched points if they already have
them in memory. This keeps the library importable from DAGs, from the
FastAPI request path, and from the scoring monorepo tests without
dragging a Qdrant dependency along.

Public API (re-exported from submodules):

    from skynet_graph import (
        # Similarity graph construction
        EdgeKind,
        SimilarityEdge,
        build_similarity_edges,
        top_k_neighbours,
        # Traversal
        traverse_from,
        reachable,
        # Co-occurrence (used by the /context hook + daily flush DAG)
        merge_cooccurrence,
    )
"""

from skynet_graph.cooccurrence import merge_cooccurrence
from skynet_graph.similarity import (
    EdgeKind,
    SimilarityEdge,
    build_similarity_edges,
    top_k_neighbours,
)
from skynet_graph.traversal import (
    reachable,
    traverse_from,
)

__all__ = [
    "EdgeKind",
    "SimilarityEdge",
    "build_similarity_edges",
    "top_k_neighbours",
    "traverse_from",
    "reachable",
    "merge_cooccurrence",
]
