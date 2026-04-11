"""Skynet Retrieval -- multi-vector search and query expansion primitives.

Phase 3 of docs/rag-memory-roadmap.md. Provides the retrieval layer that
callers (identity /context, future Phase 4 graph walker, agent dispatch)
compose instead of calling Qdrant search directly with a single vector.

The library is deliberately stateless and backend-agnostic:

- `multi_search` takes a search callable (typically bound to a
  QdrantClient instance) plus a list of `(vector, weight)` candidates,
  runs the searches, and merges results by a pluggable strategy.
- `hyde_expand` takes a query plus an optional anchor and skeleton, and
  returns a hypothetical-answer string. The LLM client and cache are
  both caller-injected so the library can be imported from DAGs, FastAPI
  request paths, and unit tests without dragging an LLM dependency or a
  Redis connection along.

Public API (re-exported from submodules):

    from skynet_retrieval import (
        # Merge strategies (plain string constants, so payloads stay portable).
        MergeStrategy,
        merge_candidates,
        # Primary entry point for /context and friends.
        multi_search,
        # HyDE query expansion.
        hyde_expand,
        HydeCache,
    )
"""

from skynet_retrieval.hyde import HydeCache, hyde_expand
from skynet_retrieval.merge import MergeStrategy, merge_candidates
from skynet_retrieval.multi_search import multi_search

__all__ = [
    "MergeStrategy",
    "merge_candidates",
    "multi_search",
    "hyde_expand",
    "HydeCache",
]
