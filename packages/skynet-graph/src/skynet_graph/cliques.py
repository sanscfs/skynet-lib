"""
cliques.py -- Louvain community detection over the memory similarity
graph.

Phase 4 of docs/rag-memory-roadmap.md. Once the nightly DAG has
populated `similar_ids` on every point with top-K cosine neighbours,
we can cluster the resulting graph to expose thematic groups -- a
point's `clique_id` answers "which group of memories does this
belong to?".

Louvain is chosen over spectral / HDBSCAN because:
  - it scales to the 10k-100k node range we need without tuning,
  - it doesn't require a distance matrix (O(N^2) memory blow-up),
  - it respects edge weights directly (we feed cosine scores),
  - it has no hyperparameter apart from the resolution knob that
    controls cluster granularity.

The `python-louvain` package is a hard-but-optional dependency: the
library imports it lazily at call time so downstream services that
never do clique detection (e.g. skynet-agent, identity /context
itself) don't pay the install cost. The airflow-skynet image, which
runs the weekly DAG, pulls `skynet-graph[louvain]` to bring it in.

API surface is deliberately tiny:
  - `compute_cliques(edges, resolution)` takes the output of the
    similarity build DAG (a flat list of `(source_id, target_id,
    weight)` tuples) and returns a `{id: clique_id}` dict.
  - `clique_sizes(cliques)` is a helper that inverts the mapping so
    the DAG can write `clique_size` alongside `clique_id` for each
    point without rescanning.
  - `filter_edges_by_cos` is a utility the caller uses to prune
    low-weight edges before clustering -- Louvain's quality suffers
    when noise edges dominate.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Iterable

logger = logging.getLogger(__name__)


#: Minimum cosine weight for an edge to participate in the Louvain
#: clustering. Below this threshold the edge is treated as noise --
#: the DAG filters before passing to compute_cliques to keep the
#: community detection stable across runs. Matches the floor the
#: nightly similarity DAG already enforces so the default is "use
#: every edge the similarity DAG wrote".
DEFAULT_MIN_COS = 0.70

#: Louvain resolution parameter. >1.0 gives more, smaller clusters;
#: <1.0 gives fewer, larger clusters. 1.0 is the canonical default
#: and what `python-louvain` ships with. Exposed so the DAG can tune
#: without a code change.
DEFAULT_RESOLUTION = 1.0


def filter_edges_by_cos(
    edges: Iterable[tuple[Any, Any, float]],
    *,
    min_cos: float = DEFAULT_MIN_COS,
) -> list[tuple[Any, Any, float]]:
    """Drop edges with weight below `min_cos`.

    Returns a fresh list so the caller can pass the iterable directly
    to compute_cliques without accidentally consuming a generator
    twice. Edges with non-numeric / NaN weights are dropped too -- the
    similarity DAG shouldn't emit them, but being defensive costs
    nothing here and protects the downstream Louvain call from
    failing on malformed input.
    """
    out: list[tuple[Any, Any, float]] = []
    for entry in edges:
        if len(entry) != 3:
            continue
        src, dst, w = entry
        if src is None or dst is None:
            continue
        try:
            wf = float(w)
        except (TypeError, ValueError):
            continue
        if wf != wf:  # NaN
            continue
        if wf < min_cos:
            continue
        out.append((src, dst, wf))
    return out


def compute_cliques(
    edges: Iterable[tuple[Any, Any, float]],
    *,
    resolution: float = DEFAULT_RESOLUTION,
    random_state: int | None = 42,
) -> dict[Any, int]:
    """Run Louvain community detection and return `{id: clique_id}`.

    `edges` is a flat iterable of `(source_id, target_id, weight)`
    tuples -- typically the result of unpacking `similar_ids` from
    every point into a global list. Self-loops and duplicate edges
    are handled internally.

    Returns a dict mapping every id that appeared in any edge to its
    community index. Communities are numbered 0..K-1; the numbering
    is stable under a fixed `random_state` so repeat runs on the
    same input produce the same labels (important for payload diff
    minimisation in the DAG).

    Raises `RuntimeError` when `python-louvain` is not installed --
    the package lives behind the `[louvain]` extra so DAG images
    that need this helper install `skynet-graph[louvain]` explicitly.
    """
    try:
        import community as community_louvain  # python-louvain
        import networkx as nx
    except ImportError as e:
        raise RuntimeError(
            "compute_cliques requires `python-louvain` and `networkx`; install skynet-graph[louvain]"
        ) from e

    graph = nx.Graph()
    for src, dst, weight in edges:
        if src == dst:
            # Self-loops confuse Louvain and carry no partitioning
            # information — memories are trivially in their own group.
            continue
        if graph.has_edge(src, dst):
            # Two directional similar_ids from opposite anchors both
            # point at the same undirected edge. Keep the larger
            # weight so neither direction is lost if they disagreed.
            if weight > graph[src][dst].get("weight", 0.0):
                graph[src][dst]["weight"] = weight
        else:
            graph.add_edge(src, dst, weight=weight)

    if graph.number_of_nodes() == 0:
        return {}

    # best_partition does community detection in one call. It's not
    # deterministic across versions without a random_state, but
    # python-louvain >=0.16 exposes one and we pass it through so
    # the DAG run-to-run diff is minimal.
    partition = community_louvain.best_partition(
        graph,
        weight="weight",
        resolution=resolution,
        random_state=random_state,
    )

    # Renumber communities by descending size so clique_id=0 is the
    # largest cluster (makes the Atlas UI's "top cliques" chart trivial).
    size_by_raw_id = Counter(partition.values())
    # sort raw ids by (-size, raw_id) to be deterministic on ties
    order = sorted(size_by_raw_id.keys(), key=lambda k: (-size_by_raw_id[k], k))
    remap = {raw: new for new, raw in enumerate(order)}
    return {node: remap[raw] for node, raw in partition.items()}


def clique_sizes(cliques: dict[Any, int]) -> dict[int, int]:
    """Invert `{id: clique_id}` into `{clique_id: size}`.

    Cheap O(N) helper the DAG uses to write a `clique_size` payload
    field alongside `clique_id` so identity scoring can weight
    "you're in a big cluster" differently from "you're in a tiny
    niche cluster" without extra round-trips.
    """
    sizes: dict[int, int] = {}
    for cid in cliques.values():
        sizes[cid] = sizes.get(cid, 0) + 1
    return sizes


def compute_modularity(
    edges: Iterable[tuple[Any, Any, float]],
    cliques: dict[Any, int],
) -> float:
    """Compute Louvain modularity Q of the clique assignment.

    Returns the modularity score in [-0.5, 1.0] -- a value >= 0.3
    is the conventional "these communities are meaningful" floor in
    the community-detection literature. The DAG logs this on every
    run so the atlas UI can trend it over time; sudden drops
    indicate the similarity graph shape changed enough to demand a
    resolution tuning pass.

    Returns 0.0 when no edges / no partition are present; raises
    RuntimeError for the same missing-dep reason as compute_cliques.
    """
    try:
        import community as community_louvain
        import networkx as nx
    except ImportError as e:
        raise RuntimeError(
            "compute_modularity requires `python-louvain` and `networkx`; install skynet-graph[louvain]"
        ) from e

    graph = nx.Graph()
    for src, dst, weight in edges:
        if src == dst or src is None or dst is None:
            continue
        if not graph.has_edge(src, dst):
            graph.add_edge(src, dst, weight=float(weight))

    if graph.number_of_nodes() == 0 or not cliques:
        return 0.0

    # Louvain's modularity helper needs every graph node present in
    # the partition, so fill in any isolated nodes with their own id.
    partition = dict(cliques)
    for node in graph.nodes:
        if node not in partition:
            partition[node] = -hash(node)  # unique singleton bucket
    return float(community_louvain.modularity(partition, graph, weight="weight"))
