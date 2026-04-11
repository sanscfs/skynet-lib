"""Tests for skynet_graph: similarity edges, traversal, co-occurrence."""

from __future__ import annotations

from skynet_graph import (
    EdgeKind,
    SimilarityEdge,
    build_similarity_edges,
    merge_cooccurrence,
    reachable,
    top_k_neighbours,
    traverse_from,
)
from skynet_graph.traversal import STRUCTURAL_EDGE_KINDS

# --- SimilarityEdge ------------------------------------------------------


def test_similarity_edge_to_payload_minimum():
    e = SimilarityEdge(source_id="a", target_id="b", cos=0.87)
    assert e.to_payload() == {"id": "b", "cos": 0.87, "kind": EdgeKind.SIMILAR_TO}


def test_similarity_edge_to_payload_with_collection():
    e = SimilarityEdge(source_id=1, target_id=2, cos=0.5, collection="episodic")
    out = e.to_payload()
    assert out == {"id": 2, "cos": 0.5, "kind": EdgeKind.SIMILAR_TO, "collection": "episodic"}


# --- top_k_neighbours ---------------------------------------------------


def _fake_search(response):
    def _fn(vector, limit):
        return response[:limit]

    return _fn


def test_top_k_neighbours_filters_self():
    response = [
        {"id": "a", "score": 0.99},
        {"id": "b", "score": 0.8},
        {"id": "c", "score": 0.7},
    ]
    out = top_k_neighbours(_fake_search(response), [0.1, 0.2], top_k=2, exclude_self_id="a")
    assert [pid for pid, _ in out] == ["b", "c"]


def test_top_k_neighbours_respects_min_cos():
    response = [
        {"id": "a", "score": 0.9},
        {"id": "b", "score": 0.6},
        {"id": "c", "score": 0.4},
    ]
    out = top_k_neighbours(_fake_search(response), [0.1], top_k=5, min_cos=0.7)
    assert [pid for pid, _ in out] == ["a"]


def test_top_k_neighbours_swallows_search_exception():
    def raiser(vector, limit):
        raise RuntimeError("backend is gone")

    out = top_k_neighbours(raiser, [0.1], top_k=5)
    assert out == []


def test_top_k_neighbours_skips_malformed_hits():
    response = [
        "garbage",  # not a dict
        {"id": None, "score": 0.9},  # missing id
        {"id": "a", "score": 0.8},
    ]
    out = top_k_neighbours(_fake_search(response), [0.1], top_k=5)
    assert out == [("a", 0.8)]


# --- build_similarity_edges ---------------------------------------------


def test_build_similarity_edges_multiple_anchors():
    universe = {
        "a": [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.85}, {"id": "c", "score": 0.72}],
        "b": [{"id": "b", "score": 1.0}, {"id": "a", "score": 0.85}, {"id": "c", "score": 0.90}],
    }

    def search_fn(vec, limit):
        # vec[0] is the anchor id coded as a float for the test
        anchor_id = vec[0]
        return universe[anchor_id]

    anchors = [("a", ["a"]), ("b", ["b"])]
    edges = build_similarity_edges(anchors, search_fn, top_k=3, min_cos=0.70)

    # Self-edges filtered, both anchors produced outbound edges.
    kinds = {(e.source_id, e.target_id) for e in edges}
    assert ("a", "b") in kinds
    assert ("a", "c") in kinds
    assert ("b", "c") in kinds
    assert ("b", "a") in kinds
    assert ("a", "a") not in kinds
    assert ("b", "b") not in kinds


def test_build_similarity_edges_below_floor_filtered():
    universe_hits = [{"id": "x", "score": 0.65}]

    def search_fn(vec, limit):
        return universe_hits

    edges = build_similarity_edges([("anchor", [0.1])], search_fn, top_k=5, min_cos=0.70)
    assert edges == []  # nothing clears the floor


# --- traversal ----------------------------------------------------------


def _dict_lookup(graph: dict[str, dict]):
    def _fn(pid):
        return graph.get(pid)

    return _fn


def test_traverse_from_bfs_shallow_high_cos_first():
    graph = {
        "root": {
            "similar_ids": [
                {"id": "a", "cos": 0.9, "kind": "similar_to"},
                {"id": "b", "cos": 0.7, "kind": "similar_to"},
            ]
        },
        "a": {"similar_ids": [{"id": "c", "cos": 0.6, "kind": "similar_to"}]},
        "b": {"similar_ids": []},
        "c": {"similar_ids": []},
    }
    out = traverse_from("root", _dict_lookup(graph), max_depth=2)
    # a (depth=1, cos=0.9) should come before c (depth=2) and before
    # b (depth=1, cos=0.7).
    assert [n.id for n in out] == ["a", "b", "c"]
    assert [n.depth for n in out] == [1, 1, 2]


def test_traverse_from_respects_max_depth():
    graph = {
        "root": {"similar_ids": [{"id": "a"}]},
        "a": {"similar_ids": [{"id": "b"}]},
        "b": {"similar_ids": [{"id": "c"}]},
        "c": {"similar_ids": []},
    }
    out = traverse_from("root", _dict_lookup(graph), max_depth=1)
    ids = {n.id for n in out}
    assert ids == {"a"}


def test_traverse_from_edge_type_filter():
    graph = {
        "root": {
            "similar_ids": [
                {"id": "a", "cos": 0.9, "kind": "similar_to"},
                {"id": "b", "cos": 0.9, "kind": "supersedes"},
            ]
        },
        "a": {"similar_ids": []},
        "b": {"similar_ids": []},
    }
    similar_only = traverse_from("root", _dict_lookup(graph), edge_types=["similar_to"])
    assert [n.id for n in similar_only] == ["a"]

    structural = traverse_from("root", _dict_lookup(graph), edge_types=STRUCTURAL_EDGE_KINDS)
    assert [n.id for n in structural] == ["b"]


def test_traverse_from_legacy_bare_id_edges():
    # Backward compat: edges stored as bare ids (no dict wrapper).
    graph = {
        "root": {"similar_ids": ["a", "b"]},
        "a": {"similar_ids": []},
        "b": {"similar_ids": []},
    }
    out = traverse_from("root", _dict_lookup(graph))
    assert {n.id for n in out} == {"a", "b"}
    # Legacy edges default to cos=1.0 and kind=similar_to
    assert all(n.cos == 1.0 for n in out)
    assert all(n.edge_kind == EdgeKind.SIMILAR_TO for n in out)


def test_traverse_from_max_nodes_cap():
    graph = {"root": {"similar_ids": [{"id": f"n{i}", "cos": 0.8 - i * 0.01} for i in range(50)]}}
    for i in range(50):
        graph[f"n{i}"] = {"similar_ids": []}
    out = traverse_from("root", _dict_lookup(graph), max_nodes=5)
    assert len(out) == 5


def test_traverse_from_missing_node_is_ok():
    graph = {"root": {"similar_ids": [{"id": "ghost"}]}}
    # ghost doesn't exist in graph, lookup returns None, branch aborts.
    out = traverse_from("root", _dict_lookup(graph))
    assert [n.id for n in out] == ["ghost"]
    # But walking past it would find nothing.


def test_traverse_from_include_start():
    graph = {"root": {"similar_ids": []}}
    out = traverse_from("root", _dict_lookup(graph), include_start=True)
    assert out[0].id == "root"
    assert out[0].depth == 0


def test_reachable_returns_id_set():
    graph = {
        "root": {"similar_ids": [{"id": "a"}]},
        "a": {"similar_ids": [{"id": "b"}]},
        "b": {"similar_ids": []},
    }
    assert reachable("root", _dict_lookup(graph), max_depth=2) == {"a", "b"}


# --- merge_cooccurrence -------------------------------------------------


def test_merge_cooccurrence_fresh():
    out = merge_cooccurrence(None, ["a", "b", "a"])
    # "a" appeared twice in the batch -> count=2
    assert out == [{"id": "a", "count": 2}, {"id": "b", "count": 1}]


def test_merge_cooccurrence_with_existing():
    existing = [{"id": "x", "count": 5}, {"id": "y", "count": 2}]
    out = merge_cooccurrence(existing, ["y", "z"])
    ids_counts = {entry["id"]: entry["count"] for entry in out}
    assert ids_counts == {"x": 5, "y": 3, "z": 1}
    # Order: count desc, then id asc
    assert out[0]["id"] == "x"
    assert out[-1]["id"] == "z"


def test_merge_cooccurrence_cap():
    existing = [{"id": f"p{i}", "count": 50 - i} for i in range(40)]
    out = merge_cooccurrence(existing, [], max_partners=5)
    assert len(out) == 5
    # Top 5 by count
    assert [e["id"] for e in out] == ["p0", "p1", "p2", "p3", "p4"]


def test_merge_cooccurrence_prunes_zero_counts():
    existing = [{"id": "a", "count": 0}, {"id": "b", "count": 1}]
    out = merge_cooccurrence(existing, [])
    assert out == [{"id": "b", "count": 1}]


def test_merge_cooccurrence_tolerates_malformed_entries():
    existing = [
        "garbage",  # not a dict
        {"id": None, "count": 3},  # missing id
        {"id": "real", "count": "not-int"},  # non-int count
        {"id": "ok", "count": 4},
    ]
    out = merge_cooccurrence(existing, ["real"])
    ids_counts = {e["id"]: e["count"] for e in out}
    # 'real' gets count reset to 0 from bad input, then +1 from batch = 1
    assert ids_counts["real"] == 1
    assert ids_counts["ok"] == 4
