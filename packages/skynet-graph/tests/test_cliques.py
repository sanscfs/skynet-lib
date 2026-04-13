"""Tests for Louvain clique detection helpers.

Some tests are skipped when python-louvain / networkx are not
installed — the library imports them lazily so unit-test envs
without the `[louvain]` extra still pass.
"""

from __future__ import annotations

import pytest
from skynet_graph import cliques as c

try:
    import community as _community  # noqa: F401
    import networkx as _nx  # noqa: F401

    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

requires_louvain = pytest.mark.skipif(not HAS_LOUVAIN, reason="python-louvain/networkx not installed")


# --- filter_edges_by_cos (no deps) -------------------------------------


def test_filter_edges_drops_below_threshold():
    edges = [("a", "b", 0.9), ("a", "c", 0.5), ("b", "c", 0.75)]
    out = c.filter_edges_by_cos(edges, min_cos=0.7)
    assert out == [("a", "b", 0.9), ("b", "c", 0.75)]


def test_filter_edges_drops_malformed():
    edges = [
        ("a", "b", 0.9),
        ("a", "c"),  # wrong arity
        (None, "b", 0.9),  # None src
        ("a", None, 0.9),  # None dst
        ("a", "d", "not-a-number"),  # non-numeric
        ("a", "e", float("nan")),  # NaN
    ]
    out = c.filter_edges_by_cos(edges)
    assert out == [("a", "b", 0.9)]


def test_filter_edges_returns_fresh_list():
    gen = (("a", "b", 0.9) for _ in range(2))
    out = c.filter_edges_by_cos(gen, min_cos=0.5)
    # both items consumed from generator and captured
    assert len(out) == 2


# --- clique_sizes (no deps) --------------------------------------------


def test_clique_sizes_counts_each_clique():
    assert c.clique_sizes({"a": 0, "b": 0, "c": 1, "d": 1, "e": 1}) == {0: 2, 1: 3}


def test_clique_sizes_empty():
    assert c.clique_sizes({}) == {}


# --- compute_cliques (requires louvain) -------------------------------


@requires_louvain
def test_compute_cliques_partitions_two_disjoint_triangles():
    # Two triangles A-B-C and D-E-F with no edges between them —
    # Louvain must split into exactly two communities.
    edges = [
        ("A", "B", 0.9),
        ("B", "C", 0.9),
        ("A", "C", 0.9),
        ("D", "E", 0.9),
        ("E", "F", 0.9),
        ("D", "F", 0.9),
    ]
    parts = c.compute_cliques(edges)
    assert {parts["A"], parts["B"], parts["C"]} == {parts["A"]}  # all same
    assert {parts["D"], parts["E"], parts["F"]} == {parts["D"]}
    assert parts["A"] != parts["D"]


@requires_louvain
def test_compute_cliques_returns_every_id():
    edges = [("A", "B", 0.9), ("B", "C", 0.9)]
    parts = c.compute_cliques(edges)
    assert set(parts.keys()) == {"A", "B", "C"}


@requires_louvain
def test_compute_cliques_deterministic_with_random_state():
    edges = [
        ("A", "B", 0.9),
        ("B", "C", 0.9),
        ("A", "C", 0.9),
        ("D", "E", 0.9),
        ("E", "F", 0.9),
        ("D", "F", 0.9),
    ]
    a = c.compute_cliques(edges, random_state=42)
    b = c.compute_cliques(edges, random_state=42)
    assert a == b


@requires_louvain
def test_compute_cliques_self_loops_ignored():
    edges = [("A", "A", 1.0), ("A", "B", 0.9), ("B", "C", 0.9)]
    parts = c.compute_cliques(edges)
    # A must still land in a community even though its only edge to
    # itself is ignored; the A-B edge brings it in.
    assert parts["A"] == parts["B"]
    assert set(parts.keys()) == {"A", "B", "C"}


@requires_louvain
def test_compute_cliques_keeps_heavier_weight_on_duplicates():
    # Same undirected pair seen twice with different weights -- the
    # heavier one should win so the partition reflects the stronger
    # connection.
    edges = [("A", "B", 0.5), ("A", "B", 0.95), ("C", "D", 0.9)]
    parts = c.compute_cliques(edges)
    # A and B should still co-cluster, picking up the 0.95 edge.
    assert parts["A"] == parts["B"]


@requires_louvain
def test_compute_cliques_empty_returns_empty():
    assert c.compute_cliques([]) == {}


@requires_louvain
def test_compute_cliques_assigns_contiguous_ids_largest_first():
    # 3-node + 2-node cluster. Largest (3) should get id=0.
    edges = [
        ("A", "B", 0.9),
        ("B", "C", 0.9),
        ("A", "C", 0.9),
        ("X", "Y", 0.9),
    ]
    parts = c.compute_cliques(edges)
    a_cid = parts["A"]
    x_cid = parts["X"]
    assert a_cid == 0
    assert x_cid == 1


@requires_louvain
def test_compute_modularity_reasonable_for_clustered_graph():
    # Two disjoint triangles + a single weak bridge; modularity
    # should be >= 0.3, the conventional "meaningful communities"
    # floor.
    edges = [
        ("A", "B", 0.9),
        ("B", "C", 0.9),
        ("A", "C", 0.9),
        ("D", "E", 0.9),
        ("E", "F", 0.9),
        ("D", "F", 0.9),
        ("C", "D", 0.7),  # bridge
    ]
    parts = c.compute_cliques(edges)
    q = c.compute_modularity(edges, parts)
    assert q >= 0.3


@requires_louvain
def test_compute_modularity_zero_on_empty_inputs():
    assert c.compute_modularity([], {}) == 0.0
    assert c.compute_modularity([("A", "B", 0.9)], {}) == 0.0


# --- missing-dependency behaviour (only when not installed) -----------


@pytest.mark.skipif(HAS_LOUVAIN, reason="runs only when louvain is NOT installed")
def test_compute_cliques_raises_when_louvain_missing():
    with pytest.raises(RuntimeError, match="python-louvain"):
        c.compute_cliques([("A", "B", 0.9)])


@pytest.mark.skipif(HAS_LOUVAIN, reason="runs only when louvain is NOT installed")
def test_compute_modularity_raises_when_louvain_missing():
    with pytest.raises(RuntimeError, match="python-louvain"):
        c.compute_modularity([("A", "B", 0.9)], {"A": 0, "B": 0})
