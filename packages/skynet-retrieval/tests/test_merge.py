"""Tests for merge strategies.

Covers all three strategies on hand-crafted result lists so the fused
ordering is deterministic and easy to reason about.
"""

from __future__ import annotations

import pytest
from skynet_retrieval.merge import (
    RRF_K,
    MergeStrategy,
    merge_candidates,
)


def _p(pid, score, **payload):
    return {"id": pid, "score": score, "payload": payload}


# --- primary_preferred --------------------------------------------------


def test_primary_preferred_primary_only_when_no_extras():
    primary = [_p("a", 0.9), _p("b", 0.8)]
    out = merge_candidates([primary], weights=[1.0], limit=5, strategy=MergeStrategy.PRIMARY_PREFERRED)
    assert [r["id"] for r in out] == ["a", "b"]


def test_primary_preferred_secondary_fills_gaps():
    primary = [_p("a", 0.9), _p("b", 0.8)]
    hyde = [_p("c", 0.7), _p("a", 0.6)]  # a is dup, c is new
    out = merge_candidates(
        [primary, hyde], weights=[1.0, 0.8], limit=5, strategy=MergeStrategy.PRIMARY_PREFERRED
    )
    assert [r["id"] for r in out] == ["a", "b", "c"]


def test_primary_preferred_limit_respects_primary_first():
    primary = [_p("a", 0.9), _p("b", 0.8)]
    extras = [_p("c", 0.7)]
    out = merge_candidates([primary, extras], weights=[1.0, 0.8], limit=2, strategy=MergeStrategy.PRIMARY_PREFERRED)
    assert [r["id"] for r in out] == ["a", "b"]


def test_primary_preferred_extras_ordered_by_weight():
    primary = [_p("a", 0.9)]
    low_weight = [_p("b", 0.5)]
    high_weight = [_p("c", 0.5)]
    # high_weight (0.9) should slot extras ahead of low_weight (0.4)
    out = merge_candidates(
        [primary, low_weight, high_weight],
        weights=[1.0, 0.4, 0.9],
        limit=5,
        strategy=MergeStrategy.PRIMARY_PREFERRED,
    )
    assert [r["id"] for r in out] == ["a", "c", "b"]


# --- reciprocal_rank_fusion --------------------------------------------


def test_rrf_scores_match_formula():
    a = [_p("x", 0.9), _p("y", 0.8)]
    b = [_p("y", 0.95), _p("x", 0.5)]
    out = merge_candidates([a, b], weights=[1.0, 1.0], limit=5, strategy=MergeStrategy.RECIPROCAL_RANK_FUSION)
    # x: 1/(1+K) + 1/(2+K)   y: 1/(2+K) + 1/(1+K)  -> tie, same order as input
    expected_x = 1 / (1 + RRF_K) + 1 / (2 + RRF_K)
    expected_y = 1 / (2 + RRF_K) + 1 / (1 + RRF_K)
    assert expected_x == pytest.approx(expected_y)
    scores = {r["id"]: r["rrf_score"] for r in out}
    assert scores["x"] == pytest.approx(expected_x)
    assert scores["y"] == pytest.approx(expected_y)


def test_rrf_weight_breaks_tie():
    a = [_p("x", 0.9)]
    b = [_p("y", 0.9)]
    # equal rank 1 in their own lists, but y's list carries 2x the weight
    out = merge_candidates([a, b], weights=[1.0, 2.0], limit=5, strategy=MergeStrategy.RECIPROCAL_RANK_FUSION)
    assert [r["id"] for r in out] == ["y", "x"]


def test_rrf_preserves_original_score_field():
    a = [_p("x", 0.87)]
    out = merge_candidates([a], weights=[1.0], limit=5, strategy=MergeStrategy.RECIPROCAL_RANK_FUSION)
    # Single candidate goes through RRF too; score must survive
    # untouched so downstream decay/graph boosting still has cosine.
    assert out[0]["score"] == 0.87
    assert "rrf_score" in out[0]


def test_rrf_missing_from_one_list_contributes_zero_there():
    a = [_p("x", 0.9)]
    b = [_p("y", 0.9)]
    out = merge_candidates([a, b], weights=[1.0, 1.0], limit=5, strategy=MergeStrategy.RECIPROCAL_RANK_FUSION)
    ids = [r["id"] for r in out]
    assert "x" in ids and "y" in ids
    scores = {r["id"]: r["rrf_score"] for r in out}
    assert scores["x"] == pytest.approx(1 / (1 + RRF_K))


# --- max_score ----------------------------------------------------------


def test_max_score_uses_weighted_max_across_lists():
    a = [_p("x", 0.8)]
    b = [_p("x", 0.9)]
    out = merge_candidates([a, b], weights=[1.0, 0.5], limit=5, strategy=MergeStrategy.MAX_SCORE)
    # weighted: a -> 0.8 * 1.0 = 0.8, b -> 0.9 * 0.5 = 0.45; max is 0.8
    assert out[0]["id"] == "x"
    assert out[0]["score"] == pytest.approx(0.8)


def test_max_score_orders_by_weighted_score():
    a = [_p("x", 0.8), _p("y", 0.4)]
    b = [_p("z", 1.0)]
    out = merge_candidates(
        [a, b], weights=[1.0, 0.5], limit=5, strategy=MergeStrategy.MAX_SCORE
    )
    # weighted scores: x=0.8, y=0.4, z=0.5 -> order [x, z, y]
    assert [r["id"] for r in out] == ["x", "z", "y"]


# --- housekeeping ------------------------------------------------------


def test_dedupe_first_wins():
    a = [_p("x", 0.9), _p("y", 0.8)]
    b = [_p("y", 0.95)]  # y should keep the primary's representative under primary_preferred
    out = merge_candidates(
        [a, b], weights=[1.0, 1.0], limit=5, strategy=MergeStrategy.PRIMARY_PREFERRED
    )
    assert [r["id"] for r in out] == ["x", "y"]
    # Check the primary's score was preserved for y, not overwritten.
    assert next(r for r in out if r["id"] == "y")["score"] == 0.8


def test_empty_candidate_lists_do_not_crash():
    out = merge_candidates([], weights=[], limit=5, strategy=MergeStrategy.RECIPROCAL_RANK_FUSION)
    assert out == []


def test_empty_per_list_is_tolerated():
    a = [_p("x", 0.9)]
    out = merge_candidates([a, []], weights=[1.0, 0.5], limit=5, strategy=MergeStrategy.RECIPROCAL_RANK_FUSION)
    assert [r["id"] for r in out] == ["x"]


def test_missing_id_raises():
    bad = [{"score": 0.9, "payload": {}}]
    with pytest.raises(ValueError, match="missing id"):
        merge_candidates([bad], weights=[1.0], limit=5, strategy=MergeStrategy.MAX_SCORE)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown merge strategy"):
        merge_candidates([[_p("x", 0.9)]], weights=[1.0], limit=5, strategy="magic")
