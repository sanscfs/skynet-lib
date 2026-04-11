"""Tests for multi_search orchestration."""

from __future__ import annotations

from skynet_retrieval import MergeStrategy, multi_search


def _p(pid, score):
    return {"id": pid, "score": score, "payload": {}}


class _RecordingSearch:
    """Fake backend: serves fixed responses per-vector-id.

    The `vector` itself is just a list of floats, so we key responses
    by the tuple version of the vector. Test fixtures pre-register
    answers so each call returns the configured hit list.
    """

    def __init__(self, responses: dict[tuple[float, ...], list[dict]]):
        self.responses = responses
        self.calls: list[tuple[tuple[float, ...], int, object]] = []

    def __call__(self, vector, limit, filter):
        key = tuple(vector)
        self.calls.append((key, limit, filter))
        return self.responses.get(key, [])[:limit]


def test_multi_search_calls_once_per_candidate():
    search = _RecordingSearch(
        {
            (1.0,): [_p("a", 0.9)],
            (2.0,): [_p("b", 0.8)],
            (3.0,): [_p("c", 0.7)],
        }
    )
    out = multi_search(
        search,
        candidates=[([1.0], 1.0), ([2.0], 0.5), ([3.0], 0.8)],
        limit=5,
        filter=None,
        merge_strategy=MergeStrategy.RECIPROCAL_RANK_FUSION,
    )
    assert len(search.calls) == 3
    ids = [r["id"] for r in out]
    assert set(ids) == {"a", "b", "c"}


def test_multi_search_single_candidate_passthrough():
    search = _RecordingSearch({(1.0,): [_p("a", 0.9), _p("b", 0.8)]})
    out = multi_search(
        search,
        candidates=[([1.0], 1.0)],
        limit=1,
        filter=None,
        merge_strategy=MergeStrategy.PRIMARY_PREFERRED,
    )
    assert [r["id"] for r in out] == ["a"]
    # Single-candidate goes through the fast path: no per-candidate overfetch.
    assert search.calls[0][1] == 1


def test_multi_search_skips_empty_vectors():
    search = _RecordingSearch({(1.0,): [_p("a", 0.9)]})
    out = multi_search(
        search,
        candidates=[([1.0], 1.0), (None, 0.5), ([], 0.5)],
        limit=5,
        filter=None,
    )
    # Only the first candidate was searched.
    assert len(search.calls) == 1
    assert [r["id"] for r in out] == ["a"]


def test_multi_search_per_candidate_overfetch():
    # Multi-candidate calls overfetch: 2 * final_limit, floor 10.
    search = _RecordingSearch({(1.0,): [_p("a", 0.9)], (2.0,): [_p("b", 0.8)]})
    multi_search(
        search,
        candidates=[([1.0], 1.0), ([2.0], 0.5)],
        limit=3,
        filter=None,
    )
    # 2 * 3 = 6 < 10 floor => per-candidate limit becomes 10
    for _, limit, _ in search.calls:
        assert limit == 10


def test_multi_search_filter_forwarded():
    search = _RecordingSearch({(1.0,): [_p("a", 0.9)]})
    filter_dict = {"must": [{"key": "archived", "match": {"value": False}}]}
    multi_search(search, candidates=[([1.0], 1.0)], limit=5, filter=filter_dict)
    assert search.calls[0][2] is filter_dict


def test_multi_search_tolerates_candidate_failure():
    def flaky_search(vec, limit, filt):
        if vec == [2.0]:
            raise RuntimeError("boom")
        return [{"id": "a", "score": 0.9, "payload": {}}]

    out = multi_search(
        flaky_search,
        candidates=[([1.0], 1.0), ([2.0], 0.5)],
        limit=5,
        filter=None,
    )
    assert [r["id"] for r in out] == ["a"]


def test_multi_search_empty_candidates_returns_empty():
    search = _RecordingSearch({})
    out = multi_search(search, candidates=[], limit=5)
    assert out == []
    assert search.calls == []
