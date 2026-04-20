"""Tests for the disjunctive vibe_signal_filter helper.

These are pure-dict assertions -- no Qdrant, no FakeQdrant. The goal is
to pin the SHAPE of the emitted filter so downstream consumers
(profile-synthesis aggregator, skynet-movies / skynet-music pool_stats
fallback, VibeStore default pool_filter) can rely on a stable contract.
"""

from __future__ import annotations

from skynet_vibe.filters import DEFAULT_VIBE_CATEGORY, vibe_signal_filter


def test_filter_default_is_disjunction() -> None:
    """Without extras, the filter is a top-level ``should`` clause."""
    f = vibe_signal_filter()
    assert f == {
        "should": [
            {"key": "signal_version", "range": {"gte": 2}},
            {"key": "category", "match": {"value": DEFAULT_VIBE_CATEGORY}},
        ]
    }
    assert "must" not in f
    assert "must_not" not in f


def test_filter_custom_category_used_in_match_branch() -> None:
    f = vibe_signal_filter(vibe_category="legacy_vibe")
    should = f["should"]
    assert {"key": "category", "match": {"value": "legacy_vibe"}} in should
    assert {"key": "signal_version", "range": {"gte": 2}} in should


def test_filter_with_extra_must_nests_disjunction() -> None:
    """``extra.must`` clauses AND with the disjunction.

    The disjunction is wrapped in a ``should`` sub-clause inside
    ``must``, so Qdrant evaluates ``(extra) AND (v>=2 OR category==...)``.
    """
    extra = {
        "must": [
            {"key": "timestamp", "range": {"gte": "2026-04-01T00:00:00+00:00"}}
        ]
    }
    f = vibe_signal_filter(extra)
    assert "must" in f
    assert len(f["must"]) == 2
    # First: the caller's clause passed through.
    assert f["must"][0] == {
        "key": "timestamp",
        "range": {"gte": "2026-04-01T00:00:00+00:00"},
    }
    # Second: the disjunction, nested.
    nested = f["must"][1]
    assert "should" in nested
    assert {"key": "signal_version", "range": {"gte": 2}} in nested["should"]
    assert (
        {"key": "category", "match": {"value": DEFAULT_VIBE_CATEGORY}}
        in nested["should"]
    )


def test_filter_with_shortcut_dict_promotes_to_must_match() -> None:
    """``{"archived": False}`` shortcut becomes a must-match clause."""
    f = vibe_signal_filter({"archived": False})
    must = f["must"]
    assert {"key": "archived", "match": {"value": False}} in must
    # Disjunction nested in alongside.
    nested = next(c for c in must if "should" in c)
    assert len(nested["should"]) == 2


def test_filter_passes_through_must_not() -> None:
    extra = {"must_not": [{"key": "archived", "match": {"value": True}}]}
    f = vibe_signal_filter(extra)
    assert f.get("must_not") == [{"key": "archived", "match": {"value": True}}]
    # Disjunction still present as the sole ``must`` clause.
    assert len(f["must"]) == 1
    assert "should" in f["must"][0]


def test_filter_preserves_caller_should_alongside_base() -> None:
    """Caller's own ``should`` is AND'd in as a separate sub-filter.

    This keeps the base disjunction intact -- Qdrant would treat two
    top-level ``should`` arrays as a single merged OR otherwise, which
    would silently widen matches.
    """
    extra = {
        "should": [
            {"key": "source_v2.type", "match": {"value": "chat"}},
            {"key": "source_v2.type", "match": {"value": "telemetry"}},
        ]
    }
    f = vibe_signal_filter(extra)
    # Two must entries: the base disjunction AND the caller's should
    # (both nested so each is evaluated independently).
    assert len(f["must"]) == 2
    shoulds = [c for c in f["must"] if "should" in c]
    assert len(shoulds) == 2
    # Base disjunction has signal_version + category.
    base = next(
        s
        for s in shoulds
        if any(
            c.get("key") == "signal_version" and "range" in c
            for c in s["should"]
        )
    )
    assert (
        {"key": "category", "match": {"value": DEFAULT_VIBE_CATEGORY}}
        in base["should"]
    )
