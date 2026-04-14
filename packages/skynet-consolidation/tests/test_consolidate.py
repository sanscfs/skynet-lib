"""Tests for skynet_consolidation.consolidate_clique."""

from __future__ import annotations

import json

from skynet_consolidation import ConsolidationResult, Contradiction, consolidate_clique
from skynet_consolidation.consolidate import (
    MAX_MEMBERS,
    MEMBER_TEXT_CAP,
    _coerce_bool,
    _coerce_confidence,
    _coerce_id_list,
    _coerce_optional_string,
    _parse_response,
    summarise_prompt,
)


def _m(mid, text):
    return {"id": mid, "text": text}


# --- prompt shape -------------------------------------------------------


def test_prompt_includes_all_members():
    members = [_m("a", "first fact"), _m("b", "second fact")]
    p = summarise_prompt(members)
    assert "first fact" in p
    assert "second fact" in p
    assert "id=a" in p and "id=b" in p
    assert "JSON" in p


def test_prompt_truncates_long_member_text():
    long_text = "x" * 5000
    p = summarise_prompt([_m("a", long_text)])
    # Cap is enforced.
    assert "x" * MEMBER_TEXT_CAP in p
    assert "x" * (MEMBER_TEXT_CAP + 1) not in p


def test_prompt_lists_structural_edges():
    p = summarise_prompt(
        [_m("a", "new"), _m("b", "old")],
        structural_edges=[{"source_id": "a", "id": "b", "kind": "supersedes"}],
    )
    assert "a -[supersedes]-> b" in p


def test_prompt_splits_summaries_from_raws():
    """Mixed-level members: higher compression_level go to the
    'prior summaries' bucket; level-0 (default) go to 'raw evidence'."""
    members = [
        {"id": "s1", "text": "user likes plants", "compression_level": 1},
        {"id": "r1", "text": "watered lemon ak at 15:42"},
        {"id": "r2", "text": "transferred to autopot"},
    ]
    p = summarise_prompt(members)
    assert "Prior summaries" in p
    assert "id=s1 level=1" in p
    assert "Raw evidence" in p
    assert "id=r1:" in p and "id=r2:" in p
    # Prior summaries block appears BEFORE raws block so the LLM
    # reads them as the integration target.
    assert p.index("Prior summaries") < p.index("Raw evidence")


def test_prompt_no_summary_header_when_all_level_0():
    """When all members are level-0, prompt uses the old 'Cluster
    members:' header (no prior-summaries section at all)."""
    p = summarise_prompt([_m("a", "x"), _m("b", "y")])
    assert "Prior summaries" not in p
    assert "Cluster members:" in p


# --- _parse_response ----------------------------------------------------


def _valid_ids(*ids):
    s = set()
    for i in ids:
        s.add(i)
        s.add(str(i))
    return s


def test_parse_strict_json():
    r = _parse_response(
        '{"summary": "merged", "discarded_ids": ["a", "b"], "contradictions": [], "confidence": 0.85}',
        _valid_ids("a", "b", "c"),
    )
    assert r.summary == "merged"
    assert r.discarded_ids == ["a", "b"]
    assert r.contradictions == []
    assert r.confidence == 0.85


def test_parse_markdown_fence():
    text = 'Here is the result:\n```json\n{"summary": "ok", "discarded_ids": [], "confidence": 0.7}\n```'
    r = _parse_response(text, _valid_ids("a"))
    assert r.summary == "ok"
    assert r.confidence == 0.7


def test_parse_empty_returns_empty():
    r = _parse_response("", _valid_ids("a"))
    assert r.summary == ""
    assert r.confidence == 0.0


def test_parse_unparseable_keeps_raw_for_debug():
    r = _parse_response("not json at all!", _valid_ids("a"))
    assert r.summary == ""
    assert "not json" in r.raw_response


def test_parse_filters_hallucinated_discards():
    r = _parse_response(
        '{"summary": "x", "discarded_ids": ["a", "FAKE_ID"], "confidence": 0.5}',
        _valid_ids("a", "b"),
    )
    assert r.discarded_ids == ["a"]


def test_parse_contradictions():
    r = _parse_response(
        '{"summary": "x", "contradictions": ['
        '{"id_a":"a","id_b":"b","explanation":"disagree on colour"},'
        '{"id_a":"FAKE","id_b":"b","explanation":"hallucinated"}'
        '], "confidence": 0.6}',
        _valid_ids("a", "b"),
    )
    assert len(r.contradictions) == 1
    assert r.contradictions[0].id_a == "a"
    assert r.contradictions[0].explanation == "disagree on colour"


def test_parse_clamps_confidence():
    r = _parse_response('{"summary":"x","confidence":1.7}', _valid_ids("a"))
    assert r.confidence == 1.0
    r = _parse_response('{"summary":"x","confidence":-0.3}', _valid_ids("a"))
    assert r.confidence == 0.0


# --- consolidate_clique -------------------------------------------------


def test_consolidate_happy_path():
    def fake_llm(prompt, model):
        return '{"summary": "two facts merged", "discarded_ids": ["a"], "contradictions": [], "confidence": 0.9}'

    res = consolidate_clique(
        [_m("a", "fact 1"), _m("b", "fact 2")],
        fake_llm,
        model="deepseek/deepseek-v3.2",
    )
    assert res.summary == "two facts merged"
    assert res.discarded_ids == ["a"]
    assert res.confidence == 0.9


def test_consolidate_empty_on_llm_exception():
    def bad_llm(prompt, model):
        raise RuntimeError("openrouter 503")

    res = consolidate_clique([_m("a", "x"), _m("b", "y")], bad_llm, model="x")
    assert res.summary == ""
    assert res.confidence == 0.0
    assert "openrouter 503" in res.raw_response


def test_consolidate_skips_single_member():
    called = []

    def fake_llm(prompt, model):
        called.append(1)
        return "won't be called"

    res = consolidate_clique([_m("a", "only")], fake_llm, model="x")
    assert res.summary == ""
    assert called == []


def test_consolidate_clamps_member_count():
    # 50 members; MAX_MEMBERS should clamp to 30.
    members = [_m(f"id{i}", f"text {i}") for i in range(50)]
    captured = {}

    def fake_llm(prompt, model):
        captured["prompt"] = prompt
        return '{"summary":"x","confidence":0.5}'

    consolidate_clique(members, fake_llm, model="x")
    assert captured["prompt"].count("id=") == MAX_MEMBERS


def test_consolidate_to_payload_shape():
    res = ConsolidationResult(
        summary="s",
        discarded_ids=["a", "b"],
        contradictions=[Contradiction(id_a="a", id_b="c", explanation="diff")],
        confidence=0.77777,
    )
    payload = res.to_payload()
    assert payload["summary"] == "s"
    assert payload["discarded_ids"] == ["a", "b"]
    assert payload["contradictions"][0]["explanation"] == "diff"
    assert payload["confidence"] == 0.7778


def test_coerce_confidence_handles_garbage():
    assert _coerce_confidence(None) == 0.0
    assert _coerce_confidence("nope") == 0.0
    assert _coerce_confidence(float("nan")) == 0.0
    assert _coerce_confidence(0.5) == 0.5


def test_coerce_id_list_round_trips_int_strings():
    # LLM often returns stringified ids even when input was int.
    assert _coerce_id_list([42, "1"], {1, "1", 42, "42"}) == [42, "1"]


# --- Phase 12.5 actionable intent extraction ----------------------------


def test_prompt_asks_for_actionable_fields():
    """Prompt must ask the LLM for the three Phase 12.5 JSON keys so
    downstream active-memory DAG has something to read."""
    p = summarise_prompt([_m("a", "x"), _m("b", "y")])
    assert '"actionable": true|false' in p
    assert '"action_type"' in p
    assert '"action_description"' in p
    # Prompt must also tell the LLM when to set actionable=true vs false.
    assert "remind me" in p
    assert "actionable=false" in p


def test_parse_actionable_positive_case():
    """Clear intent → LLM returns actionable=true with description."""
    r = _parse_response(
        '{"summary": "user wants to water plants weekly",'
        ' "confidence": 0.9,'
        ' "actionable": true,'
        ' "action_type": "recurring_reminder",'
        ' "action_description": "Remind me weekly to water the plants"}',
        _valid_ids("a", "b"),
    )
    assert r.actionable is True
    assert r.action_type == "recurring_reminder"
    assert r.action_description == "Remind me weekly to water the plants"


def test_parse_actionable_negative_case():
    """Mere observation → LLM returns actionable=false, fields null."""
    r = _parse_response(
        '{"summary": "user enjoys gardening",'
        ' "confidence": 0.8,'
        ' "actionable": false,'
        ' "action_type": null,'
        ' "action_description": null}',
        _valid_ids("a", "b"),
    )
    assert r.actionable is False
    assert r.action_type is None
    assert r.action_description is None


def test_parse_actionable_malformed_defaults_to_false():
    """Non-bool actionable ('maybe', 'yes', 1) → False. Conservative
    default keeps Phase 12 from proposing on garbage input."""
    for bad in ("maybe", "yes", "1", 1, None):
        r = _parse_response(
            '{"summary":"x","confidence":0.5,"actionable":'
            + json.dumps(bad)
            + "}",
            _valid_ids("a"),
        )
        assert r.actionable is False, f"expected False for {bad!r}"
        assert r.action_type is None
        assert r.action_description is None


def test_parse_actionable_true_but_empty_description_downgrades():
    """actionable=true with blank/missing description is not usable by
    Phase 12 (empty Matrix prompt). Library downgrades the whole block
    to not-actionable so to_payload() doesn't emit a half-set record."""
    r = _parse_response(
        '{"summary":"x","confidence":0.5,"actionable":true,'
        '"action_type":"recurring_reminder","action_description":""}',
        _valid_ids("a"),
    )
    assert r.actionable is False
    assert r.action_type is None
    assert r.action_description is None


def test_parse_missing_actionable_fields_backward_compat():
    """Old LLM responses without the three new keys must still parse
    and default to not-actionable. This protects against staged
    rollouts where the prompt bump hasn't reached every worker yet."""
    r = _parse_response(
        '{"summary": "merged", "discarded_ids": [], "confidence": 0.6}',
        _valid_ids("a"),
    )
    assert r.summary == "merged"
    assert r.actionable is False
    assert r.action_type is None
    assert r.action_description is None


def test_to_payload_omits_actionable_when_false():
    """Non-actionable summaries keep the payload lean — no three null
    keys on every consolidation."""
    res = ConsolidationResult(summary="s", confidence=0.5)
    payload = res.to_payload()
    assert "actionable" not in payload
    assert "action_type" not in payload
    assert "action_description" not in payload


def test_to_payload_includes_actionable_when_true():
    """Actionable=true serialises all three fields so the DAG can
    merge them into the Qdrant summary payload."""
    res = ConsolidationResult(
        summary="s",
        confidence=0.9,
        actionable=True,
        action_type="recurring_reminder",
        action_description="Remind me weekly to check humidity",
    )
    payload = res.to_payload()
    assert payload["actionable"] is True
    assert payload["action_type"] == "recurring_reminder"
    assert payload["action_description"] == "Remind me weekly to check humidity"


def test_coerce_bool_strict():
    assert _coerce_bool(True) is True
    assert _coerce_bool("true") is True
    assert _coerce_bool("TRUE") is True
    assert _coerce_bool(False) is False
    assert _coerce_bool("false") is False
    assert _coerce_bool("yes") is False
    assert _coerce_bool(1) is False
    assert _coerce_bool(None) is False


def test_coerce_optional_string_handles_blanks():
    assert _coerce_optional_string(None) is None
    assert _coerce_optional_string("") is None
    assert _coerce_optional_string("   ") is None
    assert _coerce_optional_string(42) is None
    assert _coerce_optional_string("hello") == "hello"
    # Cap enforced.
    assert _coerce_optional_string("x" * 1000, cap=50) == "x" * 50
