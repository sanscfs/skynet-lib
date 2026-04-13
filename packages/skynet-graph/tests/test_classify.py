"""Tests for LLM edge classification."""

from __future__ import annotations

from skynet_graph.classify import (
    RELATED_KIND,
    STRUCTURAL_KINDS,
    ClassifiedEdge,
    _build_prompt,
    _coerce_confidence,
    _coerce_kind,
    _parse_response,
    classify_edge,
    classify_pair_bidirectional,
)

# --- coercion helpers ---------------------------------------------------


def test_coerce_kind_accepts_all_valid_kinds_case_insensitive():
    for k in STRUCTURAL_KINDS:
        assert _coerce_kind(k.upper()) == k
        assert _coerce_kind(f"  {k}  ") == k


def test_coerce_kind_falls_back_to_related_on_unknown():
    for bad in [None, "", "foo", 42, {"kind": "supersedes"}]:
        assert _coerce_kind(bad) == RELATED_KIND


def test_coerce_confidence_clamps_to_unit_interval():
    assert _coerce_confidence(0.5) == 0.5
    assert _coerce_confidence(2.0) == 1.0
    assert _coerce_confidence(-0.3) == 0.0
    assert _coerce_confidence("0.42") == 0.42


def test_coerce_confidence_defaults_to_zero_on_garbage():
    for bad in [None, "nan", "abc", {"v": 1}, []]:
        assert _coerce_confidence(bad) == 0.0


# --- _parse_response ---------------------------------------------------


def test_parse_response_strict_json():
    out = _parse_response('{"kind": "supersedes", "confidence": 0.9, "reason": "A is newer"}')
    assert out == ClassifiedEdge(kind="supersedes", confidence=0.9, reason="A is newer")


def test_parse_response_unwraps_markdown_fence():
    out = _parse_response(
        'Here is the answer:\n```json\n{"kind": "contradicts", "confidence": 0.7, "reason": "opposite claims"}\n```'
    )
    assert out.kind == "contradicts"
    assert out.confidence == 0.7
    assert out.reason == "opposite claims"


def test_parse_response_tolerates_prose_prefix():
    out = _parse_response('Sure, here: {"kind": "elaborates", "confidence": 0.5, "reason": "B expands A"}')
    assert out.kind == "elaborates"


def test_parse_response_empty_yields_related_zero():
    out = _parse_response("")
    assert out == ClassifiedEdge(kind=RELATED_KIND, confidence=0.0, reason="empty response")


def test_parse_response_unparseable_preserves_raw_head_in_reason():
    out = _parse_response("This is not JSON at all!")
    assert out.kind == RELATED_KIND
    assert out.confidence == 0.0
    assert "unparseable" in out.reason
    assert "not JSON" in out.reason


def test_parse_response_clamps_confidence_and_trims_reason():
    long = "r" * 500
    out = _parse_response(f'{{"kind": "supersedes", "confidence": 1.5, "reason": "{long}"}}')
    assert out.confidence == 1.0
    assert len(out.reason) <= 200


def test_parse_response_unknown_kind_collapses_to_related():
    out = _parse_response('{"kind": "totally-new-kind", "confidence": 0.9}')
    assert out.kind == RELATED_KIND


# --- _build_prompt ------------------------------------------------------


def test_build_prompt_contains_both_inputs():
    p = _build_prompt("note A", "note B")
    assert "note A" in p
    assert "note B" in p
    assert "JSON" in p


def test_build_prompt_truncates_long_inputs():
    long_a = "a" * 5000
    p = _build_prompt(long_a, "b")
    # Trimmed to 1200 chars -- 1200 a's present, 1201 absent.
    assert "a" * 1200 in p
    assert "a" * 1201 not in p


def test_build_prompt_lists_all_allowed_kinds():
    p = _build_prompt("a", "b")
    for k in ("supersedes", "contradicts", "elaborates", "caused_by", "example_of", "related"):
        assert k in p


# --- classify_edge ------------------------------------------------------


def test_classify_edge_calls_llm_and_returns_parsed():
    calls = []

    def fake_llm(prompt, model):
        calls.append((prompt, model))
        return '{"kind": "elaborates", "confidence": 0.8, "reason": "B expands A"}'

    out = classify_edge("note A", "note B", fake_llm, model="deepseek/deepseek-v3.2")
    assert out == ClassifiedEdge(kind="elaborates", confidence=0.8, reason="B expands A")
    assert calls[0][1] == "deepseek/deepseek-v3.2"
    assert "note A" in calls[0][0]
    assert "note B" in calls[0][0]


def test_classify_edge_handles_llm_exception():
    def bad_llm(prompt, model):
        raise RuntimeError("openrouter 503")

    out = classify_edge("a", "b", bad_llm, model="x")
    assert out.kind == RELATED_KIND
    assert out.confidence == 0.0
    assert "openrouter 503" in out.reason


def test_classify_edge_empty_input_short_circuits():
    calls = []

    def fake_llm(prompt, model):
        calls.append(1)
        return '{"kind": "supersedes", "confidence": 0.9}'

    out = classify_edge("", "something", fake_llm, model="x")
    assert out.kind == RELATED_KIND
    assert out.confidence == 0.0
    assert calls == []  # LLM never called


def test_classify_edge_whitespace_input_short_circuits():
    out = classify_edge("   \n", "b", lambda p, m: "should not be called", model="x")
    assert out.kind == RELATED_KIND


def test_classify_edge_empty_response_yields_related():
    out = classify_edge("a", "b", lambda p, m: "", model="x")
    assert out.kind == RELATED_KIND
    assert out.reason == "empty response"


def test_classified_edge_to_payload_shape():
    e = ClassifiedEdge(kind="supersedes", confidence=0.8333333, reason="newer")
    assert e.to_payload() == {"kind": "supersedes", "confidence": 0.8333, "reason": "newer"}


# --- classify_pair_bidirectional ---------------------------------------


def test_classify_pair_bidirectional_calls_twice_with_swapped_order():
    seen = []

    def fake_llm(prompt, model):
        seen.append(prompt)
        return '{"kind": "supersedes", "confidence": 0.8, "reason": "r"}'

    out = classify_pair_bidirectional("alpha", "bravo", fake_llm, model="m")
    assert set(out.keys()) == {"a_to_b", "b_to_a"}
    assert len(seen) == 2
    # first prompt has alpha before bravo, second has bravo before alpha
    first_a_idx = seen[0].index("alpha")
    first_b_idx = seen[0].index("bravo")
    assert first_a_idx < first_b_idx
    second_a_idx = seen[1].index("alpha")
    second_b_idx = seen[1].index("bravo")
    assert second_b_idx < second_a_idx
