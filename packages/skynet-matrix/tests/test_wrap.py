"""Tests for the footer-composing wrapper helpers."""

from __future__ import annotations

from skynet_matrix.wrap import build_edit_payload, build_footer_payload


def test_build_footer_payload_no_trace_is_plain():
    body, extra = build_footer_payload("hello")
    assert body == "hello"
    assert extra == {}


def test_build_footer_payload_adds_footer_html_and_trace():
    body, extra = build_footer_payload(
        "hello",
        trace_id="deadbeef",
        duration_s=1.5,
        prompt_tokens=100,
        completion_tokens=50,
        service="skynet-agent",
    )
    assert "<small>" in body
    assert "deadbeef" in body
    assert "1.5s" in body
    assert "100" in body and "50" in body
    assert "skynet-agent" in body
    assert extra["format"] == "org.matrix.custom.html"
    assert "<small>" in extra["formatted_body"]
    meta = extra["dev.skynet.trace"]
    assert meta["trace_id"] == "deadbeef"
    assert meta["prompt_tokens"] == 100
    assert meta["completion_tokens"] == 50
    assert meta["duration_ms"] == 1500
    assert meta["service"] == "skynet-agent"


def test_build_footer_payload_duration_ms_precedence():
    _, extra = build_footer_payload("hi", duration_ms=2000, service="dag:active_memory")
    meta = extra["dev.skynet.trace"]
    assert meta["duration_ms"] == 2000


def test_build_footer_payload_preserves_formatted_body():
    _, extra = build_footer_payload(
        "hello",
        service="dag:probe",
        formatted_body="<p>hello</p>",
    )
    assert extra["formatted_body"].startswith("<p>hello</p>")
    assert "<small>" in extra["formatted_body"]


def test_build_footer_payload_merges_trace_meta_extra():
    _, extra = build_footer_payload(
        "x",
        trace_id="aa",
        trace_meta_extra={"dag_id": "active_memory", "run_id": "r1"},
    )
    meta = extra["dev.skynet.trace"]
    assert meta["dag_id"] == "active_memory"
    assert meta["run_id"] == "r1"


def test_build_footer_payload_respects_extra_content_override():
    _, extra = build_footer_payload(
        "x",
        service="dag:x",
        extra_content={"format": "org.matrix.custom.html", "formatted_body": "<b>x</b>"},
    )
    assert extra["formatted_body"] == "<b>x</b>"


def test_build_edit_payload_returns_trace_meta_separately():
    body, formatted, meta = build_edit_payload(
        "edited",
        trace_id="cafe",
        duration_ms=500,
        service="skynet-agent",
    )
    assert "<small>" in body
    assert formatted is not None and "<small>" in formatted
    assert meta is not None and meta["trace_id"] == "cafe"
    assert meta["duration_ms"] == 500
