"""Unit tests for the pg helpers. Real asyncpg is NOT required."""

from __future__ import annotations

from skynet_capture.common.pg import resolve_dsn


def test_resolve_dsn_missing(monkeypatch):
    monkeypatch.delenv("TEST_CAPTURE_DSN", raising=False)
    assert resolve_dsn("TEST_CAPTURE_DSN") is None


def test_resolve_dsn_empty_string(monkeypatch):
    monkeypatch.setenv("TEST_CAPTURE_DSN", "   ")
    assert resolve_dsn("TEST_CAPTURE_DSN") is None


def test_resolve_dsn_set(monkeypatch):
    monkeypatch.setenv("TEST_CAPTURE_DSN", "postgresql://u:p@h/db")
    assert resolve_dsn("TEST_CAPTURE_DSN") == "postgresql://u:p@h/db"
