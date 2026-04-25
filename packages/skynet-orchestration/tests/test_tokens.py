"""HMAC token mint/verify."""

from __future__ import annotations

import time

import pytest
from skynet_orchestration import tokens


def test_mint_then_verify_passes():
    t = tokens.mint(invocation_id="inv1", caller="main")
    tokens.verify(t, invocation_id="inv1", caller="main")


def test_verify_rejects_invocation_swap():
    t = tokens.mint(invocation_id="inv1", caller="main")
    with pytest.raises(tokens.TokenError):
        tokens.verify(t, invocation_id="inv2", caller="main")


def test_verify_rejects_caller_swap():
    """Token bound to caller -- another agent can't reuse it."""
    t = tokens.mint(invocation_id="inv1", caller="main")
    with pytest.raises(tokens.TokenError):
        tokens.verify(t, invocation_id="inv1", caller="music")


def test_verify_rejects_tampered_payload():
    """Flipping a single base64 byte breaks the HMAC."""
    t = tokens.mint(invocation_id="inv1", caller="main")
    parts = t.split(".")
    # corrupt the payload by changing one character
    corrupted = parts[0] + "." + ("X" + parts[1][1:]) + "." + parts[2]
    with pytest.raises(tokens.TokenError):
        tokens.verify(corrupted, invocation_id="inv1", caller="main")


def test_verify_rejects_expired_token(monkeypatch):
    """Tokens older than TOKEN_TTL_SECONDS are rejected."""
    real_time = time.time
    monkeypatch.setattr(tokens, "time", type("T", (), {"time": staticmethod(lambda: real_time() - 9999)}))
    t = tokens.mint(invocation_id="inv1", caller="main")
    monkeypatch.setattr(tokens, "time", type("T", (), {"time": staticmethod(real_time)}))
    with pytest.raises(tokens.TokenError, match="expired"):
        tokens.verify(t, invocation_id="inv1", caller="main")


def test_mint_requires_secret(monkeypatch):
    """Without ORCHESTRATION_HMAC_SECRET, mint refuses to proceed."""
    monkeypatch.delenv("ORCHESTRATION_HMAC_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="ORCHESTRATION_HMAC_SECRET"):
        tokens.mint(invocation_id="inv1", caller="main")
