"""Unit tests for skynet_vault.

All Vault I/O is mocked via an injected hvac.Client stub -- these
tests never talk to a real Vault. They exercise:

- K8s SA auth success path
- auth failure when the SA token file is missing
- KV-v2 read success + missing-key path
- dynamic DB creds: fresh fetch + cached reuse + lease math
- re-auth on 403 (Forbidden) followed by a successful retry
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from hvac.exceptions import Forbidden, InvalidPath
from skynet_vault import (
    DynamicDBCreds,
    VaultAuthError,
    VaultClient,
    VaultSecretNotFound,
)

# --- helpers ---------------------------------------------------------


def _make_hvac_stub(authenticated: bool = True) -> MagicMock:
    """Build an hvac.Client-shaped MagicMock."""
    stub = MagicMock()
    stub.is_authenticated.return_value = authenticated
    # Default-success login; tests that care override this.
    stub.auth.kubernetes.login.return_value = None
    return stub


def _client_with_token(tmp_path, *, role: str = "skynet-test") -> tuple[VaultClient, MagicMock]:
    """Build a VaultClient wired to a throw-away SA token file + stub hvac."""
    token_file = tmp_path / "token"
    token_file.write_text("fake-jwt-value")
    hvac_stub = _make_hvac_stub(authenticated=False)
    # Pre-create the hvac client so authenticate() takes the "already
    # have a client, just log in" branch.
    client = VaultClient(
        addr="http://vault.test:8200",
        role=role,
        token_path=str(token_file),
        hvac_client=hvac_stub,
    )
    return client, hvac_stub


# --- auth ------------------------------------------------------------


def test_authenticate_calls_k8s_login_with_sa_jwt(tmp_path):
    client, hvac_stub = _client_with_token(tmp_path, role="skynet-movies")

    # After login the stub should report authenticated=True so
    # subsequent calls don't loop.
    def _login(**kwargs):
        hvac_stub.is_authenticated.return_value = True

    hvac_stub.auth.kubernetes.login.side_effect = _login

    client.authenticate()

    hvac_stub.auth.kubernetes.login.assert_called_once_with(role="skynet-movies", jwt="fake-jwt-value")


def test_authenticate_raises_when_token_file_missing(tmp_path):
    hvac_stub = _make_hvac_stub(authenticated=False)
    client = VaultClient(
        addr="http://vault.test:8200",
        role="skynet-test",
        token_path=str(tmp_path / "does-not-exist"),
        hvac_client=hvac_stub,
    )
    with pytest.raises(VaultAuthError):
        client.authenticate()
    # No login attempt should have been issued.
    hvac_stub.auth.kubernetes.login.assert_not_called()


# --- KV reads --------------------------------------------------------


def test_read_kv_returns_requested_key(tmp_path):
    client, hvac_stub = _client_with_token(tmp_path)
    hvac_stub.is_authenticated.return_value = True
    hvac_stub.secrets.kv.v2.read_secret_version.return_value = {
        "data": {"data": {"api_key": "sk-abc", "extra": "noise"}},
    }

    value = client.read_kv("secret/openrouter", "api_key")

    assert value == "sk-abc"
    # Legacy "secret/<path>" form should be stripped to "<path>".
    args, kwargs = hvac_stub.secrets.kv.v2.read_secret_version.call_args
    assert kwargs["path"] == "openrouter"
    assert kwargs["mount_point"] == "secret"


def test_read_kv_raises_when_key_missing(tmp_path):
    client, hvac_stub = _client_with_token(tmp_path)
    hvac_stub.is_authenticated.return_value = True
    hvac_stub.secrets.kv.v2.read_secret_version.return_value = {
        "data": {"data": {"other": "value"}},
    }

    with pytest.raises(VaultSecretNotFound):
        client.read_kv("secret/openrouter", "api_key")


def test_read_kv_raises_when_path_missing(tmp_path):
    client, hvac_stub = _client_with_token(tmp_path)
    hvac_stub.is_authenticated.return_value = True
    hvac_stub.secrets.kv.v2.read_secret_version.side_effect = InvalidPath()

    with pytest.raises(VaultSecretNotFound):
        client.read_kv("secret/nope", "whatever")


# --- dynamic DB creds -----------------------------------------------


def test_get_db_creds_returns_lease_and_caches(tmp_path):
    client, hvac_stub = _client_with_token(tmp_path)
    hvac_stub.is_authenticated.return_value = True
    hvac_stub.secrets.database.generate_credentials.return_value = {
        "lease_id": "database/creds/skynet-movies-role/abc123",
        "lease_duration": 3600,
        "data": {"username": "v-k8s-abc", "password": "p@ss"},
    }

    first = client.get_db_creds("skynet-movies-role")
    assert isinstance(first, DynamicDBCreds)
    assert first.username == "v-k8s-abc"
    assert first.password == "p@ss"
    assert first.lease_duration == 3600
    assert not first.is_expired()
    assert not first.needs_renewal(margin_seconds=300)

    # Second call within the TTL must reuse the cache, not re-hit Vault.
    second = client.get_db_creds("skynet-movies-role")
    assert second is first
    assert hvac_stub.secrets.database.generate_credentials.call_count == 1


def test_db_creds_needs_renewal_within_margin():
    now = time.time()
    # Issued 3500s ago with a 3600s TTL -> 100s until expiry.
    creds = DynamicDBCreds(
        username="u",
        password="p",
        lease_id="lease/x",
        lease_duration=3600,
        issued_at=now - 3500,
        role="r",
    )
    assert not creds.is_expired(now=now)
    assert creds.needs_renewal(margin_seconds=300, now=now)
    assert not creds.needs_renewal(margin_seconds=60, now=now)


# --- re-auth on 403 -------------------------------------------------


def test_read_kv_reauthenticates_once_on_forbidden(tmp_path):
    client, hvac_stub = _client_with_token(tmp_path)
    hvac_stub.is_authenticated.return_value = True

    calls = {"n": 0}

    def _read(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Forbidden("token expired")
        return {"data": {"data": {"api_key": "sk-after-reauth"}}}

    hvac_stub.secrets.kv.v2.read_secret_version.side_effect = _read

    value = client.read_kv("openrouter", "api_key")

    assert value == "sk-after-reauth"
    # Login was triggered a second time by the 403 retry path.
    assert hvac_stub.auth.kubernetes.login.call_count == 1
    # The KV read itself was retried exactly once.
    assert calls["n"] == 2
