"""Vault client with Kubernetes ServiceAccount auth.

This subsumes two patterns that previously lived as copy-pasted
snippets in `skynet-agent/vault_secrets.py` and
`skynet-movies/vault_client.py`:

1. Static KV-v2 reads -- load API keys, tokens, and other long-lived
   secrets at startup.
2. Dynamic database credentials -- Vault-generated short-lived
   username/password pairs for Postgres (and friends) with automatic
   lease caching and renewal-window awareness.

Design choices:

- Auth is explicit. `authenticate()` must be called (or allowed to
  run implicitly on the first request via the internal retry). We
  never silently no-op when the SA token file is missing; that path
  raises VaultAuthError so misconfiguration fails fast.
- Re-auth is bounded. On HTTP 403 we re-login once and retry the
  call once. If the retry also fails we propagate the error rather
  than looping.
- Thread-safety. A single module-level lock protects the login
  handshake and the per-role DB-creds cache; concurrent reads of
  already-cached creds are lock-free after the initial store.
- No secrets in logs. We log role names, vault paths, and lease
  durations, never the secret bytes themselves.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import hvac
from hvac.exceptions import Forbidden, InvalidPath

from .dbcreds import DynamicDBCreds
from .exceptions import (
    VaultAuthError,
    VaultConfigError,
    VaultDBCredsError,
    VaultSecretNotFound,
)

logger = logging.getLogger("skynet_vault")

DEFAULT_VAULT_ADDR = "http://vault.vault.svc:8200"
DEFAULT_K8S_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"


class VaultClient:
    """Thin wrapper around `hvac.Client` with K8s auth and lease cache.

    The client is designed to live as a long-lived singleton inside a
    service process. Instantiation is cheap; the first Vault round-trip
    happens in `authenticate()`.

    Typical use:

        vault = VaultClient(addr=..., role="skynet-movies")
        vault.authenticate()
        api_key = vault.read_kv("secret/openrouter", "api_key")
        creds = vault.get_db_creds("skynet-movies-role")
    """

    def __init__(
        self,
        addr: str | None = None,
        role: str | None = None,
        token_path: str | None = None,
        *,
        hvac_client: hvac.Client | None = None,
    ) -> None:
        self.addr = addr or os.getenv("VAULT_ADDR", DEFAULT_VAULT_ADDR)
        self.role = role or os.getenv("VAULT_ROLE")
        self.token_path = token_path or os.getenv(
            "VAULT_K8S_TOKEN_PATH", DEFAULT_K8S_TOKEN_PATH
        )
        # hvac_client injection is purely a test seam.
        self._client: hvac.Client | None = hvac_client
        self._lock = threading.Lock()
        # Per-role cache of the most recent DB creds lease.
        self._db_creds: dict[str, DynamicDBCreds] = {}

    # -- auth ------------------------------------------------------------

    def _read_sa_token(self) -> str:
        path = Path(self.token_path)
        if not path.exists():
            raise VaultAuthError(
                f"K8s ServiceAccount token not found at {self.token_path}"
            )
        try:
            return path.read_text().strip()
        except OSError as exc:
            raise VaultAuthError(
                f"failed to read SA token at {self.token_path}: {exc}"
            ) from exc

    def authenticate(self) -> None:
        """Log in to Vault using the pod's K8s ServiceAccount JWT.

        Idempotent: if the cached client already reports
        `is_authenticated()` we skip the round-trip.
        """
        if not self.role:
            raise VaultConfigError(
                "VAULT_ROLE is not set; cannot authenticate to Vault"
            )

        with self._lock:
            if self._client is not None:
                try:
                    if self._client.is_authenticated():
                        return
                except Exception:  # noqa: BLE001 -- treat any glitch as "re-login"
                    pass
            else:
                self._client = hvac.Client(url=self.addr)

            jwt = self._read_sa_token()
            try:
                self._client.auth.kubernetes.login(role=self.role, jwt=jwt)
            except Exception as exc:  # hvac raises a zoo of error classes
                raise VaultAuthError(
                    f"Vault K8s auth failed (addr={self.addr}, role={self.role}): "
                    f"{type(exc).__name__}"
                ) from exc

            if not self._client.is_authenticated():
                raise VaultAuthError(
                    f"Vault K8s auth returned no usable token "
                    f"(addr={self.addr}, role={self.role})"
                )
            logger.info(
                "authenticated to Vault (addr=%s, role=%s)", self.addr, self.role
            )

    def _ensure_client(self) -> hvac.Client:
        if self._client is None or not self._client.is_authenticated():
            self.authenticate()
        assert self._client is not None  # for type-checkers
        return self._client

    def _with_reauth(self, fn):
        """Run `fn(client)` and, on Forbidden, re-auth once and retry once.

        We keep the same hvac.Client instance across the re-login so
        that test doubles injected via the constructor survive; only
        the Vault-side token is renewed.
        """
        client = self._ensure_client()
        try:
            return fn(client)
        except Forbidden:
            logger.warning("Vault returned 403; re-authenticating and retrying once")
            # Re-run the K8s login handshake on the existing client.
            with self._lock:
                jwt = self._read_sa_token()
                try:
                    client.auth.kubernetes.login(role=self.role, jwt=jwt)
                except Exception as exc:  # noqa: BLE001
                    raise VaultAuthError(
                        f"Vault K8s re-auth failed (addr={self.addr}, "
                        f"role={self.role}): {type(exc).__name__}"
                    ) from exc
            return fn(client)

    # -- KV v2 -----------------------------------------------------------

    def read_kv(self, path: str, key: str, *, mount_point: str = "secret") -> str:
        """Read a single key from a KV-v2 secret.

        `path` is relative to the mount point (e.g. "openrouter",
        "matrix/bot"). For backward compat with the skynet-agent
        pattern we also accept a full "secret/openrouter" style path
        and strip the mount prefix.

        Raises VaultSecretNotFound if the path or key is missing.
        """
        # Accept both "secret/openrouter" (legacy) and "openrouter" forms.
        relative = path
        prefix = f"{mount_point}/"
        if relative.startswith(prefix):
            relative = relative[len(prefix):]

        data = self.read_kv_all(relative, mount_point=mount_point)
        if key not in data:
            raise VaultSecretNotFound(
                f"key '{key}' not present at {mount_point}/{relative}"
            )
        return data[key]

    def read_kv_all(
        self, path: str, *, mount_point: str = "secret"
    ) -> dict[str, Any]:
        """Read every key at a KV-v2 secret path.

        Useful when a single Vault path bundles related credentials
        (e.g. `secret/github/repo-token` -> {username, token}).
        """
        relative = path
        prefix = f"{mount_point}/"
        if relative.startswith(prefix):
            relative = relative[len(prefix):]

        def _call(client: hvac.Client) -> dict[str, Any]:
            try:
                resp = client.secrets.kv.v2.read_secret_version(
                    path=relative,
                    mount_point=mount_point,
                    raise_on_deleted_version=True,
                )
            except InvalidPath as exc:
                raise VaultSecretNotFound(
                    f"no secret at {mount_point}/{relative}"
                ) from exc
            return resp["data"]["data"]

        return self._with_reauth(_call)

    # -- dynamic DB creds ------------------------------------------------

    def get_db_creds(
        self,
        role: str,
        *,
        mount_point: str = "database",
        renewal_margin_seconds: int = 300,
        force_refresh: bool = False,
    ) -> DynamicDBCreds:
        """Return dynamic DB creds for a Vault database role.

        Cached per-role; a cache hit is returned until it reports
        `needs_renewal(renewal_margin_seconds)`. Pass
        `force_refresh=True` to bypass the cache (e.g. after a
        connection error that suggests the password was revoked).
        """
        if not force_refresh:
            cached = self._db_creds.get(role)
            if cached is not None and not cached.needs_renewal(renewal_margin_seconds):
                return cached

        def _call(client: hvac.Client) -> DynamicDBCreds:
            try:
                resp = client.secrets.database.generate_credentials(
                    name=role, mount_point=mount_point
                )
            except InvalidPath as exc:
                raise VaultDBCredsError(
                    f"no database role '{role}' at mount '{mount_point}'"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise VaultDBCredsError(
                    f"Vault refused to issue DB creds for role '{role}': "
                    f"{type(exc).__name__}"
                ) from exc

            data = resp.get("data") or {}
            username = data.get("username")
            password = data.get("password")
            if not username or not password:
                raise VaultDBCredsError(
                    f"Vault returned malformed DB creds payload for role '{role}'"
                )
            return DynamicDBCreds(
                username=username,
                password=password,
                lease_id=resp.get("lease_id", ""),
                lease_duration=int(resp.get("lease_duration", 3600)),
                issued_at=time.time(),
                role=role,
            )

        creds = self._with_reauth(_call)
        # Cache updates go through the lock so concurrent callers either
        # observe the old value or the new one -- never a torn read.
        with self._lock:
            self._db_creds[role] = creds
        logger.info(
            "issued DB creds (role=%s, user=%s, ttl=%ds)",
            role, creds.username, creds.lease_duration,
        )
        return creds

    def invalidate_db_creds(self, role: str | None = None) -> None:
        """Drop cached DB creds so the next get_db_creds() refetches.

        Pass `role=None` to clear every cached role.
        """
        with self._lock:
            if role is None:
                self._db_creds.clear()
            else:
                self._db_creds.pop(role, None)
