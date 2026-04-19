"""Skynet Vault -- K8s ServiceAccount auth, KV reads, dynamic DB creds.

Public surface:

    from skynet_vault import (
        VaultClient,
        DynamicDBCreds,
        get_secret,
        get_dynamic_db_creds,
        VaultError,
        VaultAuthError,
        VaultSecretNotFound,
        VaultDBCredsError,
        VaultConfigError,
    )

The module-level helpers (`get_secret`, `get_dynamic_db_creds`) wrap a
process-global singleton `VaultClient` configured from environment
variables -- the ergonomic default for a single-tenant service pod.
Code that needs multiple Vault roles in one process should instantiate
VaultClient directly.
"""

from __future__ import annotations

import threading
from typing import Any

from .client import VaultClient
from .dbcreds import DynamicDBCreds
from .exceptions import (
    VaultAuthError,
    VaultConfigError,
    VaultDBCredsError,
    VaultError,
    VaultSecretNotFound,
)

__all__ = [
    "VaultClient",
    "DynamicDBCreds",
    "get_secret",
    "get_secrets",
    "get_dynamic_db_creds",
    "get_default_client",
    "reset_default_client",
    "VaultError",
    "VaultAuthError",
    "VaultSecretNotFound",
    "VaultDBCredsError",
    "VaultConfigError",
]

_default_client: VaultClient | None = None
_default_lock = threading.Lock()


def get_default_client() -> VaultClient:
    """Return (and lazily create) the env-configured process singleton.

    Reads VAULT_ADDR / VAULT_ROLE / VAULT_K8S_TOKEN_PATH from the
    environment. Raises VaultConfigError on the first Vault call if
    VAULT_ROLE is unset -- not here, so importing the module is safe
    in environments that don't use Vault.
    """
    global _default_client
    if _default_client is None:
        with _default_lock:
            if _default_client is None:
                _default_client = VaultClient()
    return _default_client


def reset_default_client() -> None:
    """Drop the process-global client. Mainly for tests."""
    global _default_client
    with _default_lock:
        _default_client = None


def get_secret(path: str, key: str, *, mount_point: str = "secret") -> str:
    """Read a single KV-v2 key via the default client."""
    return get_default_client().read_kv(path, key, mount_point=mount_point)


def get_secrets(path: str, *, mount_point: str = "secret") -> dict[str, Any]:
    """Read every KV-v2 key at a path via the default client."""
    return get_default_client().read_kv_all(path, mount_point=mount_point)


def get_dynamic_db_creds(
    role: str,
    *,
    mount_point: str = "database",
    renewal_margin_seconds: int = 300,
    force_refresh: bool = False,
) -> DynamicDBCreds:
    """Fetch (or reuse cached) dynamic DB creds via the default client."""
    return get_default_client().get_db_creds(
        role,
        mount_point=mount_point,
        renewal_margin_seconds=renewal_margin_seconds,
        force_refresh=force_refresh,
    )
