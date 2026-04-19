"""Exceptions raised by skynet_vault.

All errors inherit from VaultError so callers can catch the whole
family with a single except clause. None of the exception messages
should ever contain secret material -- only paths, role names, and
HTTP status codes.
"""

from __future__ import annotations


class VaultError(Exception):
    """Base class for all skynet_vault errors."""


class VaultAuthError(VaultError):
    """Kubernetes ServiceAccount auth to Vault failed.

    Raised when the SA token file is missing, the Vault role is
    unknown, or Vault rejects the JWT.
    """


class VaultSecretNotFound(VaultError):
    """A KV secret path or key inside it does not exist."""


class VaultDBCredsError(VaultError):
    """The Vault database secrets engine failed to issue creds.

    Typically means the Vault role is misconfigured or the DB
    backend is unreachable.
    """


class VaultConfigError(VaultError):
    """Required configuration (env var, constructor arg) is missing."""
