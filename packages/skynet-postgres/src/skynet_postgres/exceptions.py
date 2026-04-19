"""Exceptions raised by skynet-postgres."""

from __future__ import annotations


class SkynetPostgresError(Exception):
    """Base exception for skynet-postgres."""


class PoolNotStartedError(SkynetPostgresError):
    """Raised when the pool is used before ``start()`` has been awaited."""


class PoolClosedError(SkynetPostgresError):
    """Raised when the pool is used after ``close()`` has been awaited."""


class CredentialsRotationFailed(SkynetPostgresError):
    """Raised when a retry with fresh credentials still fails to authenticate.

    The pool gives up after a single rotation attempt -- if Vault is still
    handing out bad credentials or the database is refusing them, bailing is
    better than spinning forever.
    """
