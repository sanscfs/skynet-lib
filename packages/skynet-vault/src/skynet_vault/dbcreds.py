"""Dynamic DB credentials issued by Vault's database secrets engine.

Vault's `/v1/database/creds/<role>` endpoint returns a short-lived
username/password pair with a `lease_duration` (seconds). This module
models the lease so callers can decide when to renew before
expiry.

The renewal math is intentionally conservative: we treat the
credentials as expired `margin_seconds` (default 300s / 5 min) before
Vault's stated TTL. That way the application refreshes while the old
creds still work, avoiding the race where a pool hands out a
just-revoked password.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DynamicDBCreds:
    """Immutable snapshot of a dynamic DB credential lease.

    Attributes:
        username: Vault-generated DB user.
        password: Vault-generated DB password. Never logged.
        lease_id: Vault lease identifier (for explicit revoke/renew).
        lease_duration: Seconds the lease is valid for, as of `issued_at`.
        issued_at: Unix timestamp when the creds were fetched.
        role: Vault database role name that produced these creds.
    """

    username: str
    password: str = field(repr=False)
    lease_id: str
    lease_duration: int
    issued_at: float
    role: str

    @property
    def expires_at(self) -> float:
        """Unix timestamp at which Vault considers the lease expired."""
        return self.issued_at + self.lease_duration

    def is_expired(self, now: float | None = None) -> bool:
        """True once the lease's nominal TTL has elapsed."""
        t = time.time() if now is None else now
        return t >= self.expires_at

    def needs_renewal(self, margin_seconds: int = 300, now: float | None = None) -> bool:
        """True when we are within `margin_seconds` of expiry.

        Default margin is 5 minutes, which matches the rollover window
        used by skynet-movies in production.
        """
        t = time.time() if now is None else now
        return t >= (self.expires_at - margin_seconds)
