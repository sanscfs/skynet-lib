"""HMAC-signed caller tokens.

Trust boundary is intra-cluster (Headscale mesh) so we don't need
asymmetric crypto. HMAC-SHA256 with a shared secret pulled from
Vault is enough to:
- prove a token was minted by *some* legitimate caller (not a
  stray client-side fabrication);
- bind it to a specific ``invocation_id`` + ``caller`` so it
  can't be replayed under a different identity.

The shared secret lives at Vault path ``secret/orchestration/hmac``
and is bootstrapped by the same Vault role that grants Skynet
services Vault auth. Rotating it requires a coordinated restart of
all agent services -- live rotation is out of scope.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

ENV_SECRET = "ORCHESTRATION_HMAC_SECRET"
TOKEN_TTL_SECONDS = 600  # tokens valid for 10 minutes after mint


class TokenError(Exception):
    """Token verification failed: malformed, wrong HMAC, or expired."""


def _load_secret() -> bytes:
    raw = os.environ.get(ENV_SECRET, "")
    if not raw:
        raise RuntimeError(
            f"{ENV_SECRET} not set -- skynet-orchestration tokens require "
            "a shared HMAC secret (mount from Vault path "
            "secret/orchestration/hmac)"
        )
    return raw.encode("utf-8")


def _sign(payload: bytes, secret: bytes) -> str:
    sig = hmac.new(secret, payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")


def mint(*, invocation_id: str, caller: str, ts: float | None = None) -> str:
    """Build a fresh token for one outbound call.

    Format: ``v1.<base64(payload)>.<base64(hmac)>``. Payload carries
    ``invocation_id``, ``caller``, ``ts`` so the verifier can pin
    identity and reject expired tokens.
    """
    secret = _load_secret()
    payload = {
        "invocation_id": invocation_id,
        "caller": caller,
        "ts": ts if ts is not None else time.time(),
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")
    sig = _sign(payload_bytes, secret)
    return f"v1.{payload_b64}.{sig}"


def verify(token: str, *, invocation_id: str, caller: str) -> None:
    """Raise :class:`TokenError` if ``token`` is invalid for this call.

    Checks: format, HMAC match, identity match, TTL not expired.
    """
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "v1":
        raise TokenError("token format: expected v1.<payload>.<sig>")
    secret = _load_secret()
    try:
        payload_bytes = base64.urlsafe_b64decode(parts[1] + "==")
    except Exception as e:  # noqa: BLE001
        raise TokenError(f"invalid base64 payload: {e}") from None
    expected_sig = _sign(payload_bytes, secret)
    if not hmac.compare_digest(expected_sig, parts[2]):
        raise TokenError("HMAC signature mismatch")
    try:
        payload = json.loads(payload_bytes)
    except Exception as e:  # noqa: BLE001
        raise TokenError(f"payload not JSON: {e}") from None
    if payload.get("invocation_id") != invocation_id:
        raise TokenError("invocation_id binding mismatch")
    if payload.get("caller") != caller:
        raise TokenError("caller binding mismatch")
    age = time.time() - float(payload.get("ts", 0))
    if age > TOKEN_TTL_SECONDS:
        raise TokenError(f"token expired ({age:.0f}s old, ttl={TOKEN_TTL_SECONDS}s)")
