"""Reusable health check helpers for Skynet services."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_redis(client) -> bool:
    """Check Redis connectivity. Returns True if ping succeeds."""
    if client is None:
        return False
    try:
        return client.ping()
    except Exception as e:
        logger.warning("Redis health check failed: %s", e)
        return False


def check_qdrant(url: str) -> bool:
    """Check Qdrant connectivity via HTTP. Returns True if reachable."""
    try:
        import httpx

        resp = httpx.get(f"{url}/collections", timeout=5)
        return resp.status_code == 200
    except Exception as e:
        logger.warning("Qdrant health check failed: %s", e)
        return False
