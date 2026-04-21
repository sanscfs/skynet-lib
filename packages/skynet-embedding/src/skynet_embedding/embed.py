"""Main embedding functions with 3-tier fallback chain.

Fallback order:
1. Ollama (local, free, fast)
2. OpenRouter (remote, paid)
3. Hash-based deterministic fallback (no semantics, but never fails)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from skynet_embedding.providers import hash_embed, ollama_embed, openrouter_embed

logger = logging.getLogger(__name__)

_DEFAULT_DIM = int(os.getenv("EMBEDDING_DIM", "512"))


def embed(
    text: str,
    dim: int | None = None,
    *,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    embedding_model: str | None = None,
) -> list[float]:
    """Embed text using the 3-tier fallback chain.

    Returns a normalized vector of the specified dimension.
    Never raises -- always falls back to hash embedding.
    """
    dim = dim or _DEFAULT_DIM

    # Tier 1: Ollama (local)
    vec = ollama_embed(text, dim, url=ollama_url, model=ollama_model)
    if vec is not None:
        return vec

    # Tier 2: OpenRouter (remote)
    vec = openrouter_embed(text, dim, api_key=api_key, api_url=api_url, model=embedding_model)
    if vec is not None:
        return vec

    # Tier 3: Hash fallback (deterministic, no semantics)
    logger.debug("All embedding providers failed, using hash fallback")
    return hash_embed(text, dim)


def embed_with_tier(
    text: str,
    dim: int | None = None,
    *,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    embedding_model: str | None = None,
) -> tuple[list[float], str]:
    """Like embed(), but also returns the tier name that succeeded.

    Returns ``(vector, tier)`` where tier is one of:
    ``"ollama"`` | ``"openrouter"`` | ``"hash"``.
    Never raises.
    """
    dim = dim or _DEFAULT_DIM

    vec = ollama_embed(text, dim, url=ollama_url, model=ollama_model)
    if vec is not None:
        return vec, "ollama"

    vec = openrouter_embed(text, dim, api_key=api_key, api_url=api_url, model=embedding_model)
    if vec is not None:
        return vec, "openrouter"

    logger.debug("All embedding providers failed, using hash fallback")
    return hash_embed(text, dim), "hash"


def embed_cached(
    text: str,
    dim: int | None = None,
    *,
    redis_client: Any | None = None,
    ttl: int = 3600,
    **kwargs,
) -> list[float]:
    """Embed with Redis caching layer.

    If redis_client is provided and the embedding for this text is cached,
    returns the cached version. Otherwise computes, caches, and returns.
    """
    dim = dim or _DEFAULT_DIM
    if not text:
        return embed("", dim, **kwargs)

    cache_key = f"emb:q:{hashlib.sha256(text.encode()).hexdigest()[:16]}"

    # Try cache read
    if redis_client is not None:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug("embed_cached read failed: %s", e)

    # Compute
    vec = embed(text, dim, **kwargs)

    # Cache write
    if redis_client is not None:
        try:
            redis_client.setex(cache_key, ttl, json.dumps(vec))
        except Exception as e:
            logger.debug("embed_cached write failed: %s", e)

    return vec
