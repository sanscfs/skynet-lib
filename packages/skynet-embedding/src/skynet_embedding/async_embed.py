"""Async embedding functions with 3-tier fallback chain."""

from __future__ import annotations

import hashlib
import json
import logging
import os

from skynet_embedding.providers import async_ollama_embed, async_openrouter_embed, hash_embed

logger = logging.getLogger(__name__)

_DEFAULT_DIM = int(os.getenv("EMBEDDING_DIM", "512"))


async def async_embed(
    text: str,
    dim: int | None = None,
    *,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    embedding_model: str | None = None,
) -> list[float]:
    """Async version of embed() with the same 3-tier fallback chain."""
    dim = dim or _DEFAULT_DIM

    vec = await async_ollama_embed(text, dim, url=ollama_url, model=ollama_model)
    if vec is not None:
        return vec

    vec = await async_openrouter_embed(text, dim, api_key=api_key, api_url=api_url, model=embedding_model)
    if vec is not None:
        return vec

    logger.debug("All async embedding providers failed, using hash fallback")
    return hash_embed(text, dim)


async def async_embed_cached(
    text: str,
    dim: int | None = None,
    *,
    redis_client=None,
    ttl: int = 3600,
    **kwargs,
) -> list[float]:
    """Async embed with Redis caching (requires redis.asyncio client)."""
    dim = dim or _DEFAULT_DIM
    if not text:
        return await async_embed("", dim, **kwargs)

    cache_key = f"emb:q:{hashlib.sha256(text.encode()).hexdigest()[:16]}"

    if redis_client is not None:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug("async embed_cached read failed: %s", e)

    vec = await async_embed(text, dim, **kwargs)

    if redis_client is not None:
        try:
            await redis_client.setex(cache_key, ttl, json.dumps(vec))
        except Exception as e:
            logger.debug("async embed_cached write failed: %s", e)

    return vec
