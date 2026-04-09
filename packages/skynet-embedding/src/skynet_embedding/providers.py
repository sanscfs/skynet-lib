"""Embedding providers: Ollama, OpenRouter, deterministic hash fallback."""

from __future__ import annotations

import hashlib
import logging
import os

import httpx

from skynet_embedding.normalize import truncate_and_normalize

logger = logging.getLogger(__name__)


def ollama_embed(
    text: str,
    dim: int = 512,
    *,
    url: str | None = None,
    model: str | None = None,
    timeout: float = 10.0,
) -> list[float] | None:
    """Get embedding from Ollama. Returns None on failure."""
    url = url or os.getenv("OLLAMA_URL", "http://100.64.0.3:11434")
    model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{url}/api/embeddings",
                json={"model": model, "prompt": text[:8000]},
            )
            if resp.is_success:
                full = resp.json().get("embedding", [])
                if len(full) >= dim:
                    return truncate_and_normalize(full, dim)
    except Exception as e:
        logger.debug("Ollama embed failed: %s", e)
    return None


def openrouter_embed(
    text: str,
    dim: int = 512,
    *,
    api_key: str | None = None,
    api_url: str | None = None,
    model: str | None = None,
    timeout: float = 15.0,
) -> list[float] | None:
    """Get embedding from OpenRouter (OpenAI-compatible API). Returns None on failure."""
    api_key = api_key or os.getenv("LLM_API_KEY", "")
    api_url = api_url or os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1")
    model = model or os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

    if not api_key:
        return None

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{api_url}/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "input": text[:8000]},
            )
            if resp.is_success:
                data = resp.json().get("data", [])
                if data:
                    full = data[0].get("embedding", [])
                    if len(full) >= dim:
                        return truncate_and_normalize(full, dim)
    except Exception as e:
        logger.debug("OpenRouter embed failed: %s", e)
    return None


def hash_embed(text: str, dim: int = 512) -> list[float]:
    """Deterministic hash-based pseudo-embedding (last-resort fallback).

    Produces a normalized vector that is consistent for the same input,
    but has no semantic meaning. Used when all real embedding providers
    are unavailable.
    """
    result: list[float] = []
    seed = 0
    while len(result) < dim:
        h = hashlib.sha512(f"{seed}:{text.lower()}".encode()).digest()
        result.extend([b / 255.0 - 0.5 for b in h])
        seed += 1
    result = result[:dim]
    norm = sum(x * x for x in result) ** 0.5
    return [x / norm for x in result] if norm > 0 else result


async def async_ollama_embed(
    text: str,
    dim: int = 512,
    *,
    url: str | None = None,
    model: str | None = None,
    timeout: float = 10.0,
) -> list[float] | None:
    """Async version of ollama_embed."""
    url = url or os.getenv("OLLAMA_URL", "http://100.64.0.3:11434")
    model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{url}/api/embeddings",
                json={"model": model, "prompt": text[:8000]},
            )
            if resp.is_success:
                full = resp.json().get("embedding", [])
                if len(full) >= dim:
                    return truncate_and_normalize(full, dim)
    except Exception as e:
        logger.debug("Async Ollama embed failed: %s", e)
    return None


async def async_openrouter_embed(
    text: str,
    dim: int = 512,
    *,
    api_key: str | None = None,
    api_url: str | None = None,
    model: str | None = None,
    timeout: float = 15.0,
) -> list[float] | None:
    """Async version of openrouter_embed."""
    api_key = api_key or os.getenv("LLM_API_KEY", "")
    api_url = api_url or os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1")
    model = model or os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

    if not api_key:
        return None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{api_url}/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "input": text[:8000]},
            )
            if resp.is_success:
                data = resp.json().get("data", [])
                if data:
                    full = data[0].get("embedding", [])
                    if len(full) >= dim:
                        return truncate_and_normalize(full, dim)
    except Exception as e:
        logger.debug("Async OpenRouter embed failed: %s", e)
    return None
