"""Emoji -> vibe-phrase lookup + runtime embedding cache.

The shipped ``emoji_vectors.json`` maps ~60 curated emoji to short vibe
phrases. Embeddings are not precomputed at build time (keeps the file
portable and embedder-agnostic) -- the first call to :func:`embed_emoji`
embeds the phrase through the injected embedder and caches the result
in-process for subsequent calls.
"""

from __future__ import annotations

import asyncio
import json
from importlib import resources
from typing import Awaitable, Callable

from skynet_vibe.exceptions import EmbeddingError


def _load_emoji_map() -> dict[str, str]:
    # importlib.resources handles both installed-wheel and src layouts.
    data = resources.files("skynet_vibe").joinpath("emoji_vectors.json").read_text(encoding="utf-8")
    return json.loads(data)


EMOJI_TO_PHRASE: dict[str, str] = _load_emoji_map()


_embedding_cache: dict[str, list[float]] = {}

Embedder = Callable[[str], Awaitable[list[float]] | list[float]]


def phrase_for(emoji: str) -> str | None:
    """Return the curated vibe phrase for ``emoji`` or None if unknown."""
    return EMOJI_TO_PHRASE.get(emoji)


async def embed_emoji(emoji: str, embedder: Embedder) -> list[float] | None:
    """Return the embedding of ``emoji``'s curated vibe phrase.

    Returns ``None`` if the emoji has no mapping. Caches by emoji to avoid
    re-embedding the same phrase across a process lifetime.

    Raises :class:`EmbeddingError` if the embedder returns an invalid vector.
    """
    phrase = EMOJI_TO_PHRASE.get(emoji)
    if not phrase:
        return None
    if emoji in _embedding_cache:
        return list(_embedding_cache[emoji])
    result = embedder(phrase)
    if asyncio.iscoroutine(result):
        result = await result  # type: ignore[assignment]
    if not isinstance(result, list) or not result:
        raise EmbeddingError(f"embedder returned invalid vector for emoji {emoji!r}")
    vec: list[float] = list(result)  # type: ignore[arg-type]
    _embedding_cache[emoji] = vec
    return list(vec)


def clear_cache() -> None:
    """Clear the process-lifetime embedding cache (useful for tests)."""
    _embedding_cache.clear()
