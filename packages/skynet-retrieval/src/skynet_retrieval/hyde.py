"""HyDE query expansion: generate a hypothetical answer to the user's
query and embed IT instead of (or alongside) the literal query.

Introduced in Gao et al. 2022 (`Precise Zero-Shot Dense Retrieval
without Relevance Labels`) as a cheap way to fix short / vague /
cross-lingual queries: an LLM's guess at "what a correct answer would
look like" sits much closer in embedding space to actual answer
documents than the raw question does.

In Skynet the cost model is different -- we're not using GPT-4 for
HyDE, we use qwen3:4b on the laptop over Ollama, which is free. The
expansion runs behind Redis cache keyed by the query hash, so
repeated questions hit the cache and skip the LLM entirely.

Personalisation: we pass the tier-1 skeleton (name, current session,
top traits) into the prompt so the hypothetical answer is phrased in
the user's own context. This matters when the user asks
"what did I decide about X" -- a generic LLM guess would drift, but a
skeleton-aware one stays close to the user's own vocabulary and
project names.

This module is intentionally ABSTRACT: it does not import httpx, it
does not import redis. Callers inject:

  - `llm_client`: a callable `llm_client(prompt, model) -> str` that
    talks to Ollama / OpenRouter / whatever. Raising or returning an
    empty string both count as "no HyDE" and fall back cleanly.
  - `cache`: any object with `get(key)` and `set(key, value, ttl)`
    methods, or `None` to disable caching entirely. In production the
    caller passes a `HydeCache` wrapper around redis.

This keeps the library importable from tests with zero network
dependencies, and keeps the identity pod in charge of timeouts,
retries, observability, and failure budgets.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


class HydeCache(Protocol):
    """Duck-typed cache interface for HyDE outputs.

    Any object with these three methods works:
    - get(key) -> str | None
    - set(key, value, ttl) -> None

    In production this is wrapped around `redis.Redis` with a TTL
    applied per call so stale expansions clear themselves out
    without manual eviction.
    """

    def get(self, key: str) -> str | None: ...  # pragma: no cover
    def set(self, key: str, value: str, ttl: int) -> None: ...  # pragma: no cover


LlmClient = Callable[[str, str], str]
"""Signature of the caller-supplied LLM client.

Takes `(prompt, model)` and returns the raw generated text. Must NOT
raise for network errors -- it's easier for the caller to wrap its
own httpx/ollama client and return "" on failure than for this
library to catch specific exception types.
"""


# Stable cache key namespace so rotating other Redis keys doesn't
# invalidate HyDE expansions, and vice versa.
CACHE_NAMESPACE = "hyde:"

#: Default Ollama model for HyDE expansion. qwen3:4b runs in ~3s on
#: the laptop RTX 3050 Ti and produces usable one-paragraph guesses.
#: Overridable per-call or via identity env var.
DEFAULT_HYDE_MODEL = "qwen3:4b"

#: Max characters from the skeleton we feed into the prompt. Enough
#: to carry a handful of traits and the current project name, but
#: tight enough that HyDE latency doesn't blow up on long profiles.
SKELETON_CHAR_BUDGET = 800

#: Max characters of the anchor (previous assistant sentence) we
#: include. Long anchors drown the query; we truncate from the end so
#: the most recent conversational context survives.
ANCHOR_CHAR_BUDGET = 400


def _cache_key(query: str, skeleton: str | None, anchor: str | None, model: str) -> str:
    """Deterministic SHA-1 cache key covering every input that changes
    the hypothetical answer.

    We hash the full input rather than just the query because the
    skeleton rotates as the user's profile evolves -- reusing an
    expansion generated against an old skeleton would drift in the
    wrong direction over time.
    """
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(query.encode("utf-8"))
    h.update(b"\x00")
    h.update((skeleton or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((anchor or "").encode("utf-8"))
    return CACHE_NAMESPACE + h.hexdigest()


def _build_prompt(query: str, skeleton: str | None, anchor: str | None) -> str:
    """Assemble the HyDE prompt.

    Kept as a separate function so tests can assert on the exact
    prompt shape (and so the atlas formulas catalog can render it
    verbatim in /constants without parsing the whole module).
    """
    lines = [
        "You are expanding a user query for document retrieval.",
        "Write a SHORT hypothetical passage (max 3 sentences) that a",
        "perfectly-matching retrieved document might contain. Use the",
        "user's own vocabulary from the context below. Do not answer",
        "the question yourself; imagine a note in the user's personal",
        "knowledge base that would be THE answer, and write that.",
    ]
    if skeleton:
        trimmed = skeleton[:SKELETON_CHAR_BUDGET]
        lines.append("")
        lines.append("User context:")
        lines.append(trimmed)
    if anchor:
        trimmed = anchor[-ANCHOR_CHAR_BUDGET:]
        lines.append("")
        lines.append("Previous turn (for topic anchoring, NOT to answer):")
        lines.append(trimmed)
    lines.append("")
    lines.append(f"User query: {query}")
    lines.append("")
    lines.append("Hypothetical passage:")
    return "\n".join(lines)


def hyde_expand(
    query: str,
    llm_client: LlmClient,
    skeleton: str | None = None,
    anchor: str | None = None,
    cache: HydeCache | None = None,
    model: str = DEFAULT_HYDE_MODEL,
    cache_ttl: int = 86400,
) -> str:
    """Return a hypothetical passage for `query`, using cache if present.

    Returns "" on empty input or on LLM failure. Callers should check
    for empty and skip adding a HyDE candidate to multi_search rather
    than feeding an empty string into the embedder.

    Parameters
    ----------
    query:
        The user's original query string.
    llm_client:
        Callable `(prompt, model) -> str`. Must not raise.
    skeleton:
        Tier-1 context skeleton (name, active session, top traits).
        Optional but strongly recommended -- it's what makes the
        expansion personalised rather than generic.
    anchor:
        Optional previous-turn assistant sentence for topic continuity.
        Included in the prompt as context but NOT blended into the
        embedding (the caller handles that separately in multi_search).
    cache:
        Optional cache object (see `HydeCache`). `None` disables
        caching, which is fine for tests but wasteful in production.
    model:
        Ollama model tag to pass to the LLM client. Default qwen3:4b.
    cache_ttl:
        Seconds to keep a cached expansion. Default 24h -- long enough
        for a day of conversation reuse, short enough that a rotating
        user profile eventually invalidates stale context.

    Returns
    -------
    str
        Hypothetical passage suitable for `embed(...)`, or "" on
        failure / empty input.
    """
    if not query or not query.strip():
        return ""

    key = _cache_key(query, skeleton, anchor, model)
    if cache is not None:
        try:
            cached = cache.get(key)
        except Exception as e:
            logger.warning("hyde cache.get failed: %s", e)
            cached = None
        if cached:
            return cached

    prompt = _build_prompt(query, skeleton, anchor)
    try:
        expansion = llm_client(prompt, model) or ""
    except Exception as e:
        logger.warning("hyde llm_client failed: %s", e)
        return ""

    expansion = expansion.strip()
    if not expansion:
        return ""

    if cache is not None:
        try:
            cache.set(key, expansion, cache_ttl)
        except Exception as e:
            logger.warning("hyde cache.set failed: %s", e)

    return expansion
