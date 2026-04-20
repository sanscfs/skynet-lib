"""Route an OpenAI-compatible URL to its Vault-backed API key.

The mapping is intentionally substring-based so intermediate proxies
(skynet-cache fronting OpenRouter) still resolve to the right
downstream key without an extra config knob.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger("skynet_providers.resolver")

# Process-local cache of successful Vault reads. We explicitly do NOT
# cache empty/missing results — a pod that started before its Vault
# policy landed would otherwise stay blind to the key forever, which
# is exactly the 401 we hit in production the first time. Cache hit =
# the secret was actually fetched at least once.
_vault_cache: dict[tuple[str, str], str] = {}
_vault_cache_lock = threading.Lock()

# (url substring, vault KV path, secret key name)
# Ordered: more specific hosts first. Mistral's KV field name uses a
# dash ("api-key") while OpenRouter's uses an underscore ("api_key") —
# preserve whatever the operator originally wrote when placing each
# secret; don't try to rename either to force consistency.
_PROVIDER_REGISTRY: tuple[tuple[str, str, str], ...] = (
    ("api.mistral.ai", "mistral", "api-key"),
    ("openrouter.ai", "openrouter", "api_key"),
    ("skynet-cache", "openrouter", "api_key"),
)

# URLs that don't need a bearer token (Ollama on laptop/Mac/phone).
_LOCAL_MARKERS: tuple[str, ...] = (
    "localhost",
    "127.0.0.1",
    "100.64.0.",
    ":11434",
)


def is_local_endpoint(api_url: str) -> bool:
    """True for URLs that accept un-authenticated OpenAI-compatible calls.

    Currently these are Ollama instances on the Headscale mesh. The
    check is a plain substring match because the callers that matter
    (HyDE fallback, impulse gate) have full control over the URLs
    they pass in.
    """
    low = (api_url or "").lower()
    return any(marker in low for marker in _LOCAL_MARKERS)


def resolve_api_key(api_url: str) -> str:
    """Return the Vault-sourced API key matching ``api_url``.

    Empty string on miss — the caller must check and decide whether to
    raise. For local endpoints (Ollama) the empty string is the
    correct value, so ``chat_completion`` special-cases that path.
    """
    if not api_url:
        return ""
    low = api_url.lower()
    for needle, vault_path, secret_key in _PROVIDER_REGISTRY:
        if needle in low:
            return _read_vault(vault_path, secret_key)
    return ""


def _read_vault(path: str, key: str) -> str:
    """Return a Vault KV field, caching ONLY successful reads.

    Why not @lru_cache: during the first rollout of a new provider, a
    pod can come up before the matching Vault policy is in place; the
    read returns empty and lru_cache pins that empty forever. We want
    subsequent calls to retry transparently until the secret appears.

    On any error / empty result we return "" and the cache stays
    un-populated for (path, key). On a non-empty hit the value is
    stored in-process and reused for the pod's lifetime.
    """
    cache_key = (path, key)
    cached = _vault_cache.get(cache_key)
    if cached:
        return cached

    try:
        from skynet_vault import get_default_client

        client = get_default_client()
        client.authenticate()
        value = client.read_kv(path, key) or ""
    except Exception as e:
        # Log at debug — callers log at ERROR when they raise.
        logger.debug("Vault read %s.%s failed: %s", path, key, e)
        return ""

    if value:
        with _vault_cache_lock:
            _vault_cache[cache_key] = value
    return value


def _reset_cache() -> None:
    """Flush the Vault-read cache. Tests only."""
    with _vault_cache_lock:
        _vault_cache.clear()
