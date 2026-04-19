"""Route an OpenAI-compatible URL to its Vault-backed API key.

The mapping is intentionally substring-based so intermediate proxies
(skynet-cache fronting OpenRouter) still resolve to the right
downstream key without an extra config knob.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger("skynet_providers.resolver")

# (url substring, vault KV path, secret key name)
# Ordered: more specific hosts first.
_PROVIDER_REGISTRY: tuple[tuple[str, str, str], ...] = (
    ("api.mistral.ai", "mistral/api-key", "api_key"),
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


@lru_cache(maxsize=8)
def _read_vault(path: str, key: str) -> str:
    """Cached Vault KV read. Caches per (path, key) for the process lifetime.

    Cache size is small (8) because we only expect a handful of
    provider paths. If the token expires the underlying hvac client
    re-auths transparently; we never cache a specific token, only the
    final secret value. On any error we return an empty string so the
    caller raises ProviderAuthError rather than surfacing hvac internals.
    """
    try:
        from skynet_vault import get_default_client

        client = get_default_client()
        client.authenticate()
        return client.read_kv(path, key)
    except Exception as e:
        # Log at debug — callers log at ERROR when they raise.
        logger.debug("Vault read %s.%s failed: %s", path, key, e)
        return ""


def _reset_cache() -> None:
    """Flush the Vault-read LRU. Tests only."""
    _read_vault.cache_clear()
