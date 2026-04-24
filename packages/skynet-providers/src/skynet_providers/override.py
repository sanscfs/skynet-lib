"""Operator-set model overrides, shared across every Skynet chat agent.

Every agent that exposes a ``/model`` (or equivalent) command stores the
operator's choice as a single string in its own Redis key, then routes
LLM traffic to the correct upstream.  Doing the parsing + URL lookup in
one place here means skynet-agent, skynet-profile-synthesis and the
future skynet-vibe-agent all share the same provider vocabulary.

Override string format:

    ``<provider>:<slug>``

where ``provider`` is one of ``local`` / ``mistral`` / ``openrouter`` /
``phala`` and ``slug`` is whatever model identifier the upstream accepts.
Examples::

    local:gemma3:27b                       -> Ollama on the Mac
    mistral:mistral-large-latest           -> Mistral La Plateforme
    openrouter:mistralai/mistral-large-2512 -> OpenRouter via skynet-cache
    phala:z-ai/glm-5.1                     -> OpenRouter + TEE-pinned to Phala

A bare string without a ``:`` prefix is treated as a legacy cloud slug;
the caller can still resolve it against its own ``fallback_url``.
Empty string means "no override, use defaults".

The endpoint URLs are env-overridable per pod so one agent can tunnel
OpenRouter through skynet-cache while another talks direct::

    LLM_LOCAL_URL         default http://100.64.0.4:11434/v1
    MISTRAL_API_URL       default https://api.mistral.ai/v1
    OPENROUTER_API_URL    default http://skynet-cache.skynet-cache.svc:8080/v1
    PHALA_API_URL         defaults to OPENROUTER_API_URL (same transport)

``phala`` shares OpenRouter's transport and Vault key but forces the
``provider.order=["phala"]`` routing preference in every request body,
so OpenRouter pins the call to the Phala TEE endpoint instead of
auto-selecting any provider that hosts the model. ``allow_fallbacks`` is
disabled — if Phala can't serve the request we'd rather fail than
silently leak through a non-TEE provider.

The API key is resolved via :func:`resolve_api_key` (URL substring →
Vault path) for everything except ``local``, which passes the literal
string ``"ollama"`` because Ollama doesn't check bearer tokens.
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple

from .resolver import is_local_endpoint, resolve_api_key

VALID_PROVIDERS: tuple[str, ...] = ("local", "mistral", "openrouter", "phala")

_ENDPOINT_DEFAULTS: dict[str, tuple[str, str, str | None]] = {
    # provider -> (env var name, default URL, canned api_key or None for Vault)
    "local": ("LLM_LOCAL_URL", "http://100.64.0.4:11434/v1", "ollama"),
    "mistral": ("MISTRAL_API_URL", "https://api.mistral.ai/v1", None),
    "openrouter": (
        "OPENROUTER_API_URL",
        "http://skynet-cache.skynet-cache.svc:8080/v1",
        None,
    ),
    "phala": (
        "PHALA_API_URL",
        "http://skynet-cache.skynet-cache.svc:8080/v1",
        None,
    ),
}

_PHALA_ROUTING: dict[str, Any] = {
    "provider": {"order": ["phala"], "allow_fallbacks": False},
}


class Endpoint(NamedTuple):
    """Resolved routing tuple ready to feed into an OpenAI-compat client."""

    base_url: str
    api_key: str
    model: str
    is_local: bool
    extra_body: dict[str, Any] | None = None


def parse_override(override: str) -> tuple[str, str]:
    """Split ``'<provider>:<slug>'`` into ``(provider, slug)``.

    An empty override returns ``('', '')``. A string without a ``:``
    separator returns ``('', <whole string>)`` so the caller can treat
    it as a legacy cloud slug.  Ollama tags contain a ``:`` of their
    own (``gemma3:27b``); ``partition`` keeps them intact because we
    split on the *first* ``:`` only.
    """
    if not override:
        return ("", "")
    if ":" not in override:
        return ("", override)
    provider, _, slug = override.partition(":")
    return (provider, slug)


def resolve_endpoint(
    provider: str,
    model: str,
    *,
    fallback_url: str = "",
    fallback_model: str = "",
) -> Endpoint:
    """Return the endpoint for ``(provider, model)``.

    When ``provider`` is empty, falls through to ``fallback_url`` with
    the API key resolved by URL substring and ``model`` (or, if blank,
    ``fallback_model``). This is the "no override" or "legacy bare
    slug" path.

    Raises :class:`ValueError` on an unknown provider or a missing
    fallback when needed.
    """
    if not provider:
        if not fallback_url:
            raise ValueError("no provider and no fallback_url")
        return Endpoint(
            base_url=fallback_url,
            api_key=resolve_api_key(fallback_url),
            model=model or fallback_model,
            is_local=is_local_endpoint(fallback_url),
        )

    if provider not in _ENDPOINT_DEFAULTS:
        raise ValueError(f"unknown provider '{provider}'; expected one of {VALID_PROVIDERS}")

    env_var, default_url, canned_key = _ENDPOINT_DEFAULTS[provider]
    url = os.environ.get(env_var) or default_url
    # phala rides OpenRouter's transport but pins the call to the Phala
    # TEE via provider.order. If PHALA_API_URL wasn't set explicitly we
    # fall through to OPENROUTER_API_URL so operators only have to
    # configure one base URL.
    extra_body: dict[str, Any] | None = None
    if provider == "phala":
        if not os.environ.get(env_var):
            url = os.environ.get("OPENROUTER_API_URL") or default_url
        extra_body = _PHALA_ROUTING
    if canned_key is not None:
        return Endpoint(url, canned_key, model or fallback_model, True, extra_body)
    return Endpoint(
        url,
        resolve_api_key(url),
        model or fallback_model,
        is_local_endpoint(url),
        extra_body,
    )
