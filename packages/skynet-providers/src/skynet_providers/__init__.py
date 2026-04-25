"""Skynet LLM provider abstraction.

Public surface:

    from skynet_providers import (
        chat_completion,
        async_chat_completion,
        resolve_api_key,
        ProviderError,
        ProviderAuthError,
    )

One module, one responsibility: take an OpenAI-compatible URL + model
and return assistant text. Every consumer that used to hand-roll an
`httpx.post("{url}/chat/completions", headers=..., json=...)` block
now calls `chat_completion(...)` instead, and the provider-specific
API key is resolved from Vault based on the URL.

Why URL-based dispatch:
  * callers only need to know which endpoint they want (Mistral direct
    vs OpenRouter vs skynet-cache proxy), not which Vault path holds
    the matching key
  * a deploy-time config change (`LLM_API_URL=https://api.mistral.ai/v1`)
    flips the downstream provider WITHOUT touching any code
  * local Ollama ("100.64.0.4:11434") needs no auth and is handled
    transparently — same call site works for cloud and on-box
"""

from __future__ import annotations

from .chat import async_chat_completion, chat_completion
from .exceptions import ProviderAuthError, ProviderError
from .model import (
    VALID_TIERS,
    InvalidModelOverride,
    LLMClient,
    ModelOverride,
    get_redis_override,
    parse_model_args,
    resolve_model,
    set_redis_override,
    slot_allows_override,
)
from .override import VALID_PROVIDERS, Endpoint, parse_override, resolve_endpoint
from .resolver import resolve_api_key

__all__ = [
    "chat_completion",
    "async_chat_completion",
    "resolve_api_key",
    "parse_override",
    "resolve_endpoint",
    "Endpoint",
    "VALID_PROVIDERS",
    "ProviderError",
    "ProviderAuthError",
    "ModelOverride",
    "VALID_TIERS",
    "InvalidModelOverride",
    "parse_model_args",
    "resolve_model",
    "slot_allows_override",
    "get_redis_override",
    "set_redis_override",
    "LLMClient",
]
