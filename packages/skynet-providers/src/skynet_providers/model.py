"""Tier-based model override for multi-slot, multi-component routing.

Three-level model selection — each level is optional and narrows the
previous one:

    tier      quality intent:  "big" | "medium" | "small"
    provider  LLM vendor:      "phala" | "mistral" | "local" | "openrouter"
    model     explicit slug:   "z-ai/glm-5.1" | "gemma3:27b" | …

Resolution order (first non-empty wins):
    explicit model + provider
    → tier → ConfigMap env (TIER_{TIER}_{PROVIDER}) → hardcoded default

Redis key schema:
    skynet:model:{component}:{slot}   e.g. skynet:model:agent:chat
    Value: JSON-encoded ModelOverride

Tier env vars (all readable from skynet-models ConfigMap):
    TIER_BIG_PHALA, TIER_BIG_LOCAL, TIER_BIG_OPENROUTER, TIER_BIG_MISTRAL
    TIER_MEDIUM_PHALA, TIER_MEDIUM_LOCAL, TIER_MEDIUM_OPENROUTER, TIER_MEDIUM_MISTRAL
    TIER_SMALL_PHALA, TIER_SMALL_LOCAL, TIER_SMALL_OPENROUTER, TIER_SMALL_MISTRAL
    TIER_BIG_DEFAULT_PROVIDER, TIER_MEDIUM_DEFAULT_PROVIDER, TIER_SMALL_DEFAULT_PROVIDER
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .override import Endpoint

logger = logging.getLogger("skynet_providers.model")

VALID_TIERS: frozenset[str] = frozenset({"big", "medium", "small"})

_REDIS_KEY_PREFIX = "skynet:model"

# Slots that cannot be changed via /model — require a migration process
_IMMUTABLE_SLOTS: frozenset[str] = frozenset({"embed"})

# Slots with a max tier cap (may not exceed this tier).
# hyde is capped at small: it generates short hypothetical passages (160
# tokens, temp 0.2) — big/medium models are overkill and expensive here.
# The provider is not restricted; local-with-cloud-fallback is the default
# behavior managed by HYDE_OPENROUTER_FALLBACK in skynet-identity.
_SLOT_MAX_TIER: dict[str, str] = {"rerank": "small", "hyde": "small"}

_TIER_ORDER: list[str] = ["small", "medium", "big"]

# Hardcoded defaults — overridden by TIER_* env vars from ConfigMap.
# Defaults target Phala TEE (via OpenRouter) because that's the
# production cloud provider as of the 2026-04 Mistral pause; `mistral`
# slugs are kept as an escape hatch for the day we un-pause.
_TIER_DEFAULTS: dict[str, dict[str, str]] = {
    "big": {
        "default_provider": "phala",
        "phala": "z-ai/glm-5.1",
        "local": "gemma3:27b",
        "openrouter": "z-ai/glm-5.1",
        "mistral": "mistral-large-latest",
    },
    "medium": {
        "default_provider": "phala",
        "phala": "z-ai/glm-4.7-flash",
        "local": "qwen3:14b",
        "openrouter": "z-ai/glm-4.7-flash",
        "mistral": "mistral-medium-latest",
    },
    "small": {
        "default_provider": "local",
        "phala": "google/gemma-3-27b-it",
        "local": "qwen3:1.7b",
        "openrouter": "google/gemma-3-27b-it",
        "mistral": "mistral-nemo-2407",
    },
}


@dataclass
class ModelOverride:
    tier: str | None = None
    provider: str | None = None
    model: str | None = None

    def is_empty(self) -> bool:
        return not any([self.tier, self.provider, self.model])

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None})

    @classmethod
    def from_json(cls, value: str) -> ModelOverride:
        try:
            return cls(**json.loads(value))
        except Exception:
            return cls()

    def display(self) -> str:
        """Human-readable string for Matrix responses."""
        parts = []
        if self.tier:
            parts.append(f"tier=**{self.tier}**")
        if self.provider:
            parts.append(f"provider=**{self.provider}**")
        if self.model:
            parts.append(f"`{self.model}`")
        return " / ".join(parts) if parts else "default"


def _tier_env(tier: str, provider: str) -> str:
    key = f"TIER_{tier.upper()}_{provider.upper()}"
    return os.environ.get(key) or _TIER_DEFAULTS.get(tier, {}).get(provider, "")


def _tier_default_provider(tier: str) -> str:
    key = f"TIER_{tier.upper()}_DEFAULT_PROVIDER"
    return os.environ.get(key) or _TIER_DEFAULTS.get(tier, {}).get("default_provider", "phala")


def resolve_model(
    override: ModelOverride,
    *,
    fallback_provider: str = "phala",
    fallback_model: str = "",
) -> tuple[str, str]:
    """Return ``(provider, model)`` from an override.

    Falls through to tier env vars, then hardcoded defaults, then
    ``fallback_*`` args (which come from cfg.LLM_FALLBACK_URL / cfg.CHAT_MODEL).
    """
    if override.is_empty():
        return (fallback_provider, fallback_model)

    # Fully explicit — short-circuit
    if override.model and override.provider:
        return (override.provider, override.model)

    provider = override.provider
    if not provider and override.tier:
        provider = _tier_default_provider(override.tier)
    provider = provider or fallback_provider

    model = override.model
    if not model and override.tier:
        model = _tier_env(override.tier, provider)
    model = model or fallback_model

    return (provider, model)


def parse_model_args(args: str) -> ModelOverride:
    """Parse the portion of a ``/model`` command after the optional target.

    Accepted forms (tokens after target is stripped by the caller):
        big                            tier only
        big mistral                    tier + provider
        big mistral mistral-large-2512 tier + provider + model
        mistral mistral-large-latest   provider + model (no tier)
        reset / default / (empty)      clear override
    """
    from .override import VALID_PROVIDERS  # avoid module-level circular

    tokens = args.strip().split()
    if not tokens or tokens[0].lower() in ("reset", "default"):
        return ModelOverride()

    first = tokens[0].lower()
    if first in VALID_TIERS:
        tier, rest = first, tokens[1:]
    else:
        tier, rest = None, tokens

    provider = model = None
    if rest:
        if rest[0].lower() in VALID_PROVIDERS:
            provider = rest[0].lower()
            model = " ".join(rest[1:]) or None
        else:
            model = " ".join(rest)

    return ModelOverride(tier=tier, provider=provider, model=model)


def slot_allows_override(slot: str, override: ModelOverride) -> tuple[bool, str]:
    """Return ``(allowed, reason_if_not)``."""
    if slot in _IMMUTABLE_SLOTS:
        return False, f"`{slot}` requires a migration — use `!embed-migrate` instead"
    if slot in _SLOT_MAX_TIER and override.tier:
        max_t = _SLOT_MAX_TIER[slot]
        if override.tier in _TIER_ORDER and max_t in _TIER_ORDER:
            if _TIER_ORDER.index(override.tier) > _TIER_ORDER.index(max_t):
                return False, f"`{slot}` is capped at **{max_t}**"
    return True, ""


def redis_key(component: str, slot: str) -> str:
    return f"{_REDIS_KEY_PREFIX}:{component}:{slot}"


class LLMClient:
    """Redis-backed LLM endpoint resolver for any Skynet component.

    Wraps tier/provider/model resolution + Vault key lookup into one object.
    Works in both sync and async contexts — ``endpoint()`` is always sync
    (Redis GET is a fast blocking call, cached for ``cache_ttl`` seconds).

    When ``redis`` is None the client degrades gracefully: no runtime
    overrides, just ``fallback_url``/``fallback_model`` from construction.
    Inject Redis later via ``set_redis()`` without rebuilding the object.

    Usage (async context)::

        ep = ctx.llm.endpoint()
        result = await async_chat_completion(
            prompt, model=ep.model, api_url=ep.base_url,
            api_key=ep.api_key or None,
        )

    Usage (sync context)::

        result = ctx.llm.complete_sync(prompt, max_tokens=1000)
    """

    def __init__(
        self,
        component: str,
        slot: str,
        *,
        redis=None,
        fallback_url: str = "",
        fallback_model: str = "",
        cache_ttl: float = 30.0,
    ) -> None:
        self.component = component
        self.slot = slot
        self._redis = redis
        self._fallback_url = fallback_url
        self._fallback_model = fallback_model
        self._cache_ttl = cache_ttl
        self._cached_override: ModelOverride | None = None
        self._cache_ts: float = 0.0

    def set_redis(self, redis) -> None:
        """Inject or replace Redis client; forces an immediate cache refresh."""
        self._redis = redis
        self._cache_ts = 0.0

    def _refresh_override(self) -> ModelOverride:
        now = time.time()
        if self._redis and (now - self._cache_ts) > self._cache_ttl:
            try:
                fresh = get_redis_override(self._redis, self.component, self.slot)
                self._cached_override = fresh
            except Exception:
                pass
            self._cache_ts = now
        return self._cached_override or ModelOverride()

    def endpoint(self) -> "Endpoint":
        """Return resolved Endpoint (cached for cache_ttl seconds).

        No override → fallback_url + fallback_model (env-based).
        Override with tier → resolves from TIER_* env vars in ConfigMap.
        Override with provider+model → fully explicit.

        When no override is set and ``fallback_url`` already routes to
        OpenRouter (direct or via skynet-cache), the resolver defaults
        the provider to ``phala`` so production cloud traffic lands in
        the TEE by default. Operators can still force plain OpenRouter
        with ``/model openrouter:<slug>`` or pure-local with
        ``/model local:<slug>``.
        """
        from .override import resolve_endpoint

        override = self._refresh_override()
        low = (self._fallback_url or "").lower()
        default_provider = "phala" if ("openrouter.ai" in low or "skynet-cache" in low) else ""
        provider, model = resolve_model(
            override,
            fallback_provider=default_provider,
            fallback_model=self._fallback_model,
        )
        return resolve_endpoint(
            provider,
            model,
            fallback_url=self._fallback_url,
            fallback_model=self._fallback_model,
        )

    def complete_sync(self, prompt: str, **kwargs) -> str:
        """Sync chat completion via resolved endpoint."""
        from .chat import chat_completion

        ep = self.endpoint()
        kwargs["extra"] = _merge_extra(ep.extra_body, kwargs.get("extra"))
        return chat_completion(
            prompt,
            model=ep.model,
            api_url=ep.base_url,
            api_key=ep.api_key or None,
            **kwargs,
        )

    async def complete(self, prompt: str, **kwargs) -> str:
        """Async chat completion via resolved endpoint."""
        from .chat import async_chat_completion

        ep = self.endpoint()
        kwargs["extra"] = _merge_extra(ep.extra_body, kwargs.get("extra"))
        return await async_chat_completion(
            prompt,
            model=ep.model,
            api_url=ep.base_url,
            api_key=ep.api_key or None,
            **kwargs,
        )


def _merge_extra(endpoint_extra: dict | None, caller_extra: dict | None) -> dict | None:
    """Merge endpoint-injected body fields with caller-supplied ones.

    Endpoint fields come from provider routing (e.g. Phala's
    ``provider.order`` pin). Caller fields come from the request
    itself (e.g. ``response_format``). Caller wins on key conflict so
    per-call overrides still work, which means operator choices like
    ``/model phala:...`` stay enforced only while the caller doesn't
    explicitly set the same key.
    """
    if not endpoint_extra and not caller_extra:
        return None
    if not endpoint_extra:
        return caller_extra
    if not caller_extra:
        return dict(endpoint_extra)
    merged = dict(endpoint_extra)
    merged.update(caller_extra)
    return merged


def get_redis_override(redis, component: str, slot: str) -> ModelOverride | None:
    """Read a ModelOverride from Redis. Returns None if not set."""
    try:
        raw = redis.get(redis_key(component, slot))
        if not raw:
            return None
        value = raw.decode() if isinstance(raw, bytes) else raw
        return ModelOverride.from_json(value)
    except Exception as e:
        logger.debug("Failed to read model override %s/%s: %s", component, slot, e)
        return None


def set_redis_override(redis, component: str, slot: str, override: ModelOverride) -> None:
    try:
        if override.is_empty():
            redis.delete(redis_key(component, slot))
        else:
            redis.set(redis_key(component, slot), override.to_json())
    except Exception as e:
        logger.warning("Failed to persist model override %s/%s: %s", component, slot, e)
