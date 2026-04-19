"""OpenAI-compatible /chat/completions caller.

Sync and async variants share the same signature; pick whichever
matches your event loop. Both functions raise ``ProviderError`` on
any HTTP failure so callers see real errors instead of empty
assistant strings.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .exceptions import ProviderAuthError, ProviderError
from .resolver import is_local_endpoint, resolve_api_key

logger = logging.getLogger("skynet_providers.chat")


def _build_messages(prompt: str, system: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _build_payload(
    prompt: str,
    system: str,
    model: str,
    temperature: float,
    max_tokens: int,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(prompt, system),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra:
        payload.update(extra)
    return payload


def _resolve_key(api_url: str, api_key: str | None) -> str:
    """Pick the key to use: explicit override > Vault lookup > none-for-local."""
    if api_key is not None:
        return api_key
    if is_local_endpoint(api_url):
        return ""
    key = resolve_api_key(api_url)
    if not key:
        raise ProviderAuthError(f"no API key resolved for {api_url}")
    return key


def _extract_content(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise ProviderError(f"no 'choices' in response: {str(body)[:200]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ProviderError(f"choice[0].message.content missing/non-string: {str(body)[:200]}")
    return content


def chat_completion(
    prompt: str,
    *,
    model: str,
    api_url: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    timeout: float = 60.0,
    api_key: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """POST to ``{api_url}/chat/completions`` and return assistant text.

    ``api_key=None`` (default) triggers the URL → Vault key dispatch.
    Pass an explicit string to bypass Vault (tests, CLI tools).

    ``extra`` is merged into the request body after the standard fields,
    so callers can set ``frequency_penalty``, ``presence_penalty``,
    ``response_format`` etc. without a new wrapper argument each time.
    """
    key = _resolve_key(api_url, api_key)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    payload = _build_payload(prompt, system, model, temperature, max_tokens, extra)

    url = f"{api_url.rstrip('/')}/chat/completions"
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    except httpx.HTTPError as e:
        raise ProviderError(f"transport error for {url}: {e}") from e

    if resp.status_code >= 400:
        raise ProviderError(
            f"upstream {resp.status_code} for {url}: {resp.text[:200]}"
        )
    try:
        return _extract_content(resp.json())
    except ProviderError:
        raise
    except Exception as e:
        raise ProviderError(f"malformed response from {url}: {e}") from e


async def async_chat_completion(
    prompt: str,
    *,
    model: str,
    api_url: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    timeout: float = 60.0,
    api_key: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Async version of :func:`chat_completion`. Same semantics."""
    key = _resolve_key(api_url, api_key)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    payload = _build_payload(prompt, system, model, temperature, max_tokens, extra)

    url = f"{api_url.rstrip('/')}/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as e:
        raise ProviderError(f"transport error for {url}: {e}") from e

    if resp.status_code >= 400:
        raise ProviderError(
            f"upstream {resp.status_code} for {url}: {resp.text[:200]}"
        )
    try:
        return _extract_content(resp.json())
    except ProviderError:
        raise
    except Exception as e:
        raise ProviderError(f"malformed response from {url}: {e}") from e
