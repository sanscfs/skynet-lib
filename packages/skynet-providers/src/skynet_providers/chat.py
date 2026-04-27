"""OpenAI-compatible /chat/completions caller.

Sync and async variants share the same signature; pick whichever
matches your event loop. Both functions raise ``ProviderError`` on
any HTTP failure so callers see real errors instead of empty
assistant strings.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .exceptions import ProviderAuthError, ProviderError
from .resolver import is_local_endpoint, resolve_api_key

logger = logging.getLogger("skynet_providers.chat")

# Optional live-stream hooks — resolved lazily on the first call so
# ``sys.path`` tweaks (conftest, ad-hoc scripts) take effect before we
# commit to "skynet-matrix unavailable".
_hooks_cache: dict[str, Any] | None = None


def _live_hooks() -> dict[str, Any]:
    global _hooks_cache
    if _hooks_cache is not None:
        return _hooks_cache
    cache: dict[str, Any] = {"start": None, "token": None, "end": None}
    try:  # pragma: no cover - optional integration
        from skynet_matrix.async_live_stream import (  # noqa: PLC0415
            emit_llm_end_if_live,
            emit_llm_start_if_live,
            emit_token_if_live,
        )

        cache["start"] = emit_llm_start_if_live
        cache["token"] = emit_token_if_live
        cache["end"] = emit_llm_end_if_live
    except Exception:  # pragma: no cover
        pass
    _hooks_cache = cache
    return cache


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
    messages: list[dict[str, Any]] | None = None,
    *,
    api_url: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages if messages is not None else _build_messages(prompt, system),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra:
        # Ollama-served gpt-oss:20b (and likely other gpt-oss variants)
        # returns empty ``content`` when ``response_format={"type":
        # "json_object"}`` is set. The model still produces valid JSON
        # when the system prompt asks for it; the structured-output flag
        # is what poisons the output. Strip it for any local endpoint
        # so callers can pass the same ``extra`` to both primary
        # (Ollama) and cloud fallback paths without splitting kwargs.
        # Cloud endpoints keep the flag — they handle it correctly.
        if api_url and is_local_endpoint(api_url) and "response_format" in extra:
            extra = {k: v for k, v in extra.items() if k != "response_format"}
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
    """Pull assistant text out of a /chat/completions body, lenient about shape.

    OpenAI-compatible providers diverge on where the answer lives when
    structured output / reasoning models are in play:

    1. ``choices[0].message.content`` — canonical OpenAI shape.
    2. ``choices[0].message.tool_calls[0].function.arguments`` — some
       providers route JSON-mode output through a synthetic tool call
       under ``response_format={"type":"json_object"}``.
    3. ``choices[0].message.reasoning`` — DeepSeek-R1 / GLM-4.7 style.
       Glm-4.7 via OpenRouter+DeepInfra was observed (2026-04-27)
       returning ``content=null`` and the JSON answer in ``reasoning``.
       ``reasoning_details[0].text`` carries the same payload as a
       fallback for providers that omit ``reasoning`` itself.
    4. ``choices[0].text`` — legacy ``/completions`` shape some
       OpenAI-compatible proxies still emit.

    If none of the above yields a non-empty string, the original
    ProviderError is raised so callers can iterate. We deliberately do
    NOT silently swallow into an empty string.
    """
    choices = body.get("choices") or []
    if not choices:
        raise ProviderError(f"no 'choices' in response: {str(body)[:200]}")
    choice = choices[0]
    message = choice.get("message") or {}

    # Priority 1: canonical content
    content = message.get("content")
    if isinstance(content, str) and content != "":
        return content

    # Priority 2: tool_calls[0].function.arguments (JSON-mode routed via tool call)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        first = tool_calls[0] if isinstance(tool_calls[0], dict) else None
        fn = first.get("function") if first else None
        if isinstance(fn, dict):
            args = fn.get("arguments")
            if isinstance(args, str) and args != "":
                logger.info(
                    "extracted content from tool_calls[0].function.arguments (model=%s, finish=%s)",
                    body.get("model"),
                    choice.get("finish_reason"),
                )
                return args

    # Priority 3: reasoning / reasoning_details (DeepSeek-R1 / GLM-4.7)
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning != "":
        logger.info(
            "extracted content from message.reasoning (model=%s, finish=%s)",
            body.get("model"),
            choice.get("finish_reason"),
        )
        return reasoning
    rdetails = message.get("reasoning_details")
    if isinstance(rdetails, list) and rdetails:
        first = rdetails[0] if isinstance(rdetails[0], dict) else None
        if first:
            text = first.get("text")
            if isinstance(text, str) and text != "":
                logger.info(
                    "extracted content from reasoning_details[0].text (model=%s, finish=%s)",
                    body.get("model"),
                    choice.get("finish_reason"),
                )
                return text

    # Priority 4: legacy completion shape
    text = choice.get("text")
    if isinstance(text, str) and text != "":
        logger.info(
            "extracted content from choices[0].text (model=%s, finish=%s)",
            body.get("model"),
            choice.get("finish_reason"),
        )
        return text

    raise ProviderError(
        f"choice[0].message.content missing/non-string and no fallback "
        f"(tool_calls/reasoning/text) found: {str(body)[:200]}"
    )


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
    payload = _build_payload(prompt, system, model, temperature, max_tokens, extra, api_url=api_url)

    url = f"{api_url.rstrip('/')}/chat/completions"
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    except httpx.HTTPError as e:
        raise ProviderError(f"transport error for {url}: {e}") from e

    if resp.status_code >= 400:
        raise ProviderError(f"upstream {resp.status_code} for {url}: {resp.text[:200]}")
    try:
        return _extract_content(resp.json())
    except ProviderError:
        raise
    except Exception as e:
        raise ProviderError(f"malformed response from {url}: {e}") from e


async def _post_chat(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    model: str = "",
) -> str:
    """Send one POST to ``{url}/chat/completions`` and extract ``content``.

    Always uses SSE streaming (``stream=True``) so callers wired up with
    the ``skynet_matrix`` live-stream hooks see per-token updates.
    Providers that don't support ``stream`` (rare) still return a single
    ``choices[0].message.content`` block which we handle as the only
    SSE line. The returned value matches the non-streaming contract
    exactly — full accumulated assistant content as a string.
    """
    full = f"{url.rstrip('/')}/chat/completions"
    stream_payload = {**payload, "stream": True}
    hooks = _live_hooks()
    emit_start = hooks["start"]
    emit_token = hooks["token"]
    emit_end = hooks["end"]

    if emit_start is not None:
        try:
            await emit_start(url=full, model=model)
        except Exception:  # noqa: BLE001  — telemetry must never break the call
            pass

    pieces: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", full, headers=headers, json=stream_payload) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise ProviderError(
                        f"upstream {resp.status_code} for {full}: {body.decode(errors='replace')[:200]}"
                    )
                async for line in resp.aiter_lines():
                    token = _parse_sse_delta(line)
                    if token is None:
                        continue
                    if token == "":
                        continue
                    pieces.append(token)
                    if emit_token is not None:
                        try:
                            await emit_token(token)
                        except Exception:  # noqa: BLE001
                            pass
    except httpx.HTTPError as e:
        if emit_end is not None:
            try:
                await emit_end(tokens=0)
            except Exception:  # noqa: BLE001
                pass
        raise ProviderError(f"transport error for {full}: {e}") from e

    if emit_end is not None:
        try:
            await emit_end(tokens=len(pieces))
        except Exception:  # noqa: BLE001
            pass

    content = "".join(pieces).strip()
    if content:
        return content

    # Empty stream — common when the provider routed the answer through
    # ``choices[0].message.content`` (non-SSE body, even with stream=True
    # in the request) or when only ``tool_calls`` deltas were emitted.
    # One non-streaming retry recovers both shapes without spamming the
    # upstream. If THAT also returns nothing, raise as before.
    logger.warning(
        "empty stream from %s; retrying once with stream=False",
        full,
    )
    nostream_payload = {**payload, "stream": False}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(full, headers=headers, json=nostream_payload)
            if resp.status_code >= 400:
                raise ProviderError(f"upstream {resp.status_code} for {full}: {resp.text[:200]}")
            content = _extract_content(resp.json()).strip()
    except httpx.HTTPError as e:
        raise ProviderError(f"transport error on retry for {full}: {e}") from e

    if not content:
        raise ProviderError(f"empty content from {full}")
    return content


def _parse_sse_delta(line: str) -> str | None:
    """Pull the ``choices[0].delta.content`` out of one SSE line.

    Returns ``None`` for lines we want to skip (comments, empty,
    ``[DONE]`` sentinel, errors) and an empty string for a valid frame
    with no content delta (e.g. the first frame carrying a ``role``).
    """
    if not line:
        return None
    if line.startswith(":"):  # SSE comment / keepalive
        return None
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "" or data == "[DONE]":
        return None
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None
    choices = obj.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    token = delta.get("content")
    if token is None:
        return ""
    if not isinstance(token, str):
        return None
    return token


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
    messages: list[dict[str, Any]] | None = None,
    fallback_model: str | None = None,
    fallback_api_url: str | None = None,
    fallback_api_key: str | None = None,
) -> str:
    """Async version of :func:`chat_completion`.

    Optional one-shot fallback: if ``fallback_api_url`` (and usually
    ``fallback_model``) is provided, any :class:`ProviderError` from the
    primary attempt triggers a single retry against the fallback
    endpoint. The fallback uses the same ``system`` / ``prompt`` /
    ``temperature`` / ``max_tokens`` / ``extra`` as the primary —
    typical wiring is "Mistral direct → OpenRouter". If both fail, the
    exception from the fallback is raised (so callers can still see the
    last failure reason).
    """
    key = _resolve_key(api_url, api_key)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    payload = _build_payload(prompt, system, model, temperature, max_tokens, extra, messages, api_url=api_url)

    try:
        return await _post_chat(
            url=api_url,
            headers=headers,
            payload=payload,
            timeout=timeout,
            model=model,
        )
    except ProviderError as primary_exc:
        if not fallback_api_url:
            raise
        logger.warning(
            "primary LLM %s failed (%s); falling back to %s",
            api_url,
            str(primary_exc)[:200],
            fallback_api_url,
        )
        fb_key = _resolve_key(fallback_api_url, fallback_api_key)
        fb_headers = {"Authorization": f"Bearer {fb_key}"} if fb_key else {}
        fb_model = fallback_model or model
        fb_payload = _build_payload(
            prompt,
            system,
            fb_model,
            temperature,
            max_tokens,
            extra,
            messages,
            api_url=fallback_api_url,
        )
        return await _post_chat(
            url=fallback_api_url,
            headers=fb_headers,
            payload=fb_payload,
            timeout=timeout,
            model=fb_model,
        )
