"""Conversation-history factory for ChatAgent.

Wraps skynet_core.conversation (conv_load / conv_append) and
skynet_providers.async_chat_completion into the three callables that
ChatAgent expects: history_loader, history_appender, history_llm_call.

Usage (in a service's build_chat_agent)::

    from skynet_matrix.history_llm import build_conv_history

    hist_llm, hist_loader, hist_appender = build_conv_history(
        model=settings.llm_model,
        api_url=settings.llm_base_url,
        api_key=api_key,
        fallback_model=settings.llm_fallback_model or None,
        fallback_api_url=settings.llm_fallback_base_url or None,
        fallback_api_key=fallback_api_key,
        extra={"response_format": {"type": "json_object"}},
    )

    return ChatAgent(
        ...,
        history_llm_call=hist_llm,
        history_loader=hist_loader,
        history_appender=hist_appender,
    )
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from skynet_core.conversation import conv_append, conv_load
from skynet_providers import async_chat_completion


def build_conv_history(
    *,
    model: str,
    api_url: str,
    api_key: str,
    temperature: float = 0.2,
    max_tokens: int = 400,
    timeout: float = 60.0,
    extra: dict[str, Any] | None = None,
    fallback_model: str | None = None,
    fallback_api_url: str | None = None,
    fallback_api_key: str | None = None,
    max_context_messages: int = 12,
) -> tuple[
    Callable,  # history_llm_call(system, user, messages) -> Awaitable[str]
    Callable,  # history_loader(room_id, thread_root) -> list[dict]
    Callable,  # history_appender(room_id, role, content, thread_root) -> None
]:
    """Return (history_llm_call, history_loader, history_appender) for ChatAgent.

    History is stored in Redis under ``skynet:conv:{room_id}:main`` keys
    (the existing skynet-core convention). Up to ``max_context_messages``
    turns are passed to the LLM; the store may hold more (governed by
    CONV_MAX_MESSAGES env var, default 40).
    """

    async def history_llm_call(system: str, user: str, messages: list[dict[str, Any]]) -> str:
        return await async_chat_completion(
            prompt=user,
            messages=messages,
            model=model,
            api_url=api_url,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key=api_key,
            extra=extra,
            fallback_model=fallback_model,
            fallback_api_url=fallback_api_url,
            fallback_api_key=fallback_api_key,
        )

    def loader(room_id: str, thread_root: Optional[str]) -> list[dict]:
        """Load unified room history.

        Matrix clients often start a new thread per bot reply, so
        thread-only history drops cross-thread context (e.g. "again" /
        "that was great" after a prior recommendation in a different
        thread). ``appender`` mirrors every message to the main key, so
        loading main is the complete room timeline regardless of which
        thread the current message lives in.
        """
        return conv_load(room_id, None, max_messages=max_context_messages)

    def appender(room_id: str, role: str, content: str, thread_root: Optional[str]) -> None:
        """Append to the main room timeline; mirror into the thread key.

        The thread key is kept populated for backwards-compatible
        thread-scoped lookups (older callers / UIs), but the canonical
        conversation history for LLM context is the main key.
        """
        conv_append(room_id, role, content, thread_root=None)
        if thread_root:
            conv_append(room_id, role, content, thread_root=thread_root)

    return history_llm_call, loader, appender


__all__ = ["build_conv_history"]
