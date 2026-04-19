"""LLM compose -- turn a gate "yes" into a 1-3 sentence user-facing message.

Ported from ``skynet_agent.modules.impulse.compose``; generalised via a
``ComposeClient`` Protocol so each domain injects its own LLM client + prompt
template. ``DefaultOpenAIComposeClient`` wraps an ``openai.OpenAI``-shaped
``client.chat.completions.create`` call for consumers that already have one.

The compose step knows about the selected ``Archetype`` (from the bandit) so
its system prompt can say "be playful and short" or "be curious and medium-
length". This is the single integration point where the bandit output feeds
the LLM.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Protocol

from .archetypes import Archetype
from .prompts import (
    DEFAULT_COMPOSE_SYSTEM_PROMPT,
    DEFAULT_COMPOSE_USER_TEMPLATE,
)
from .signals import Signal

log = logging.getLogger(__name__)


class ComposeClient(Protocol):
    """Turns (domain, drive, anchor, archetype, context) into a short message."""

    async def compose(
        self,
        *,
        domain: str,
        dominant_drive: str,
        anchor: Optional[str],
        archetype: Archetype,
        context: dict,
    ) -> str: ...


def _signals_blurb(signals: list[Signal], limit: int = 5) -> str:
    if not signals:
        return "(жодних конкретних подій, просто настрій)"
    lines = []
    for s in signals[:limit]:
        payload = json.dumps(s.payload, ensure_ascii=False, default=str)[:200]
        lines.append(f"- {s.kind} ({s.source}, anchor={s.anchor or '-'}): {payload}")
    return "\n".join(lines)


def format_compose_prompt(
    *,
    domain: str,
    dominant_drive: str,
    anchor: Optional[str],
    archetype: Archetype,
    context: dict,
    voice_hint: str = "",
    system_template: str = DEFAULT_COMPOSE_SYSTEM_PROMPT,
    user_template: str = DEFAULT_COMPOSE_USER_TEMPLATE,
) -> tuple[str, str]:
    """Render (system, user) for the compose call.

    ``context`` is expected to contain ``drives``, ``signals`` (list[Signal]),
    and optionally ``reason`` (gate's one-liner). Missing fields are handled
    gracefully so a caller who only has drives can still compose.
    """
    drives = context.get("drives", {})
    signals: list[Signal] = context.get("signals", [])
    reason = context.get("reason", "")
    system = system_template.format(
        domain=domain,
        voice_hint=(voice_hint + "\n") if voice_hint else "",
        tone=archetype.tone,
        anchor=anchor or "(none)",
        archetype=archetype.name,
        reason=reason,
    )
    user = user_template.format(
        drives=drives,
        signals_blurb=_signals_blurb(signals),
    )
    return system, user


class DefaultOpenAIComposeClient:
    """Wraps ``client.chat.completions.create`` for a sync OpenAI-shaped client.

    Sync underneath (the original agent called this from its sync tick loop)
    but exposed as ``async def compose`` to match the Protocol. If your
    client is truly async-native, implement ``ComposeClient`` directly --
    don't wrap it twice.
    """

    def __init__(
        self,
        llm_client,
        *,
        model: str,
        temperature: float = 0.8,
        max_tokens: int = 400,
        length_bucket_tokens: dict[str, int] | None = None,
        voice_hint: str = "",
        system_template: str = DEFAULT_COMPOSE_SYSTEM_PROMPT,
        user_template: str = DEFAULT_COMPOSE_USER_TEMPLATE,
    ):
        self._client = llm_client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Archetype "length" -> token ceiling. Override via ctor if your bot
        # wants shorter/longer blurbs per archetype than these defaults.
        self._length_tokens = length_bucket_tokens or {
            "short": 120,
            "medium": 250,
            "long": 400,
        }
        self._voice_hint = voice_hint
        self._system_template = system_template
        self._user_template = user_template

    async def compose(
        self,
        *,
        domain: str,
        dominant_drive: str,
        anchor: Optional[str],
        archetype: Archetype,
        context: dict,
    ) -> str:
        system, user = format_compose_prompt(
            domain=domain,
            dominant_drive=dominant_drive,
            anchor=anchor,
            archetype=archetype,
            context=context,
            voice_hint=self._voice_hint,
            system_template=self._system_template,
            user_template=self._user_template,
        )
        max_toks = self._length_tokens.get(archetype.length, self._max_tokens)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self._temperature,
                max_tokens=max_toks,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            log.warning("compose failed: %s", e)
            return ""


__all__ = [
    "ComposeClient",
    "DefaultOpenAIComposeClient",
    "format_compose_prompt",
]
