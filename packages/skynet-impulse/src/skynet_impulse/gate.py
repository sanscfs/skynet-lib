"""Cheap binary "should I fire?" LLM gate.

Generalised from ``skynet_agent.modules.impulse.gate``: the original hit a
hard-coded Ollama URL + OpenRouter fallback and returned a dict with keys
``speak/tone/anchor/reason/gate_model``. Here the gate is a ``Protocol`` --
any object that implements ``should_fire(...) -> (bool, str)`` plugs in --
plus ``DefaultHttpGateClient`` for consumers that want the agent's original
two-backend behavior with per-domain prompts.

The Protocol's return is deliberately minimal: ``(decision, reasoning)``.
Anchor and tone are computed by the engine BEFORE the gate is called (anchor
comes from the top-salience signal, tone is sampled from the archetype
bandit), so the gate just says yes/no and why.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Protocol

from .prompts import DEFAULT_GATE_SYSTEM_PROMPT, DEFAULT_GATE_USER_TEMPLATE
from .signals import Signal

log = logging.getLogger(__name__)


class GateClient(Protocol):
    """Asks an LLM (or deterministic rule) whether to fire this tick."""

    async def should_fire(
        self,
        *,
        domain: str,
        dominant_drive: str,
        dominant_value: float,
        baseline_p75: float,
        anchor: Optional[str],
        recent_context: str,
        drives: dict[str, float],
        signals: list[Signal],
    ) -> tuple[bool, str]:
        """Return ``(fire, reasoning)``. Reasoning is free text for debug logs."""
        ...


def _signals_summary(signals: list[Signal], limit: int = 8) -> str:
    if not signals:
        return "(немає свіжих подій)"
    lines = []
    for s in signals[:limit]:
        payload = json.dumps(s.payload, ensure_ascii=False, default=str)[:150]
        lines.append(f"- {s.kind} (sal={s.salience:.2f}, src={s.source}, anchor={s.anchor or '-'}): {payload}")
    return "\n".join(lines)


def format_gate_prompt(
    *,
    domain: str,
    dominant_drive: str,
    dominant_value: float,
    baseline_p75: float,
    anchor: Optional[str],
    recent_context: str,
    drives: dict[str, float],
    signals: list[Signal],
    percentile: int = 75,
    voice_hint: str = "",
    system_template: str = DEFAULT_GATE_SYSTEM_PROMPT,
    user_template: str = DEFAULT_GATE_USER_TEMPLATE,
) -> tuple[str, str]:
    """Render the (system, user) pair for the gate LLM call.

    Exposed as a free function so callers with an exotic LLM client can
    assemble the request themselves (e.g. hit a streaming endpoint).
    """
    anchor_hint = f"Якір теми: {anchor}\n" if anchor else ""
    system = system_template.format(
        domain=domain,
        voice_hint=(voice_hint + "\n") if voice_hint else "",
    )
    user = user_template.format(
        drives=drives,
        dominant_drive=dominant_drive,
        dominant_value=dominant_value,
        baseline=baseline_p75,
        percentile=percentile,
        anchor_hint=anchor_hint,
        recent_context=recent_context[:300] or "(тиша)",
        signals_summary=_signals_summary(signals),
    )
    return system, user


class DefaultHttpGateClient:
    """OpenAI-compatible chat/completions gate with two-backend fallback.

    Use this when the consumer doesn't already have an LLM client it wants
    to reuse. Primary = local Ollama (``qwen3:4b``), fallback = OpenRouter-
    shaped endpoint (usually skynet-cache -> ``deepseek/deepseek-v3.2``).

    The client is synchronous internally (one round-trip per call, no
    streaming) but exposes ``should_fire`` as async so it matches the
    ``GateClient`` protocol and the engine's ``await`` call site.
    """

    def __init__(
        self,
        *,
        primary_url: str = "http://100.64.0.4:11434/v1",
        primary_model: str = "qwen3:4b",
        fallback_url: str = "http://skynet-cache.skynet-cache.svc:8080/v1",
        fallback_model: str = "deepseek/deepseek-v3.2",
        timeout: float = 10.0,
        api_key_env: str = "LLM_API_KEY",
        percentile: int = 75,
        voice_hint: str = "",
        system_template: str = DEFAULT_GATE_SYSTEM_PROMPT,
        user_template: str = DEFAULT_GATE_USER_TEMPLATE,
    ):
        self._primary_url = primary_url
        self._primary_model = primary_model
        self._fallback_url = fallback_url
        self._fallback_model = fallback_model
        self._timeout = timeout
        self._api_key_env = api_key_env
        self._percentile = percentile
        self._voice_hint = voice_hint
        self._system_template = system_template
        self._user_template = user_template

    async def should_fire(
        self,
        *,
        domain: str,
        dominant_drive: str,
        dominant_value: float,
        baseline_p75: float,
        anchor: Optional[str],
        recent_context: str,
        drives: dict[str, float],
        signals: list[Signal],
    ) -> tuple[bool, str]:
        import httpx  # deferred -- not a hard dep of the library

        system, user = format_gate_prompt(
            domain=domain,
            dominant_drive=dominant_drive,
            dominant_value=dominant_value,
            baseline_p75=baseline_p75,
            anchor=anchor,
            recent_context=recent_context,
            drives=drives,
            signals=signals,
            percentile=self._percentile,
            voice_hint=self._voice_hint,
            system_template=self._system_template,
            user_template=self._user_template,
        )

        api_key = os.getenv(self._api_key_env, "")
        fb_auth = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        attempts = [
            (self._primary_url, self._primary_model, {}),
            (self._fallback_url, self._fallback_model, fb_auth),
        ]

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for url, model, headers in attempts:
                try:
                    r = await client.post(
                        f"{url}/chat/completions",
                        headers=headers,
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                            "temperature": 0.3,
                            "max_tokens": 200,
                            "response_format": {"type": "json_object"},
                        },
                    )
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"]
                    verdict = json.loads(content)
                    fire = bool(verdict.get("speak", False))
                    reason = str(verdict.get("reason", ""))[:200]
                    return fire, f"model={model} reason={reason}"
                except Exception as e:  # noqa: BLE001
                    log.warning("gate %s failed: %s", model, e)
                    continue

        return False, "gate unavailable"


__all__ = [
    "GateClient",
    "DefaultHttpGateClient",
    "format_gate_prompt",
]
