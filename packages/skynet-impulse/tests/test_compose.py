"""ComposeClient + prompt formatter."""

from __future__ import annotations

import pytest
from skynet_impulse.archetypes import Archetype
from skynet_impulse.compose import format_compose_prompt
from skynet_impulse.signals import Signal


def test_format_compose_threads_archetype():
    arch = Archetype("novelty", "playful", "short")
    sig = Signal(kind="novelty", source="analyzer", salience=0.8, anchor="music:artist:x")
    sys, usr = format_compose_prompt(
        domain="music",
        dominant_drive="curiosity",
        anchor="music:artist:x",
        archetype=arch,
        context={"drives": {"curiosity": 0.7}, "signals": [sig], "reason": "anomaly"},
    )
    assert "playful" in sys
    assert "music:artist:x" in sys
    assert "anomaly" in sys
    assert "novelty" in usr


def test_format_compose_handles_empty_context():
    arch = Archetype("*", "reflective", "short")
    sys, usr = format_compose_prompt(
        domain="agent",
        dominant_drive="curiosity",
        anchor=None,
        archetype=arch,
        context={},
    )
    assert "(none)" in sys
    assert "(жодних конкретних подій" in usr


def test_format_compose_voice_hint():
    arch = Archetype("novelty", "direct", "medium")
    sys, _ = format_compose_prompt(
        domain="music",
        dominant_drive="curiosity",
        anchor=None,
        archetype=arch,
        context={},
        voice_hint="Пиши як старий друг.",
    )
    assert "старий друг" in sys


@pytest.mark.asyncio
async def test_default_compose_client_short_circuits_on_llm_error():
    """If the wrapped LLM client throws, compose returns empty string -- not raise."""
    from skynet_impulse.compose import DefaultOpenAIComposeClient

    class FailingClient:
        class chat:
            class completions:
                @staticmethod
                def create(**_kw):
                    raise RuntimeError("network down")

    cli = DefaultOpenAIComposeClient(FailingClient(), model="x")
    result = await cli.compose(
        domain="agent",
        dominant_drive="curiosity",
        anchor=None,
        archetype=Archetype("novelty", "curious", "short"),
        context={},
    )
    assert result == ""


@pytest.mark.asyncio
async def test_default_compose_client_success_path():
    from skynet_impulse.compose import DefaultOpenAIComposeClient

    class Msg:
        content = "  Як справи з Autechre?  "

    class Choice:
        message = Msg()

    class Resp:
        choices = [Choice()]

    class GoodClient:
        def __init__(self):
            self.last_call = None

        class _CompProxy:
            def __init__(self, owner):
                self._owner = owner

            def create(self, **kwargs):
                self._owner.last_call = kwargs
                return Resp()

        @property
        def chat(self):
            outer = self

            class _Chat:
                completions = outer._CompProxy(outer)

            return _Chat()

    good = GoodClient()
    cli = DefaultOpenAIComposeClient(good, model="mistral-large-2512")
    result = await cli.compose(
        domain="music",
        dominant_drive="curiosity",
        anchor="music:artist:autechre",
        archetype=Archetype("novelty", "curious", "short"),
        context={"drives": {"curiosity": 0.8}, "signals": []},
    )
    assert result == "Як справи з Autechre?"
    # short archetype -> token budget 120
    assert good.last_call["max_tokens"] == 120
    assert good.last_call["model"] == "mistral-large-2512"
