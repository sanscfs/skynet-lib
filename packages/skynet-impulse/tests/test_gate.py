"""GateClient protocol + prompt formatter."""

from __future__ import annotations

import pytest
from skynet_impulse.gate import format_gate_prompt
from skynet_impulse.signals import Signal


def test_format_gate_includes_domain_and_drives():
    sig = Signal(kind="novelty", source="analyzer", salience=0.9, anchor="a")
    sys, usr = format_gate_prompt(
        domain="music",
        dominant_drive="curiosity",
        dominant_value=0.6,
        baseline_p75=0.4,
        anchor="music:artist:x",
        recent_context="recent user said hi",
        drives={"curiosity": 0.6, "boredom": 0.1},
        signals=[sig],
    )
    assert "music" in sys
    assert "curiosity=0.60" in usr
    assert "music:artist:x" in usr
    assert "novelty" in usr


def test_format_gate_empty_signals():
    sys, usr = format_gate_prompt(
        domain="agent",
        dominant_drive="boredom",
        dominant_value=0.5,
        baseline_p75=0.35,
        anchor=None,
        recent_context="",
        drives={},
        signals=[],
    )
    assert "agent" in sys
    assert "(немає свіжих подій)" in usr
    assert "(тиша)" in usr


def test_format_gate_voice_hint_inserted():
    sys, _ = format_gate_prompt(
        domain="movies",
        dominant_drive="curiosity",
        dominant_value=0.5,
        baseline_p75=0.3,
        anchor=None,
        recent_context="",
        drives={},
        signals=[],
        voice_hint="Голос кінокритика.",
    )
    assert "Голос кінокритика." in sys


@pytest.mark.asyncio
async def test_gate_protocol_can_be_mocked():
    """Ensure any object with the right shape passes as a GateClient."""

    class MockGate:
        async def should_fire(self, **kw):
            return True, "because tests"

    from skynet_impulse.gate import GateClient

    gate: GateClient = MockGate()  # Protocol satisfaction checked by structural typing
    fire, reason = await gate.should_fire(
        domain="x",
        dominant_drive="a",
        dominant_value=0.5,
        baseline_p75=0.3,
        anchor=None,
        recent_context="",
        drives={},
        signals=[],
    )
    assert fire is True
    assert "tests" in reason
