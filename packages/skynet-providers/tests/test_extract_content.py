"""Tests for the lenient ``_extract_content`` shape handling.

Cloud LLM providers diverge on where the assistant text lives — canonical
``message.content`` is just one of several variants. These tests pin the
priority chain so a regression flips the right tripwire.

Live observations driving the variants:

- glm-4.7 via OpenRouter+DeepInfra (2026-04-27): ``content: null``,
  payload in ``message.reasoning`` AND ``message.reasoning_details[0].text``.
- Some providers route JSON-mode output through a synthetic
  ``tool_calls[0].function.arguments`` even without a tool definition.
- Legacy ``/completions``-style proxies emit ``choices[0].text``.
"""

from __future__ import annotations

import pytest

from skynet_providers.chat import _extract_content
from skynet_providers.exceptions import ProviderError


def test_canonical_content_string():
    body = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hello world"},
                "finish_reason": "stop",
            }
        ]
    }
    assert _extract_content(body) == "hello world"


def test_content_takes_priority_over_reasoning():
    """When both content AND reasoning are present, content wins."""
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "real answer",
                    "reasoning": "scratchpad — should not be returned",
                },
                "finish_reason": "stop",
            }
        ]
    }
    assert _extract_content(body) == "real answer"


def test_tool_calls_function_arguments_fallback():
    """JSON-mode output sometimes routed through a synthetic tool call."""
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "respond",
                                "arguments": '{"reply": "hi", "tool_name": null}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    assert _extract_content(body) == '{"reply": "hi", "tool_name": null}'


def test_reasoning_field_fallback_glm47_shape():
    """Real shape observed from glm-4.7 via OpenRouter+DeepInfra 2026-04-27."""
    body = {
        "id": "gen-xyz",
        "model": "z-ai/glm-4.7-20251222",
        "provider": "DeepInfra",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "refusal": None,
                    "reasoning": '{"reply": "Привіт!", "tool_name": null}',
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "text": '{"reply": "Привіт!", "tool_name": null}',
                            "format": "unknown",
                            "index": 0,
                        }
                    ],
                },
            }
        ],
    }
    assert _extract_content(body) == '{"reply": "Привіт!", "tool_name": null}'


def test_reasoning_details_only_no_reasoning_field():
    """Some providers may populate only reasoning_details[].text."""
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "the answer"},
                    ],
                },
                "finish_reason": "stop",
            }
        ]
    }
    assert _extract_content(body) == "the answer"


def test_legacy_choices_text_fallback():
    """Legacy /completions shape some OpenAI-compatible proxies still emit."""
    body = {
        "choices": [
            {
                "text": "legacy answer",
                "finish_reason": "stop",
            }
        ]
    }
    assert _extract_content(body) == "legacy answer"


def test_empty_string_content_falls_through_to_fallback():
    """``content=""`` should NOT be returned — fall through to next variant."""
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "fallback wins",
                },
                "finish_reason": "stop",
            }
        ]
    }
    assert _extract_content(body) == "fallback wins"


def test_no_choices_raises():
    with pytest.raises(ProviderError, match="no 'choices'"):
        _extract_content({})


def test_all_variants_empty_raises_clear_error():
    """When nothing yields content, raise with a tripwire message."""
    body = {
        "choices": [
            {
                "message": {"role": "assistant", "content": None},
                "finish_reason": "stop",
            }
        ]
    }
    with pytest.raises(ProviderError, match="no fallback"):
        _extract_content(body)


def test_tool_calls_priority_over_reasoning():
    """tool_calls beats reasoning when both present (Priority 2 > Priority 3)."""
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"function": {"arguments": "from-tool"}}
                    ],
                    "reasoning": "from-reasoning",
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    assert _extract_content(body) == "from-tool"


def test_malformed_tool_calls_skipped():
    """Garbage tool_calls shouldn't crash — fall through to next variant."""
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"not_function": "garbage"}],
                    "reasoning": "the real answer",
                },
                "finish_reason": "stop",
            }
        ]
    }
    assert _extract_content(body) == "the real answer"
