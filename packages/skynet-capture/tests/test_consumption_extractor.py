"""Unit tests for the lib's extractor — LLM client is a fake.

These tests sat in skynet-profiler as part of test_movies_notes /
test_music_notes before Phase 1. They ride along with the lib now
so any future refactor of the prompt / JSON validation stays covered
independently of the profiler event loop.
"""

from __future__ import annotations

from typing import Any

import pytest
from skynet_capture.common.consumption_extractor import (
    MIN_LLM_TEXT_LEN,
    extract_consumption,
)


class _FakeLLM:
    """Minimal LLMLike impl — returns whatever the test configures."""

    def __init__(self, response: dict[str, Any] | None) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def json_completion(
        self, system: str, user: str, *, max_tokens: int = 800, temperature: float = 0.0
    ) -> dict[str, Any] | None:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return self.response


@pytest.mark.asyncio
async def test_movies_basic_shape():
    llm = _FakeLLM({"movies": [{"title": "Dune", "year": 2021, "notes": "great", "watched": True}]})
    out = await extract_consumption("watched Dune yesterday", kind="movies", llm=llm)
    assert out == [{"title": "Dune", "year": 2021, "notes": "great", "watched": True}]
    # system prompt + user text propagated
    assert "movie watch events" in llm.calls[0]["system"]
    assert llm.calls[0]["user"] == "watched Dune yesterday"


@pytest.mark.asyncio
async def test_music_basic_shape():
    llm = _FakeLLM(
        {"tracks": [{"artist": "Angine", "track": "Je", "year": None, "notes": "wild drummer", "listened": True}]}
    )
    out = await extract_consumption("слухав Angine", kind="music", llm=llm)
    assert out[0]["artist"] == "Angine"
    assert out[0]["track"] == "Je"
    assert out[0]["year"] is None
    assert out[0]["listened"] is True


@pytest.mark.asyncio
async def test_empty_text_returns_empty():
    llm = _FakeLLM({"movies": []})
    out = await extract_consumption("   ", kind="movies", llm=llm)
    assert out == []
    assert llm.calls == []  # short-circuited before LLM call


@pytest.mark.asyncio
async def test_llm_none_returns_empty():
    llm = _FakeLLM(None)
    assert await extract_consumption("watched Dune", kind="movies", llm=llm) == []


@pytest.mark.asyncio
async def test_hallucinated_year_clamped_to_none():
    llm = _FakeLLM({"movies": [{"title": "X", "year": "2026 or 2027", "notes": "", "watched": True}]})
    out = await extract_consumption("x", kind="movies", llm=llm)
    assert out[0]["year"] is None


@pytest.mark.asyncio
async def test_missing_title_dropped():
    llm = _FakeLLM({"movies": [{"title": "", "year": 2021, "notes": "x", "watched": True}, {"title": "Dune"}]})
    out = await extract_consumption("x", kind="movies", llm=llm)
    assert len(out) == 1
    assert out[0]["title"] == "Dune"


@pytest.mark.asyncio
async def test_in_progress_flag_preserved():
    llm = _FakeLLM({"tracks": [{"artist": "", "track": "X", "listened": False}]})
    out = await extract_consumption("слухаю зараз X", kind="music", llm=llm)
    assert out[0]["listened"] is False


def test_min_text_len_is_public():
    # modules import this to decide whether to even call the LLM
    assert MIN_LLM_TEXT_LEN >= 1
