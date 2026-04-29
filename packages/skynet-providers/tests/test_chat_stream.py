"""Tests for the SSE streaming path in :func:`async_chat_completion`.

We patch ``httpx.AsyncClient`` so no sockets open; the mock responds
with a canned SSE stream built to mirror Mistral/OpenRouter's format
(``data: {...}\\n\\ndata: [DONE]\\n``). The tests verify both the
streaming accumulation and that the optional live-stream hooks fire
when a stream is active in the current context.
"""

from __future__ import annotations

from contextvars import ContextVar

import pytest


class _FakeStreamResp:
    def __init__(self, status_code: int, lines: list[str], body: bytes = b""):
        self.status_code = status_code
        self._lines = lines
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._body


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, *, headers=None, json=None):
        return _FAKE_CTX.get()(method, url, headers=headers, json=json)


_FAKE_CTX: ContextVar = ContextVar("fake_stream_factory")


@pytest.fixture(autouse=True)
def patch_httpx(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)


def _sse_lines(tokens: list[str]) -> list[str]:
    import json

    out: list[str] = []
    for tok in tokens:
        out.append("data: " + json.dumps({"choices": [{"delta": {"content": tok}}]}))
        out.append("")
    out.append("data: [DONE]")
    out.append("")
    return out


@pytest.mark.asyncio
async def test_sse_stream_accumulates_tokens(monkeypatch):
    # resolver tries Vault; bypass with explicit key
    from skynet_providers.chat import async_chat_completion

    def factory(method, url, *, headers=None, json=None):
        return _FakeStreamResp(200, _sse_lines(["hello", " ", "world"]))

    token = _FAKE_CTX.set(factory)
    try:
        out = await async_chat_completion(
            prompt="hi",
            model="mistral-medium-latest",
            api_url="https://api.mistral.ai/v1",
            api_key="fake",
        )
    finally:
        _FAKE_CTX.reset(token)
    assert out == "hello world"


@pytest.mark.asyncio
async def test_sse_stream_fires_live_hooks_when_active(monkeypatch):
    """With a current_live_stream set, provider emits start/token/end."""
    from skynet_matrix.async_live_stream import current_live_stream
    from skynet_providers.chat import async_chat_completion

    hook_calls: list[tuple] = []

    class _StubStream:
        async def start_llm(self, *, url: str = "", model: str = ""):
            hook_calls.append(("start", url, model))

        async def append_token(self, chunk: str):
            hook_calls.append(("tok", chunk))

        async def finish_llm(self, *, tokens: int = 0):
            hook_calls.append(("end", tokens))

    def factory(method, url, *, headers=None, json=None):
        return _FakeStreamResp(200, _sse_lines(["foo", "bar"]))

    stream_tok = current_live_stream.set(_StubStream())
    fake_tok = _FAKE_CTX.set(factory)
    try:
        out = await async_chat_completion(
            prompt="hi",
            model="mistral-medium-latest",
            api_url="https://api.mistral.ai/v1",
            api_key="fake",
        )
    finally:
        _FAKE_CTX.reset(fake_tok)
        current_live_stream.reset(stream_tok)

    assert out == "foobar"
    kinds = [c[0] for c in hook_calls]
    assert kinds[0] == "start"
    assert "tok" in kinds
    assert kinds[-1] == "end"
    # 2 tokens received
    assert sum(1 for c in hook_calls if c[0] == "tok") == 2


def test_parse_sse_delta_reasoning_passthrough_default():
    """Default behaviour: reasoning streamed as token (DeepSeek-R1 / GLM-4.7)."""
    import json as _json

    from skynet_providers.chat import _parse_sse_delta

    line = "data: " + _json.dumps({"choices": [{"delta": {"content": "", "reasoning": "scratchpad"}}]})
    assert _parse_sse_delta(line) == "scratchpad"


def test_parse_sse_delta_reasoning_suppressed_under_prefer_content_only():
    """JSON-mode caller must never see reasoning leak through."""
    import json as _json

    from skynet_providers.chat import _parse_sse_delta

    line = "data: " + _json.dumps({"choices": [{"delta": {"content": "", "reasoning": "User says hi..."}}]})
    # Empty string = "valid frame, nothing to append" — accumulator skips it.
    assert _parse_sse_delta(line, prefer_content_only=True) == ""


def test_parse_sse_delta_content_still_wins_under_prefer_content_only():
    """The flag only suppresses reasoning — real content still flows."""
    import json as _json

    from skynet_providers.chat import _parse_sse_delta

    line = "data: " + _json.dumps({"choices": [{"delta": {"content": '{"tool"', "reasoning": "ignored"}}]})
    assert _parse_sse_delta(line, prefer_content_only=True) == '{"tool"'


def test_wants_json_only_detects_ollama_format():
    from skynet_providers.chat import _wants_json_only

    assert _wants_json_only({"format": "json"}) is True
    assert _wants_json_only({"format": "text"}) is False


def test_wants_json_only_detects_openai_response_format():
    from skynet_providers.chat import _wants_json_only

    assert _wants_json_only({"response_format": {"type": "json_object"}}) is True
    assert _wants_json_only({"response_format": {"type": "text"}}) is False
    assert _wants_json_only({}) is False


@pytest.mark.asyncio
async def test_sse_stream_drops_reasoning_under_json_mode_extra(monkeypatch):
    """End-to-end: passing extra={response_format=json_object} suppresses
    reasoning across the whole stream so JSON-only callers see clean
    content. Mirrors the gpt-oss:20b leak that motivated 2026.4.34.
    """
    import json as _json

    from skynet_providers.chat import async_chat_completion

    def factory(method, url, *, headers=None, json=None):
        sse = [
            # gpt-oss-style reasoning-only chunks before content:
            "data: " + _json.dumps({"choices": [{"delta": {"content": "", "reasoning": "user wants"}}]}),
            "",
            "data: " + _json.dumps({"choices": [{"delta": {"content": "", "reasoning": " a mix"}}]}),
            "",
            # Then the real JSON content arrives:
            "data: " + _json.dumps({"choices": [{"delta": {"content": '{"tool":"play_mix"}'}}]}),
            "",
            "data: [DONE]",
            "",
        ]
        return _FakeStreamResp(200, sse)

    token = _FAKE_CTX.set(factory)
    try:
        out = await async_chat_completion(
            prompt="включай мікс",
            model="gpt-oss:20b",
            api_url="http://100.64.0.4:11434/v1",
            api_key="fake",
            extra={"response_format": {"type": "json_object"}},
        )
    finally:
        _FAKE_CTX.reset(token)

    # Reasoning suppressed; only the JSON content survives.
    assert out == '{"tool":"play_mix"}'


@pytest.mark.asyncio
async def test_sse_stream_raises_on_http_error():
    from skynet_providers.chat import async_chat_completion
    from skynet_providers.exceptions import ProviderError

    def factory(method, url, *, headers=None, json=None):
        return _FakeStreamResp(500, [], body=b"upstream boom")

    token = _FAKE_CTX.set(factory)
    try:
        with pytest.raises(ProviderError, match="500"):
            await async_chat_completion(
                prompt="hi",
                model="m",
                api_url="https://api.mistral.ai/v1",
                api_key="fake",
            )
    finally:
        _FAKE_CTX.reset(token)
