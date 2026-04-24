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
