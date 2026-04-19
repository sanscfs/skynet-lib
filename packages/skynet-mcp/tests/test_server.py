"""Tests for mount_mcp / HTTP endpoints via FastAPI TestClient."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from skynet_mcp import ToolRegistry, mount_mcp


@pytest.fixture
def app_and_registry() -> tuple[FastAPI, ToolRegistry]:
    registry = ToolRegistry()

    @registry.tool(
        name="mark_watched",
        description="Mark a movie as watched",
        schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "year": {"type": "integer"},
            },
            "required": ["title"],
        },
    )
    async def mark_watched(title: str, year: int | None = None) -> dict:
        return {"movie_id": 42, "title": title, "year": year}

    @registry.tool(
        name="ping_sync",
        description="sync handler for test coverage",
        schema={"type": "object", "properties": {"msg": {"type": "string"}}},
    )
    def ping_sync(msg: str = "hi") -> dict:
        return {"echo": msg}

    @registry.tool(
        name="boom",
        description="raises",
        schema={"type": "object"},
    )
    async def boom() -> None:
        raise RuntimeError("do not leak this stack trace")

    app = FastAPI()
    mount_mcp(app, registry)
    return app, registry


@pytest.fixture
def client(app_and_registry) -> TestClient:
    app, _ = app_and_registry
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /tools
# ---------------------------------------------------------------------------


def test_get_tools_returns_all_registered_descriptors(client: TestClient) -> None:
    resp = client.get("/tools")
    assert resp.status_code == 200
    body = resp.json()
    assert "tools" in body
    names = {t["name"] for t in body["tools"]}
    assert names == {"mark_watched", "ping_sync", "boom"}
    mark = next(t for t in body["tools"] if t["name"] == "mark_watched")
    assert mark["description"] == "Mark a movie as watched"
    assert mark["inputSchema"]["required"] == ["title"]


# ---------------------------------------------------------------------------
# POST /tools/{name}/call
# ---------------------------------------------------------------------------


def test_call_valid_async_tool_returns_result(client: TestClient) -> None:
    resp = client.post(
        "/tools/mark_watched/call",
        json={"arguments": {"title": "Inception", "year": 2010}},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "result": {"movie_id": 42, "title": "Inception", "year": 2010},
    }


def test_call_sync_tool_also_works(client: TestClient) -> None:
    resp = client.post(
        "/tools/ping_sync/call",
        json={"arguments": {"msg": "hola"}},
    )
    assert resp.status_code == 200
    assert resp.json() == {"result": {"echo": "hola"}}


def test_call_invalid_schema_returns_400(client: TestClient) -> None:
    # `title` is required; omit it
    resp = client.post("/tools/mark_watched/call", json={"arguments": {"year": 2010}})
    assert resp.status_code == 400
    body = resp.json()
    assert "error" in body
    assert "title" in body["error"].lower() or "required" in body["error"].lower()


def test_call_wrong_type_returns_400(client: TestClient) -> None:
    resp = client.post(
        "/tools/mark_watched/call",
        json={"arguments": {"title": "X", "year": "not-an-int"}},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_call_unknown_tool_returns_404(client: TestClient) -> None:
    resp = client.post("/tools/nonexistent/call", json={"arguments": {}})
    assert resp.status_code == 404


def test_call_handler_exception_returns_500_without_stack_trace(
    client: TestClient,
) -> None:
    resp = client.post("/tools/boom/call", json={"arguments": {}})
    assert resp.status_code == 500
    body = resp.json()
    assert "error" in body
    # Error body must NOT contain a stack trace (no "Traceback", no filename/line refs).
    assert "Traceback" not in body["error"]
    assert ".py" not in body["error"]
    # But it should convey the exception type + message so the caller can react.
    assert "RuntimeError" in body["error"]


def test_call_missing_arguments_key_treats_as_empty(client: TestClient) -> None:
    # Body without `arguments` is equivalent to `arguments: {}`.
    resp = client.post("/tools/ping_sync/call", json={})
    assert resp.status_code == 200
    assert resp.json() == {"result": {"echo": "hi"}}


def test_call_non_object_arguments_returns_400(client: TestClient) -> None:
    resp = client.post("/tools/ping_sync/call", json={"arguments": "not-a-dict"})
    assert resp.status_code == 400
    assert "error" in resp.json()


# ---------------------------------------------------------------------------
# Upstream HTTP status passthrough
# ---------------------------------------------------------------------------


def _make_http_status_error(status_code: int) -> httpx.HTTPStatusError:
    """Build an HTTPStatusError with a real response we can attach to it."""
    request = httpx.Request("GET", "http://upstream.test/api")
    response = httpx.Response(status_code=status_code, request=request)
    return httpx.HTTPStatusError(
        message=f"Server error '{status_code}'",
        request=request,
        response=response,
    )


def test_call_preserves_upstream_502_from_httpx_error() -> None:
    registry = ToolRegistry()

    @registry.tool(
        name="upstream_call",
        description="calls a flaky downstream",
        schema={"type": "object"},
    )
    async def upstream_call() -> None:
        raise _make_http_status_error(502)

    app = FastAPI()
    mount_mcp(app, registry)
    client = TestClient(app)

    resp = client.post("/tools/upstream_call/call", json={"arguments": {}})
    assert resp.status_code == 502
    body = resp.json()
    assert "error" in body
    assert "502" in body["error"]


def test_call_preserves_upstream_404_from_httpx_error() -> None:
    registry = ToolRegistry()

    @registry.tool(name="upstream_404", description="", schema={"type": "object"})
    async def upstream_404() -> None:
        raise _make_http_status_error(404)

    app = FastAPI()
    mount_mcp(app, registry)
    client = TestClient(app)

    resp = client.post("/tools/upstream_404/call", json={"arguments": {}})
    assert resp.status_code == 404
    # 404 here means "upstream returned 404", NOT "unknown tool" -- the
    # error body distinguishes the two.
    assert "Upstream" in resp.json()["error"]


def test_call_passthrough_any_exception_with_response_status_code() -> None:
    """Duck-typed passthrough: non-httpx exceptions that carry a
    ``.response.status_code`` (e.g. requests.HTTPError) also round-trip."""
    registry = ToolRegistry()

    class CustomHTTPError(Exception):
        def __init__(self, status: int) -> None:
            super().__init__(f"HTTP {status}")
            self.response = SimpleNamespace(status_code=status)

    @registry.tool(name="custom_http_err", description="", schema={"type": "object"})
    async def custom_http_err() -> None:
        raise CustomHTTPError(503)

    app = FastAPI()
    mount_mcp(app, registry)
    client = TestClient(app)

    resp = client.post("/tools/custom_http_err/call", json={"arguments": {}})
    assert resp.status_code == 503


def test_call_non_http_exception_still_becomes_500() -> None:
    """Exceptions without a ``.response.status_code`` must continue to
    surface as 500 -- the passthrough path is additive, not a
    replacement for the generic fallback."""
    registry = ToolRegistry()

    @registry.tool(name="plain_err", description="", schema={"type": "object"})
    async def plain_err() -> None:
        raise ValueError("something broke")

    app = FastAPI()
    mount_mcp(app, registry)
    client = TestClient(app)

    resp = client.post("/tools/plain_err/call", json={"arguments": {}})
    assert resp.status_code == 500
    body = resp.json()
    assert "ValueError" in body["error"]


# ---------------------------------------------------------------------------
# Configurable OTel span prefix
# ---------------------------------------------------------------------------


def test_mount_mcp_default_span_prefix_is_tool_underscore(monkeypatch) -> None:
    """Default span_prefix is ``tool_`` so pre-library dashboards keep
    matching the span name ``tool_{name}``."""
    recorded: list[str] = []

    class _FakeSpanCtx:
        def __enter__(self) -> "_FakeSpanCtx":
            return self

        def __exit__(self, *args) -> None:
            return None

    class _FakeTracer:
        def start_as_current_span(self, name: str) -> _FakeSpanCtx:
            recorded.append(name)
            return _FakeSpanCtx()

    import skynet_mcp.server as server_mod

    monkeypatch.setattr(server_mod, "_TRACER", _FakeTracer())

    registry = ToolRegistry()

    @registry.tool(name="ping", description="", schema={"type": "object"})
    def ping() -> dict:
        return {"ok": True}

    app = FastAPI()
    mount_mcp(app, registry)  # default span_prefix
    client = TestClient(app)
    resp = client.post("/tools/ping/call", json={"arguments": {}})
    assert resp.status_code == 200
    assert recorded == ["tool_ping"]


def test_mount_mcp_custom_span_prefix(monkeypatch) -> None:
    """Services that want the namespaced form pass
    ``span_prefix="mcp.tool."``; the emitted span becomes
    ``mcp.tool.{name}``."""
    recorded: list[str] = []

    class _FakeSpanCtx:
        def __enter__(self) -> "_FakeSpanCtx":
            return self

        def __exit__(self, *args) -> None:
            return None

    class _FakeTracer:
        def start_as_current_span(self, name: str) -> _FakeSpanCtx:
            recorded.append(name)
            return _FakeSpanCtx()

    import skynet_mcp.server as server_mod

    monkeypatch.setattr(server_mod, "_TRACER", _FakeTracer())

    registry = ToolRegistry()

    @registry.tool(name="ping", description="", schema={"type": "object"})
    def ping() -> dict:
        return {"ok": True}

    app = FastAPI()
    mount_mcp(app, registry, span_prefix="mcp.tool.")
    client = TestClient(app)
    resp = client.post("/tools/ping/call", json={"arguments": {}})
    assert resp.status_code == 200
    assert recorded == ["mcp.tool.ping"]
