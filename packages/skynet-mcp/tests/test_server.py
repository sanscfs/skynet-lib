"""Tests for mount_mcp / HTTP endpoints via FastAPI TestClient."""

from __future__ import annotations

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
