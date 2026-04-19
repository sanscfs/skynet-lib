"""Tests for the ToolRegistry + @tool decorator."""

from __future__ import annotations

import pytest
from skynet_mcp import DuplicateToolError, ToolRegistry


def test_register_tool_via_decorator_records_metadata_and_handler():
    registry = ToolRegistry()

    @registry.tool(
        name="mark_watched",
        description="Mark a movie as watched",
        schema={
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    )
    async def handler(title: str) -> dict:
        return {"title": title}

    spec = registry.get("mark_watched")
    assert spec is not None
    assert spec.name == "mark_watched"
    assert spec.description == "Mark a movie as watched"
    assert spec.schema["required"] == ["title"]
    assert spec.handler is handler
    assert spec.is_async is True
    assert "mark_watched" in registry
    assert len(registry) == 1


def test_register_duplicate_tool_raises():
    registry = ToolRegistry()

    @registry.tool(name="foo", description="d", schema={})
    def first() -> None:
        pass

    with pytest.raises(DuplicateToolError):

        @registry.tool(name="foo", description="d2", schema={})
        def second() -> None:
            pass


def test_sync_handler_marked_is_async_false():
    registry = ToolRegistry()

    @registry.tool(name="sync_tool", description="d", schema={})
    def handler() -> str:
        return "ok"

    assert registry.get("sync_tool").is_async is False


def test_descriptors_returns_mcp_wire_shape():
    registry = ToolRegistry()

    @registry.tool(name="a", description="A", schema={"type": "object"})
    def _a() -> None:
        pass

    @registry.tool(name="b", description="B", schema={"type": "object"})
    def _b() -> None:
        pass

    descriptors = registry.descriptors()
    assert [t.name for t in descriptors] == ["a", "b"]
    for t in descriptors:
        assert set(t.model_dump().keys()) == {"name", "description", "inputSchema"}
