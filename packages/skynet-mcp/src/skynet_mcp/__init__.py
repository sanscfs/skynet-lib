"""Skynet MCP -- FastAPI MCP-over-HTTP server scaffolding.

Replaces the hand-rolled ``TOOLS`` list + ``_tool_*`` handlers pattern with
a decorator-based :class:`ToolRegistry` and a :func:`mount_mcp` helper.
"""

from skynet_mcp.exceptions import (
    DuplicateToolError,
    MCPError,
    ToolNotFoundError,
    ToolValidationError,
)
from skynet_mcp.registry import Handler, ToolRegistry, ToolSpec
from skynet_mcp.schemas import (
    Tool,
    ToolCallErrorResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolsListResponse,
)
from skynet_mcp.server import build_router, mount_mcp


def tool(
    registry: ToolRegistry,
    name: str,
    description: str,
    schema: dict | None = None,
):
    """Convenience decorator equivalent to ``registry.tool(...)``.

    Kept as a module-level re-export so callers can write::

        from skynet_mcp import tool
        @tool(registry, name="foo", description="...", schema={...})
        async def foo(): ...
    """
    return registry.tool(name=name, description=description, schema=schema)


__all__ = [
    # registry
    "ToolRegistry",
    "ToolSpec",
    "Handler",
    "tool",
    # server
    "mount_mcp",
    "build_router",
    # schemas
    "Tool",
    "ToolsListResponse",
    "ToolCallRequest",
    "ToolCallResponse",
    "ToolCallErrorResponse",
    # errors
    "MCPError",
    "DuplicateToolError",
    "ToolNotFoundError",
    "ToolValidationError",
]
