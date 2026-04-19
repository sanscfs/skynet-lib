"""Tool registry with decorator-based registration.

Replaces the dict-of-handlers / TOOLS-list pattern used by skynet-movies.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterator

from skynet_mcp.exceptions import DuplicateToolError
from skynet_mcp.schemas import Tool

Handler = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass
class ToolSpec:
    """Internal representation of a registered tool."""

    name: str
    description: str
    schema: dict[str, Any] = field(default_factory=dict)
    handler: Handler | None = None
    is_async: bool = False

    def to_descriptor(self) -> Tool:
        """Convert to the wire-shape Tool descriptor (name/description/inputSchema)."""
        return Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.schema,
        )


class ToolRegistry:
    """Registry of MCP tools.

    Usage:

        registry = ToolRegistry()

        @registry.tool(name="foo", description="...", schema={...})
        async def foo(**kwargs):
            ...
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def tool(
        self,
        name: str,
        description: str,
        schema: dict[str, Any] | None = None,
    ) -> Callable[[Handler], Handler]:
        """Decorator registering a function as a tool.

        Supports both sync and async callables -- coroutine status is
        detected via ``inspect.iscoroutinefunction`` and awaited
        appropriately at call time.
        """

        def decorator(func: Handler) -> Handler:
            self.register(
                name=name,
                description=description,
                schema=schema or {},
                handler=func,
            )
            return func

        return decorator

    def register(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        handler: Handler,
    ) -> ToolSpec:
        """Register a tool imperatively (non-decorator form)."""
        if name in self._tools:
            raise DuplicateToolError(f"Tool {name!r} is already registered")

        spec = ToolSpec(
            name=name,
            description=description,
            schema=schema,
            handler=handler,
            is_async=inspect.iscoroutinefunction(handler),
        )
        self._tools[name] = spec
        return spec

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def descriptors(self) -> list[Tool]:
        """Return all tools in the MCP wire shape."""
        return [spec.to_descriptor() for spec in self._tools.values()]


__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "Handler",
]
