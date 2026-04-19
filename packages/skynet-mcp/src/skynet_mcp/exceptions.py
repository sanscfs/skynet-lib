"""Exceptions for skynet-mcp."""

from __future__ import annotations


class MCPError(Exception):
    """Base class for skynet-mcp errors."""


class DuplicateToolError(MCPError):
    """Raised when registering a tool with a name already present in the registry."""


class ToolNotFoundError(MCPError):
    """Raised when calling a tool not registered."""


class ToolValidationError(MCPError):
    """Raised when tool arguments fail JSON schema validation."""


__all__ = [
    "MCPError",
    "DuplicateToolError",
    "ToolNotFoundError",
    "ToolValidationError",
]
