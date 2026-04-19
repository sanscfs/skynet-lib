"""Pydantic models for the MCP-over-HTTP wire contract."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Tool(BaseModel):
    """A single tool as returned by GET /tools.

    Matches the MCP tool descriptor shape: name, description, inputSchema
    (JSON schema for arguments).
    """

    name: str
    description: str
    inputSchema: dict[str, Any] = Field(default_factory=dict)


class ToolsListResponse(BaseModel):
    """Response body of GET /tools."""

    tools: list[Tool]


class ToolCallRequest(BaseModel):
    """Request body of POST /tools/{name}/call."""

    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallResponse(BaseModel):
    """Response body of POST /tools/{name}/call on success."""

    result: Any = None


class ToolCallErrorResponse(BaseModel):
    """Response body of POST /tools/{name}/call on failure."""

    error: str


__all__ = [
    "Tool",
    "ToolsListResponse",
    "ToolCallRequest",
    "ToolCallResponse",
    "ToolCallErrorResponse",
]
