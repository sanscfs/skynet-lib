"""Mount MCP-compatible FastAPI routes from a :class:`ToolRegistry`.

Exposes two endpoints:

* ``GET  {prefix}/tools`` -- returns the list of registered tools in the MCP
  descriptor shape ``{"tools": [{name, description, inputSchema}, ...]}``.
* ``POST {prefix}/tools/{name}/call`` -- validates arguments against the
  tool's JSON schema, invokes the handler (sync or async), and wraps the
  outcome as ``{"result": ...}`` or ``{"error": "..."}``. Never leaks
  stack traces to the wire.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from typing import Any, Iterator

import jsonschema
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from skynet_mcp.registry import ToolRegistry, ToolSpec
from skynet_mcp.schemas import ToolsListResponse

logger = logging.getLogger("skynet_mcp")


# ---------------------------------------------------------------------------
# Optional OTel integration
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import guard
    from opentelemetry import trace as _otel_trace

    _TRACER = _otel_trace.get_tracer("skynet_mcp")
except ImportError:  # pragma: no cover
    _TRACER = None  # type: ignore[assignment]


@contextlib.contextmanager
def _tool_span(name: str) -> Iterator[None]:
    """Open an OTel span named ``mcp.tool.{name}`` if OTel is available."""
    if _TRACER is None:
        yield
        return
    with _TRACER.start_as_current_span(f"mcp.tool.{name}"):
        yield


# ---------------------------------------------------------------------------
# Route factory
# ---------------------------------------------------------------------------


def _safe_arg_keys(arguments: dict[str, Any]) -> list[str]:
    """Keys only -- values may contain user input that we must not log."""
    return sorted(arguments.keys()) if isinstance(arguments, dict) else []


async def _invoke(spec: ToolSpec, arguments: dict[str, Any]) -> Any:
    """Invoke the handler with ``**arguments``; await if async."""
    assert spec.handler is not None
    if spec.is_async:
        return await spec.handler(**arguments)
    result = spec.handler(**arguments)
    # Handler may still return an awaitable even if not decorated `async def`
    # (e.g., coroutine returned by a sync wrapper) -- respect that.
    if inspect.isawaitable(result):
        return await result
    return result


def build_router(registry: ToolRegistry) -> APIRouter:
    """Return a FastAPI router exposing the MCP endpoints for ``registry``.

    Prefer :func:`mount_mcp` which wires this router into an existing
    ``FastAPI`` application and is what services usually want.
    """
    router = APIRouter(tags=["mcp"])

    @router.get("/tools", response_model=ToolsListResponse)
    async def list_tools() -> dict[str, Any]:
        return {"tools": [spec.to_descriptor().model_dump() for spec in registry]}

    @router.post("/tools/{tool_name}/call")
    async def call_tool(tool_name: str, request: Request) -> JSONResponse:
        spec = registry.get(tool_name)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")

        # Parse body defensively -- empty body is allowed (no args).
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        arguments = body.get("arguments", {})
        if not isinstance(arguments, dict):
            return JSONResponse(
                status_code=400,
                content={"error": "'arguments' must be a JSON object"},
            )

        # Validate arguments against the registered schema.
        if spec.schema:
            try:
                jsonschema.validate(instance=arguments, schema=spec.schema)
            except jsonschema.ValidationError as exc:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid arguments: {exc.message}"},
                )
            except jsonschema.SchemaError as exc:  # pragma: no cover - misconfig
                logger.error("tool %s has invalid schema: %s", tool_name, exc)
                return JSONResponse(
                    status_code=500,
                    content={"error": "Tool schema is invalid"},
                )

        logger.info(
            "mcp.tool.call name=%s arg_keys=%s",
            tool_name,
            _safe_arg_keys(arguments),
        )

        try:
            with _tool_span(tool_name):
                result = await _invoke(spec, arguments)
        except Exception as exc:
            # Log with traceback server-side; wire body carries only the
            # exception class name + message -- never the stack trace.
            logger.exception("mcp.tool.error name=%s", tool_name)
            return JSONResponse(
                status_code=500,
                content={"error": f"{type(exc).__name__}: {exc}"},
            )

        return JSONResponse(status_code=200, content={"result": result})

    return router


def mount_mcp(
    app: FastAPI,
    registry: ToolRegistry,
    prefix: str = "",
) -> None:
    """Attach the MCP endpoints to ``app``.

    ``prefix`` is passed straight through to ``app.include_router`` -- use
    ``""`` to mount at ``/tools`` (the default MCP convention) or
    ``"/mcp"`` to namespace under ``/mcp/tools``.
    """
    # Normalize a trailing slash -- FastAPI forbids it on ``include_router``.
    if prefix == "/":
        prefix = ""
    app.include_router(build_router(registry), prefix=prefix)


__all__ = ["mount_mcp", "build_router"]
