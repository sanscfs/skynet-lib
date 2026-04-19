# skynet-mcp

Reusable MCP-over-HTTP server scaffolding for Skynet services. Register tools via
a decorator; `mount_mcp(app, registry)` adds MCP-compatible `GET /tools` and
`POST /tools/{name}/call` routes to your FastAPI app.

## Usage

```python
from fastapi import FastAPI
from skynet_mcp import ToolRegistry, mount_mcp

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
    return {"movie_id": 42, "title": title}


app = FastAPI()
mount_mcp(app, registry)
```

## Endpoints

- `GET /tools` -> `{"tools": [{name, description, inputSchema}, ...]}`
- `POST /tools/{name}/call` body `{"arguments": {...}}`
  - 200 `{"result": ...}` on success
  - 400 on schema validation failure
  - 404 for unknown tools
  - `{status}` `{"error": "Upstream: {status}"}` when the handler raises
    `httpx.HTTPStatusError` (or any exception carrying a
    `.response.status_code` attribute, e.g. `requests.HTTPError`): the
    upstream status is preserved on the wire so callers that special-case
    502/503/404 from a downstream service keep working. httpx is a soft
    dependency -- if httpx isn't installed the passthrough still works
    via duck-typing.
  - 500 `{"error": "ClassName: msg"}` on any other handler exception
    (no stack traces on the wire)

## Optional OTel

Install the `otel` extra. When `opentelemetry` is importable, every tool call
is wrapped in a span whose name is `f"{span_prefix}{tool_name}"`. The default
`span_prefix` is `"tool_"`, which matches the hand-rolled MCP scaffolding
that Skynet services used before this library existed -- existing
Grafana/Tempo dashboards keep matching without any change.

New services that prefer the namespaced form can opt in via:

```python
mount_mcp(app, registry, span_prefix="mcp.tool.")
```

which emits spans like `mcp.tool.mark_watched`.

```
pip install "skynet-mcp[otel]"
```
