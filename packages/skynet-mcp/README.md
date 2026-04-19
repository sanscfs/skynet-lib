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
  - 500 `{"error": "..."}` on handler exception (no stack traces on the wire)

## Optional OTel

Install the `otel` extra. When `opentelemetry` is importable, every tool call
is wrapped in a span named `mcp.tool.{name}`.

```
pip install "skynet-mcp[otel]"
```
