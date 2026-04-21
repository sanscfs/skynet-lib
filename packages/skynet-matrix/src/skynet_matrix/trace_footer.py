"""Trace footer formatting for Matrix messages.

Shared across skynet-agent, skynet-sre, and any service that posts
traced responses to Matrix. Renders an HTML <small> block with
clickable Grafana trace link, timing breakdown, token counts, etc.

Skips fields that are absent or zero — callers pass only what they have.
"""

from __future__ import annotations

import json
import urllib.parse

GRAFANA_BASE = "https://grafana.sanscfs.dev"


def _fmt_tokens(n: int) -> str:
    """Format token count: 1234 → '1.2k', 890 → '890'."""
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _grafana_trace_url(trace_id: str, grafana_base: str = GRAFANA_BASE) -> str:
    left = json.dumps(
        {
            "datasource": "tempo",
            "queries": [{"refId": "A", "queryType": "traceql", "query": trace_id}],
            "range": {"from": "now-1h", "to": "now"},
        },
        separators=(",", ":"),
    )
    return f"{grafana_base}/explore?orgId=1&left={urllib.parse.quote(left, safe='')}"


def format_trace_footer(
    *,
    trace_id: str = "",
    grafana_base: str = GRAFANA_BASE,
    duration_s: float = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    steps: list[dict] | None = None,
    rag_sources: list[str] | None = None,
    tools_used: list[str] | None = None,
    cost_usd: float = 0,
    service: str = "",
) -> str:
    """Format a trace footer for a Matrix message.

    Args:
        trace_id: OpenTelemetry trace ID (hex). If set, renders a Grafana link.
        grafana_base: Grafana base URL.
        duration_s: Total wall-clock duration in seconds.
        prompt_tokens: Input tokens consumed.
        completion_tokens: Output tokens generated.
        steps: Per-phase timing breakdown, e.g.
               [{"name": "llm", "duration_s": 2.3}, {"name": "tools", "duration_s": 1.1}]
        rag_sources: List of RAG source paths (shows filenames, max 3).
        tools_used: List of tool names used (max 3).
        cost_usd: Estimated cost in USD.
        service: Service/DAG identifier (e.g. ``"dag:active_memory"`` or
            ``"skynet-agent"``). Renders as a trailing tag; useful for
            Airflow DAG runs that have no OTel trace id but still want
            source attribution in the footer.

    Returns:
        HTML string ``\\n\\n<small>... | ...</small>`` ready to append to a
        Matrix message body.  Empty string if nothing to show.
    """
    parts: list[str] = []

    if trace_id:
        url = _grafana_trace_url(trace_id, grafana_base)
        parts.append(f'<a href="{url}">\U0001f50d trace</a>')

    if duration_s > 0:
        parts.append(f"\u23f1{duration_s:.1f}s")

    if prompt_tokens or completion_tokens:
        parts.append(f"\U0001f4ca{_fmt_tokens(prompt_tokens)}\u2192{_fmt_tokens(completion_tokens)}tok")

    if steps:
        step_parts = [f"{s['name']} {s['duration_s']:.1f}s" for s in steps if s.get("duration_s", 0) > 0.05]
        if step_parts:
            parts.append("\U0001f504 " + " \u00b7 ".join(step_parts))

    if rag_sources:
        src = ", ".join(s.split("/")[-1] for s in rag_sources[:3])
        parts.append(f"\U0001f4ce{src}")

    if tools_used:
        tools = ", ".join(t.split("__")[-1] if "__" in t else t for t in tools_used[:3])
        parts.append(f"\U0001f527{tools}")

    if cost_usd > 0:
        parts.append(f"\U0001f4b0${cost_usd:.3f}")

    if service:
        parts.append(f"\U0001f6e0\ufe0f{service}")

    if not parts:
        return ""

    return "\n\n<small>" + " | ".join(parts) + "</small>"


def current_trace_id() -> str:
    """Return the current OpenTelemetry span trace ID (32-char hex) or empty string."""
    try:
        from opentelemetry import trace as _otel_trace  # noqa: PLC0415

        span = _otel_trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
    except Exception:
        pass
    return ""


def with_trace_footer(
    text: str,
    duration_s: float = 0.0,
    *,
    tools_used: list[str] | None = None,
    service: str = "",
) -> dict:
    """Wrap plain text into ``{text, html}`` with OTel trace footer appended.

    Reads the current OTel span automatically via :func:`current_trace_id`.
    Drop-in replacement for the per-service ``tracing.with_trace_footer`` shims.
    """
    import html as _html  # noqa: PLC0415

    footer = format_trace_footer(
        trace_id=current_trace_id(),
        duration_s=duration_s,
        tools_used=tools_used or [],
        service=service,
    )
    escaped = _html.escape(text).replace("\n", "<br/>")
    return {"text": text, "html": escaped + footer}


def build_trace_meta(
    trace_id: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    duration_ms: int = 0,
    service: str = "skynet",
    grafana_base: str = GRAFANA_BASE,
) -> dict:
    """Build a trace metadata dict for ``dev.skynet.trace`` Matrix event content."""
    return {
        "trace_id": trace_id,
        "grafana_url": _grafana_trace_url(trace_id, grafana_base) if trace_id else "",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "duration_ms": duration_ms,
        "service": service,
    }
