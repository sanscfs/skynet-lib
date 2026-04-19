"""Shared wrapper that composes a Matrix message with a trace footer.

Builds the body / formatted_body / extra_content trio so that both the sync
and async Matrix clients (and any legacy callers hand-rolling requests.put)
can produce the same on-the-wire event shape as the chat agent:

    body              = "<user body>\\n\\n<small>... | ...</small>"
    formatted_body    = "<rendered HTML body>\\n\\n<small>...</small>"
    dev.skynet.trace  = {trace_id, grafana_url, prompt_tokens, ...}

``dev.skynet.trace`` is a custom event field -- ignored by stock Matrix
clients, surfaced by our Matrix UI / grafana-mcp tools for drill-down.
"""

from __future__ import annotations

from typing import Any

from skynet_matrix.trace_footer import (
    GRAFANA_BASE,
    build_trace_meta,
    format_trace_footer,
)


def build_footer_payload(
    body: str,
    *,
    trace_id: str = "",
    grafana_base: str = GRAFANA_BASE,
    duration_s: float = 0,
    duration_ms: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    steps: list[dict] | None = None,
    rag_sources: list[str] | None = None,
    tools_used: list[str] | None = None,
    cost_usd: float = 0,
    service: str = "",
    formatted_body: str | None = None,
    extra_content: dict | None = None,
    trace_meta_extra: dict | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return ``(body_with_footer, extra_content)`` ready for ``send_text``.

    Accepts ``duration_s`` or ``duration_ms`` (whichever is non-zero -- ms
    wins if both are set, so DAGs that only track milliseconds can pass one
    field). All trace fields are optional: with nothing set, the returned
    body is unchanged and extra_content carries only the caller-supplied
    extra keys (if any).

    ``formatted_body`` should be the HTML rendering of the *original*
    ``body`` (without the footer). The wrapper appends the footer HTML to
    whichever it needs. If omitted the plain body is re-used as HTML
    (good enough for one-line notifications; chat agents should pre-render
    markdown themselves and pass it in).

    ``trace_meta_extra`` is merged into the ``dev.skynet.trace`` content
    key (e.g. ``{"dag_id": ..., "run_id": ..., "model": ...}``).
    """
    effective_duration_s = duration_s
    if duration_ms:
        effective_duration_s = duration_ms / 1000.0
    effective_duration_ms = duration_ms
    if not effective_duration_ms and duration_s:
        effective_duration_ms = int(duration_s * 1000)

    footer = format_trace_footer(
        trace_id=trace_id,
        grafana_base=grafana_base,
        duration_s=effective_duration_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        steps=steps,
        rag_sources=rag_sources,
        tools_used=tools_used,
        cost_usd=cost_usd,
        service=service,
    )

    combined_body = body + footer if footer else body

    extra: dict[str, Any] = dict(extra_content or {})

    if footer or formatted_body:
        extra.setdefault("format", "org.matrix.custom.html")
        html_base = formatted_body if formatted_body is not None else body
        extra.setdefault("formatted_body", html_base + footer)

    has_trace = any(
        (
            trace_id,
            prompt_tokens,
            completion_tokens,
            effective_duration_ms,
            service,
            trace_meta_extra,
        )
    )
    if has_trace:
        meta = build_trace_meta(
            trace_id=trace_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=effective_duration_ms,
            service=service or "skynet",
            grafana_base=grafana_base,
        )
        if trace_meta_extra:
            meta.update(trace_meta_extra)
        extra["dev.skynet.trace"] = meta

    return combined_body, extra


def build_edit_payload(
    new_body: str,
    *,
    trace_id: str = "",
    grafana_base: str = GRAFANA_BASE,
    duration_s: float = 0,
    duration_ms: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    steps: list[dict] | None = None,
    rag_sources: list[str] | None = None,
    tools_used: list[str] | None = None,
    cost_usd: float = 0,
    service: str = "",
    formatted_body: str | None = None,
    trace_meta_extra: dict | None = None,
) -> tuple[str, str | None, dict | None]:
    """Return ``(body_with_footer, formatted_body_or_None, trace_meta)``.

    Intended for the edit flow -- the caller still has to wire the MSC
    replace / thread relations itself (they differ between the sync and
    async clients), but footer composition + trace_meta live here.
    """
    effective_duration_s = duration_s
    if duration_ms:
        effective_duration_s = duration_ms / 1000.0
    effective_duration_ms = duration_ms
    if not effective_duration_ms and duration_s:
        effective_duration_ms = int(duration_s * 1000)

    footer = format_trace_footer(
        trace_id=trace_id,
        grafana_base=grafana_base,
        duration_s=effective_duration_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        steps=steps,
        rag_sources=rag_sources,
        tools_used=tools_used,
        cost_usd=cost_usd,
        service=service,
    )

    combined_body = new_body + footer if footer else new_body
    combined_formatted: str | None = None
    if footer or formatted_body:
        html_base = formatted_body if formatted_body is not None else new_body
        combined_formatted = html_base + footer

    trace_meta: dict | None = None
    has_trace = any(
        (
            trace_id,
            prompt_tokens,
            completion_tokens,
            effective_duration_ms,
            service,
            trace_meta_extra,
        )
    )
    if has_trace:
        trace_meta = build_trace_meta(
            trace_id=trace_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=effective_duration_ms,
            service=service or "skynet",
            grafana_base=grafana_base,
        )
        if trace_meta_extra:
            trace_meta.update(trace_meta_extra)

    return combined_body, combined_formatted, trace_meta
