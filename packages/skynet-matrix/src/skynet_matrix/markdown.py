"""Matrix-compatible markdown → HTML rendering.

Single source of truth for rendering LLM / bot replies into the HTML
subset Element accepts (per the Matrix client allow-list: headers,
tables, lists, code blocks, blockquote, links, em/strong, br).

Used by:
- ``CommandBot.send_text`` default render path (when caller passes a plain
  ``text`` without a pre-rendered ``html``).
- ``AsyncLiveStream.complete`` for the LLM-authored ``final_text``.

Hot-path renderers (``AsyncLiveStream._render_body`` for live edits,
trail entries) intentionally do NOT go through this module — those
render a controlled vocabulary (headers + tool-call rows) where full
markdown is unnecessary and a per-edit mistune call would be wasteful.

The renderer is constructed once and cached. ``mistune`` is a hard
dependency (declared in ``pyproject.toml``); the import-time fallback
is defensive only — for environments where someone strips deps.
"""

from __future__ import annotations

import html as html_lib
from functools import lru_cache
from typing import Any, Callable


@lru_cache(maxsize=1)
def _renderer() -> Callable[[str], str] | None:
    """Build (and cache) a mistune renderer with the GFM plugin set we use.

    ``escape=True`` makes raw HTML in the input be escaped — LLMs sometimes
    produce stray ``<...>`` that should render literally instead of being
    interpreted by Element. Plugin set kept minimal to what actually
    helps a chat surface: tables, strikethrough, autolinks, task lists.
    """
    try:
        import mistune  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover — defensive
        return None
    return mistune.create_markdown(
        escape=True,
        plugins=["table", "strikethrough", "url", "task_lists"],
    )


def to_matrix_html(text: str) -> str:
    """Render markdown to Matrix-compatible HTML.

    Returns "" for empty input. If ``mistune`` is missing for any reason
    (defensive — declared as a dependency), falls back to bare escape +
    ``<br/>`` so callers always get a non-None HTML body.
    """
    if not text:
        return ""
    md: Any = _renderer()
    if md is None:  # pragma: no cover — defensive
        return html_lib.escape(text).replace("\n", "<br/>")
    return md(text).strip()


__all__ = ["to_matrix_html"]
