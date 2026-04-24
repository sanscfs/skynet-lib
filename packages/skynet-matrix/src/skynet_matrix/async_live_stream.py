"""AsyncLiveStream — live-updating Matrix message for streaming agent output.

Async sibling of :class:`skynet_matrix.live_stream.LiveStream`. Differences:

* Sends / edits via ``nio.AsyncClient.room_send`` (the same client
  :class:`skynet_matrix.bot.CommandBot` already owns) instead of the sync
  :class:`skynet_matrix.client.MatrixClient`. Keeps one TCP pool, one
  access-token surface, one rate-limiter.
* First-class Matrix thread support (``m.thread`` relation mirrored on
  both the placeholder and every ``m.replace`` edit).
* Renders a header line with elapsed timer + wrap-around progress cells
  so the user sees motion even while the LLM is silently thinking.
* Propagates itself via ``current_live_stream`` (``ContextVar``) so
  handlers like :class:`ChatAgent` and tool dispatchers emit events
  without being passed a stream reference explicitly.
* Throttles edits via ``debounce_seconds`` (inherited from
  ``MATRIX_EDIT_DEBOUNCE_SECONDS``) but ``TOOL_CALL`` / ``TOOL_RESULT``
  / ``ERROR`` bypass the throttle — those are "anchor" events the user
  wants to see without the 2.5s quantum.

Typical use::

    async with AsyncLiveStream(nio_client, room_id, thread_root=root) as stream:
        tok = current_live_stream.set(stream)
        try:
            result = await handler(event, body)
        finally:
            current_live_stream.reset(tok)
        await stream.complete(final_text=result)
"""

from __future__ import annotations

import asyncio
import html as html_lib
import logging
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

from skynet_matrix.stream_events import (
    DEFAULT_STREAM_MAXLEN,
    DEFAULT_STREAM_TTL_SECONDS,
    MATRIX_EDIT_DEBOUNCE_SECONDS,
    EventType,
    StreamEvent,
    stream_key,
)

logger = logging.getLogger("skynet_matrix.async_live_stream")


current_live_stream: ContextVar[Optional["AsyncLiveStream"]] = ContextVar("current_live_stream", default=None)


_PROGRESS_FILL = "\u25b0"  # ▰
_PROGRESS_EMPTY = "\u25b1"  # ▱
_PROGRESS_CELLS = 6
_DONE_BAR = _PROGRESS_FILL * _PROGRESS_CELLS
_ERROR_BAR = "\u2715" * _PROGRESS_CELLS  # ✕

_STATUS_ICON = {"running": "\U0001f9e0", "done": "\u2705", "error": "\u274c"}


def _render_progress(phase: int, cells: int = _PROGRESS_CELLS) -> str:
    idx = phase % cells
    return "".join(_PROGRESS_FILL if i <= idx else _PROGRESS_EMPTY for i in range(cells))


def _render_elapsed(seconds: float) -> str:
    mins = int(seconds) // 60
    secs = seconds - (mins * 60)
    return f"{mins:02d}:{secs:04.1f}"


@dataclass
class _LiveEntry:
    """One rendered row in the stacked live view."""

    type: EventType
    tool: str = ""
    text: str = ""
    duration_s: float = 0.0
    done: bool = False
    ts: float = field(default_factory=time.time)


class AsyncLiveStream:
    """Manages one live-updating Matrix message for a single handler turn."""

    def __init__(
        self,
        nio_client: Any,
        room_id: str,
        *,
        thread_root: Optional[str] = None,
        bot_name: str = "Skynet",
        redis_client: Any = None,
        session_id: Optional[str] = None,
        stream_prefix: str = "skynet:agent:session",
        maxlen: int = DEFAULT_STREAM_MAXLEN,
        ttl_seconds: int = DEFAULT_STREAM_TTL_SECONDS,
        debounce_seconds: float = MATRIX_EDIT_DEBOUNCE_SECONDS,
        initial_text: str = "\u25b8 _thinking..._",
        typing: bool = True,
        max_entries_rendered: int = 12,
    ) -> None:
        self.client = nio_client
        self.room_id = room_id
        self.thread_root = thread_root
        self.bot_name = bot_name
        self.redis = redis_client
        self.session_id = session_id or f"s-{int(time.time() * 1000)}"
        self.skey = stream_key(stream_prefix, self.session_id)
        self.maxlen = maxlen
        self.ttl_seconds = ttl_seconds
        self.debounce = debounce_seconds
        self.initial_text = initial_text
        self.typing_enabled = typing
        self.max_entries_rendered = max_entries_rendered

        self._event_id: Optional[str] = None
        self._entries: list[_LiveEntry] = []
        self._phase = 0
        self._last_edit_at = 0.0
        self._edit_lock = asyncio.Lock()
        self._started_at = time.time()
        self._tools_used: list[str] = []
        self._completed = False
        self._tokens_used = 0

    # -- Lifecycle -------------------------------------------------------

    async def __aenter__(self) -> "AsyncLiveStream":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None and exc is not None:
            try:
                await self.emit(
                    EventType.ERROR,
                    f"{exc_type.__name__}: {exc}"[:500],
                    force_edit=True,
                )
            except Exception:  # noqa: BLE001
                logger.debug("emit ERROR during aexit failed", exc_info=True)
        if not self._completed:
            try:
                await self.complete()
            except Exception:  # noqa: BLE001
                logger.debug("auto complete() during aexit failed", exc_info=True)
        if self.typing_enabled:
            try:
                await self._send_typing(False)
            except Exception:  # noqa: BLE001
                pass

    async def start(self) -> Optional[str]:
        """Send the placeholder and begin the typing indicator."""
        if self.typing_enabled:
            try:
                await self._send_typing(True)
            except Exception:  # noqa: BLE001
                logger.debug("room_typing at start failed", exc_info=True)
        body = self._render_body(status="running", include_initial=True)
        self._event_id = await self._send_new(body)
        return self._event_id

    # -- Emit ------------------------------------------------------------

    async def emit(
        self,
        event_type: EventType,
        content: str = "",
        *,
        tool_name: str = "",
        duration_s: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
        force_edit: bool = False,
    ) -> None:
        """Record an event and update the Matrix message (throttled)."""
        md = dict(metadata or {})
        if tool_name and "tool_name" not in md:
            md["tool_name"] = tool_name
        if duration_s and "duration_s" not in md:
            md["duration_s"] = duration_s

        await self._write_redis(event_type, content, md)

        self._phase += 1

        if event_type == EventType.TOOL_CALL:
            tn = tool_name or md.get("tool_name", "")
            self._entries.append(_LiveEntry(type=event_type, tool=tn, text=content[:160], done=False))
            if tn and tn not in self._tools_used:
                self._tools_used.append(tn)
            force_edit = True
        elif event_type == EventType.TOOL_RESULT:
            tn = tool_name or md.get("tool_name", "")
            matched = False
            for entry in reversed(self._entries):
                if entry.type == EventType.TOOL_CALL and entry.tool == tn and not entry.done:
                    entry.done = True
                    entry.duration_s = duration_s or float(md.get("duration_s", 0) or 0)
                    if content:
                        entry.text = (entry.text + " \u2192 " + content[:160]).strip()
                    matched = True
                    break
            if not matched:
                self._entries.append(
                    _LiveEntry(
                        type=event_type,
                        tool=tn,
                        text=content[:160],
                        duration_s=duration_s,
                        done=True,
                    )
                )
            force_edit = True
        elif event_type == EventType.ERROR:
            self._entries.append(_LiveEntry(type=event_type, text=content[:300], done=True))
            force_edit = True
        elif event_type == EventType.THINKING:
            trimmed = (content or "").strip()[:200]
            if self._entries and self._entries[-1].type == EventType.THINKING and not self._entries[-1].done:
                self._entries[-1].text = trimmed
            else:
                self._entries.append(_LiveEntry(type=event_type, text=trimmed))
        else:
            self._entries.append(_LiveEntry(type=event_type, text=content[:200]))

        await self._maybe_edit(force=force_edit)

    async def complete(
        self,
        *,
        final_text: Optional[str] = None,
        html_body: Optional[str] = None,
        duration_s: Optional[float] = None,
        tokens: int = 0,
        tools_used: Optional[list[str]] = None,
    ) -> None:
        """Final edit — either a clean reply + footer or the last live body."""
        if self._completed:
            return
        self._completed = True
        elapsed = duration_s if duration_s is not None else (time.time() - self._started_at)
        self._tokens_used = tokens or self._tokens_used
        used = tools_used if tools_used is not None else self._tools_used

        await self._write_redis(
            EventType.COMPLETE,
            "",
            {
                "duration_s": round(elapsed, 3),
                "tokens_used": self._tokens_used,
                "tools_used": used,
            },
        )

        if self._event_id is None:
            return

        if final_text is None:
            body = self._render_body(status="done", elapsed=elapsed)
            await self._edit(body)
        else:
            footer_parts = [f"{elapsed:.1f}s"]
            if used:
                footer_parts.append(f"{len(used)} tools")
            if self._tokens_used:
                footer_parts.append(f"{self._tokens_used} tok")
            footer = " \u00b7 ".join(footer_parts)
            plain = f"{final_text}\n\n\u2014 {footer}"
            if html_body is None:
                html_body = _markdown_to_html(final_text)
            combined_html = f"{html_body}<br/><small>\u2014 {html_lib.escape(footer)}</small>"
            await self._edit(plain, html=combined_html)

        if self.redis is not None:
            try:
                maybe = self.redis.expire(self.skey, self.ttl_seconds)
                if asyncio.iscoroutine(maybe):
                    await maybe
            except Exception:  # noqa: BLE001
                pass

    # -- Rendering -------------------------------------------------------

    def _render_header(self, *, status: str, elapsed: Optional[float] = None) -> str:
        elapsed = elapsed if elapsed is not None else (time.time() - self._started_at)
        icon = _STATUS_ICON.get(status, _STATUS_ICON["running"])
        if status == "done":
            bar = _DONE_BAR
        elif status == "error":
            bar = _ERROR_BAR
        else:
            bar = _render_progress(self._phase)
        parts = [f"{icon} **{self.bot_name}**", _render_elapsed(elapsed), bar]
        if self._tools_used:
            parts.append(f"{len(self._tools_used)} tools")
        return " \u00b7 ".join(parts)

    def _render_body(
        self,
        *,
        status: str = "running",
        elapsed: Optional[float] = None,
        include_initial: bool = False,
    ) -> str:
        header = self._render_header(status=status, elapsed=elapsed)
        lines = [header, ""]
        visible = self._entries[-self.max_entries_rendered :]
        for entry in visible:
            if entry.type == EventType.TOOL_CALL:
                marker = "\u2713" if entry.done else "\u25b8"
                dur = f" \u00b7 {entry.duration_s:.1f}s" if entry.done and entry.duration_s else ""
                args_preview = entry.text.strip()
                args_preview = f" {args_preview}" if args_preview else ""
                lines.append(f"{marker} \U0001f527 `{entry.tool}`{args_preview}{dur}".rstrip())
            elif entry.type == EventType.TOOL_RESULT:
                lines.append(f"\u2713 \U0001f4cb {entry.text}".rstrip())
            elif entry.type == EventType.THINKING:
                marker = "\u2713" if entry.done else "\u25b8"
                text = entry.text or "reasoning..."
                lines.append(f"{marker} \U0001f9e0 _{text}_")
            elif entry.type == EventType.ERROR:
                lines.append(f"\u2717 \u274c {entry.text}")
            elif entry.type == EventType.ITERATION:
                lines.append(f"\u25b8 \U0001f504 {entry.text}".rstrip())
            elif entry.type == EventType.ACTION:
                lines.append(f"\u25b8 \u2699\ufe0f {entry.text}".rstrip())
            else:
                lines.append(f"\u25b8 {entry.text}".rstrip())
        if include_initial and not self._entries:
            lines.append(self.initial_text)
        return "\n".join(lines).rstrip()

    # -- Internal --------------------------------------------------------

    async def _maybe_edit(self, *, force: bool = False) -> None:
        if self._event_id is None:
            return
        now = time.time()
        if not force and now - self._last_edit_at < self.debounce:
            return
        body = self._render_body(status="running")
        await self._edit(body)
        self._last_edit_at = now

    async def _send_new(self, body: str) -> Optional[str]:
        content: dict[str, Any] = {
            "msgtype": "m.text",
            "body": body,
            "format": "org.matrix.custom.html",
            "formatted_body": _markdown_to_html(body),
        }
        if self.thread_root:
            content["m.relates_to"] = {
                "rel_type": "m.thread",
                "event_id": self.thread_root,
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": self.thread_root},
            }
        async with self._edit_lock:
            try:
                resp = await self.client.room_send(
                    room_id=self.room_id,
                    message_type="m.room.message",
                    content=content,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("AsyncLiveStream send failed room=%s: %s", self.room_id, exc)
                return None
            return getattr(resp, "event_id", None)

    async def _edit(self, body: str, *, html: Optional[str] = None) -> None:
        if self._event_id is None:
            return
        fmt_html = html if html is not None else _markdown_to_html(body)
        new_content: dict[str, Any] = {
            "msgtype": "m.text",
            "body": body,
            "format": "org.matrix.custom.html",
            "formatted_body": fmt_html,
        }
        if self.thread_root:
            new_content["m.relates_to"] = {
                "rel_type": "m.thread",
                "event_id": self.thread_root,
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": self.thread_root},
            }
        content: dict[str, Any] = {
            "msgtype": "m.text",
            "body": f"* {body}",
            "format": "org.matrix.custom.html",
            "formatted_body": f"* {fmt_html}",
            "m.new_content": new_content,
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": self._event_id,
            },
        }
        async with self._edit_lock:
            try:
                await self.client.room_send(
                    room_id=self.room_id,
                    message_type="m.room.message",
                    content=content,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("AsyncLiveStream edit failed: %s", exc)

    async def _send_typing(self, typing: bool, *, timeout_ms: int = 30_000) -> None:
        if not hasattr(self.client, "room_typing"):
            return
        try:
            await self.client.room_typing(
                room_id=self.room_id,
                typing_state=typing,
                timeout=timeout_ms if typing else 0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("room_typing failed: %s", exc)

    async def _write_redis(
        self,
        event_type: EventType,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        if self.redis is None:
            return
        evt = StreamEvent(
            type=event_type,
            session_id=self.session_id,
            content=content,
            metadata=metadata,
        )
        try:
            fields = evt.to_redis_fields()
            args = sum(fields.items(), ())
            maybe = self.redis.execute_command(
                "XADD",
                self.skey,
                "MAXLEN",
                "~",
                str(self.maxlen),
                "*",
                *args,
            )
            if asyncio.iscoroutine(maybe):
                await maybe
        except Exception as exc:  # noqa: BLE001
            logger.debug("Stream XADD failed: %s", exc)

    # -- Accessors (used by host CommandBot for chronicle mirror etc.) ---

    @property
    def event_id(self) -> Optional[str]:
        return self._event_id


_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_EMPH_RE = re.compile(r"_([^_]+)_")


def _markdown_to_html(text: str) -> str:
    """Lightweight markdown -> HTML for the live view.

    We escape first then re-inject HTML tags, so user-provided angle
    brackets stay literal. Only the three spans that actually appear in
    our live format are handled (inline code, bold, emphasis) — this is
    a render-per-edit hot path, not a full markdown processor.
    """
    escaped = html_lib.escape(text)
    escaped = _CODE_RE.sub(r"<code>\1</code>", escaped)
    escaped = _BOLD_RE.sub(r"<b>\1</b>", escaped)
    escaped = _EMPH_RE.sub(r"<i>\1</i>", escaped)
    return escaped.replace("\n", "<br/>")


async def emit_if_live(
    event_type: EventType,
    content: str = "",
    *,
    tool_name: str = "",
    duration_s: float = 0.0,
    force_edit: bool = False,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Emit into the ambient live stream if one is active, else no-op."""
    stream = current_live_stream.get()
    if stream is None:
        return
    try:
        await stream.emit(
            event_type,
            content,
            tool_name=tool_name,
            duration_s=duration_s,
            metadata=metadata,
            force_edit=force_edit,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("emit_if_live(%s) failed: %s", event_type, exc)


__all__ = [
    "AsyncLiveStream",
    "current_live_stream",
    "emit_if_live",
]
