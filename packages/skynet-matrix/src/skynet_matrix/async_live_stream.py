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
        heartbeat_seconds: float = 3.0,
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
        # Heartbeat: when > ``heartbeat_seconds`` passes without any other
        # edit, a background task bumps the progress bar + elapsed timer
        # so the user sees motion even when the LLM call or a tool are
        # silently in-flight. Set to 0 to disable.
        self.heartbeat_seconds = heartbeat_seconds
        self._heartbeat_task: Optional[asyncio.Task[None]] = None

        self._event_id: Optional[str] = None
        self._entries: list[_LiveEntry] = []
        self._phase = 0
        self._last_edit_at = 0.0
        self._edit_lock = asyncio.Lock()
        self._started_at = time.time()
        self._tools_used: list[str] = []
        self._completed = False
        self._tokens_used = 0
        # Live LLM buffer (Phase 4: per-token visibility). While a chat
        # completion is streaming, ``_llm_buffer`` collects the raw
        # assistant output and the tail is rendered as the last live
        # line. Cleared on ``finish_llm()``.
        self._llm_buffer: str = ""
        self._llm_active: bool = False
        self._llm_started_at: float = 0.0
        self._llm_token_preview_chars: int = 160

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
        # Record the placeholder send as an "edit" so the heartbeat
        # doesn't fire immediately — it only kicks in after the first
        # silence window of ``heartbeat_seconds``.
        self._last_edit_at = time.time()
        if self.heartbeat_seconds > 0 and self._event_id is not None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        return self._event_id

    async def _heartbeat_loop(self) -> None:
        """Keep the live message visibly moving during long silent waits.

        Only re-edits when nothing else has touched the message in the
        last ``heartbeat_seconds`` window, so a chatty turn never hits
        the heartbeat (and the Matrix homeserver never sees back-to-back
        edits from both paths). Silent turns (LLM 10s thinking, slow
        tool) get a bumped progress bar + refreshed elapsed timer every
        heartbeat tick, which is the only thing the user sees move.
        """
        interval = self.heartbeat_seconds
        try:
            while not self._completed:
                await asyncio.sleep(interval)
                if self._completed:
                    return
                silence = time.time() - self._last_edit_at
                if silence < interval:
                    continue
                self._phase += 1
                body = self._render_body(status="running")
                await self._edit(body)
                self._last_edit_at = time.time()
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001
            logger.debug("heartbeat loop crashed: %s", exc)

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
                # Fresh THINKING entry → anchor so the user sees the first
                # reasoning line as soon as it arrives; subsequent THINKING
                # updates (same entry overwritten) ride the debounce.
                force_edit = True
        else:
            self._entries.append(_LiveEntry(type=event_type, text=content[:200]))

        await self._maybe_edit(force=force_edit)

    # -- LLM per-token streaming (Phase 4) --------------------------------

    async def start_llm(self, *, url: str = "", model: str = "") -> None:
        """Open a live LLM request. Emitted by providers at request start.

        Adds one anchor entry (``POST host model``) and resets the token
        buffer. The buffer is rendered as a live tail under the entries
        stack until :meth:`finish_llm` closes the block.
        """
        self._llm_buffer = ""
        self._llm_active = True
        self._llm_started_at = time.time()
        host = _host_only(url) if url else "LLM"
        label = f"POST {host}" + (f" \u00b7 {model}" if model else "")
        self._entries.append(_LiveEntry(type=EventType.ACTION, text=label, done=False))
        self._phase += 1
        await self._write_redis(
            EventType.ACTION,
            label,
            {"kind": "llm_request", "url": url, "model": model},
        )
        await self._maybe_edit(force=True)

    async def append_token(self, chunk: str) -> None:
        """Extend the live LLM buffer with one streamed delta."""
        if not chunk:
            return
        if not self._llm_active:
            # Token arrived without ``start_llm`` — tolerate it by
            # opening an anonymous request block so we still render.
            self._llm_active = True
            self._llm_started_at = time.time()
        self._llm_buffer += chunk
        # Phase bumps per-chunk so the progress cells animate even
        # while the single LLM entry stays in place.
        self._phase += 1
        now = time.time()
        if now - self._last_edit_at >= self.debounce:
            body = self._render_body(status="running")
            await self._edit(body)
            self._last_edit_at = now

    async def finish_llm(self, *, tokens: int = 0) -> None:
        """Close the live LLM block — freezes the entry, clears the buffer."""
        if not self._llm_active:
            return
        elapsed = time.time() - self._llm_started_at
        self._llm_active = False
        # Bump token usage so the final footer is accurate; if the caller
        # didn't pass a count, fall back to a whitespace heuristic.
        self._tokens_used += tokens or _estimate_tokens(self._llm_buffer)
        for entry in reversed(self._entries):
            if entry.type == EventType.ACTION and not entry.done:
                entry.done = True
                entry.duration_s = elapsed
                break
        await self._write_redis(
            EventType.ACTION_RESULT,
            f"{len(self._llm_buffer)} chars",
            {"kind": "llm_response", "duration_s": round(elapsed, 3)},
        )
        self._llm_buffer = ""
        await self._maybe_edit(force=True)

    async def complete(
        self,
        *,
        final_text: Optional[str] = None,
        html_body: Optional[str] = None,
        duration_s: Optional[float] = None,
        tokens: int = 0,
        tools_used: Optional[list[str]] = None,
    ) -> None:
        """Final edit — either a clean reply + footer or the last live body.

        When ``final_text`` is None *and* the handler never emitted any
        intermediate tool/LLM events, the placeholder is redacted instead
        of edited to a "done" header — a silent turn must produce zero
        visible messages, otherwise a batch of no-op replays at startup
        spams the room with bare status bars.
        """
        if self._completed:
            return
        self._completed = True
        # Stop the heartbeat first so we don't race with the final edit.
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
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
            if not self._entries:
                # Nothing happened — remove the placeholder entirely.
                await self._redact_placeholder()
                return
            body = self._render_body(status="done", elapsed=elapsed)
            await self._edit(body)
        else:
            footer_parts = [f"{elapsed:.1f}s"]
            if used:
                footer_parts.append(f"{len(used)} tools")
            if self._tokens_used:
                footer_parts.append(f"{self._tokens_used} tok")
            footer = " \u00b7 ".join(footer_parts)

            # Render the trail (tool calls, thinking lines, errors)
            # that fired during the turn, so the final edit keeps the
            # reasoning visible in a collapsed ``<details>`` block \u2014
            # Element renders this as expandable. Plain-body fallback
            # appends the trail under a "\u2500\u2500 trace" delimiter so a
            # client that doesn't render HTML still has the raw text.
            trail_lines = self._render_trail_lines()
            plain_parts = [final_text, "", f"\u2014 {footer}"]
            if trail_lines:
                plain_parts.append("")
                plain_parts.append("\u2500\u2500 trace")
                plain_parts.extend(trail_lines)
            plain = "\n".join(plain_parts)

            if html_body is None:
                html_body = _markdown_to_html(final_text)
            html_chunks = [
                html_body,
                f"<br/><small>\u2014 {html_lib.escape(footer)}</small>",
            ]
            if trail_lines:
                trail_html = "<br/>".join(_markdown_to_html(line) for line in trail_lines)
                html_chunks.append(
                    f"<br/><details><summary><i>trace ({len(trail_lines)} steps)</i></summary>"
                    f"<blockquote>{trail_html}</blockquote></details>"
                )
            combined_html = "".join(html_chunks)
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

    def _render_trail_lines(self) -> list[str]:
        """Return the per-step trail (tools / thinking / errors) for the
        collapsed reasoning block on the final edit.

        Same vocabulary as ``_render_body`` but without the header — the
        caller already prints a clean reply on top, the trail is only
        the historical sequence of what happened during the turn.
        """
        lines: list[str] = []
        for entry in self._entries:
            if entry.type == EventType.TOOL_CALL:
                marker = "✓" if entry.done else "▸"
                dur = f" · {entry.duration_s:.1f}s" if entry.done and entry.duration_s else ""
                args_preview = entry.text.strip()
                args_preview = f" {args_preview}" if args_preview else ""
                lines.append(f"{marker} 🔧 `{entry.tool}`{args_preview}{dur}".rstrip())
            elif entry.type == EventType.TOOL_RESULT:
                lines.append(f"✓ 📋 {entry.text}".rstrip())
            elif entry.type == EventType.THINKING:
                lines.append(f"✓ 🧠 _{entry.text or 'reasoning…'}_")
            elif entry.type == EventType.ERROR:
                lines.append(f"✗ ❌ {entry.text}")
            elif entry.type == EventType.ITERATION:
                lines.append(f"▸ 🔄 {entry.text}".rstrip())
            elif entry.type == EventType.ACTION:
                marker = "✓" if entry.done else "▸"
                dur = f" · {entry.duration_s:.1f}s" if entry.done and entry.duration_s else ""
                lines.append(f"{marker} → {entry.text}{dur}".rstrip())
            else:
                if entry.text:
                    lines.append(f"▸ {entry.text}".rstrip())
        return lines

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
                marker = "\u2713" if entry.done else "\u25b8"
                arrow = "\u2192"  # →
                dur = f" \u00b7 {entry.duration_s:.1f}s" if entry.done and entry.duration_s else ""
                lines.append(f"{marker} {arrow} {entry.text}{dur}".rstrip())
            else:
                lines.append(f"\u25b8 {entry.text}".rstrip())
        if self._llm_active and self._llm_buffer:
            tail = self._llm_buffer[-self._llm_token_preview_chars :]
            if len(self._llm_buffer) > self._llm_token_preview_chars:
                tail = "\u2026" + tail
            tail = tail.replace("\n", " ")
            lines.append(f"\u25b8 \u23f3 `{tail}`")
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

    async def _redact_placeholder(self) -> None:
        """Redact the initial placeholder event (silent-turn cleanup)."""
        if self._event_id is None:
            return
        try:
            redact = getattr(self.client, "room_redact", None)
            if redact is None:
                return
            await redact(
                room_id=self.room_id,
                event_id=self._event_id,
                reason="skynet: silent turn — no content",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("redact placeholder failed: %s", exc)

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


async def emit_llm_start_if_live(url: str = "", model: str = "") -> None:
    """Called by skynet-providers at POST /chat/completions start.

    Opens an anchor entry ``→ POST host · model`` in the live view so
    the user sees which endpoint is being contacted. Safe no-op when
    streaming is disabled or skynet-matrix isn't wired into the caller.
    """
    stream = current_live_stream.get()
    if stream is None:
        return
    try:
        await stream.start_llm(url=url, model=model)
    except Exception as exc:  # noqa: BLE001
        logger.debug("emit_llm_start_if_live failed: %s", exc)


async def emit_token_if_live(chunk: str) -> None:
    """Called by skynet-providers for each SSE delta token."""
    stream = current_live_stream.get()
    if stream is None:
        return
    try:
        await stream.append_token(chunk)
    except Exception as exc:  # noqa: BLE001
        logger.debug("emit_token_if_live failed: %s", exc)


async def emit_llm_end_if_live(tokens: int = 0) -> None:
    """Called by skynet-providers once the SSE stream closes."""
    stream = current_live_stream.get()
    if stream is None:
        return
    try:
        await stream.finish_llm(tokens=tokens)
    except Exception as exc:  # noqa: BLE001
        logger.debug("emit_llm_end_if_live failed: %s", exc)


def _host_only(url: str) -> str:
    """Trim a URL to ``host[:port]`` for compact rendering in the live view."""
    if not url:
        return ""
    s = url.split("://", 1)[-1]
    s = s.split("/", 1)[0]
    return s


def _estimate_tokens(text: str) -> int:
    """Very rough token estimate (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


__all__ = [
    "AsyncLiveStream",
    "current_live_stream",
    "emit_if_live",
    "emit_llm_start_if_live",
    "emit_token_if_live",
    "emit_llm_end_if_live",
]
