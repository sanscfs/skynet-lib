"""Synchronous Matrix HTTP client.

Replaces the duplicated Matrix client code across skynet-agent,
skynet-alert-bridge, skynet-sre, and skynet-profile-synthesis.
Uses httpx (not matrix-nio or aiohttp).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from skynet_matrix.wrap import build_edit_payload, build_footer_payload

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15.0


class MatrixClient:
    """Sync Matrix client using the Client-Server API v3."""

    def __init__(
        self,
        homeserver_url: str = "http://conduwuit.matrix.svc:6167",
        token: str = "",
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self.homeserver_url = homeserver_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._txn_counter = int(time.time() * 1000)

    def _next_txn(self) -> str:
        self._txn_counter += 1
        return f"skynet_{self._txn_counter}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def send_text(
        self,
        room_id: str,
        body: str,
        *,
        reply_to: str | None = None,
        thread_root: str | None = None,
        extra_content: dict | None = None,
    ) -> str | None:
        """Send a text message. Returns the event_id or None on failure.

        Args:
            reply_to: event_id to reply to (in-reply-to, no thread).
            thread_root: event_id of the thread root. Creates/continues
                a Matrix thread. Takes precedence over reply_to.
        """
        txn = self._next_txn()
        url = f"{self.homeserver_url}/_matrix/client/v3/rooms/{room_id}/send/m.room.message/{txn}"
        content: dict = {
            "msgtype": "m.text",
            "body": body,
        }
        if thread_root:
            content["m.relates_to"] = {
                "rel_type": "m.thread",
                "event_id": thread_root,
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": thread_root},
            }
        elif reply_to:
            content["m.relates_to"] = {
                "m.in_reply_to": {"event_id": reply_to},
            }
        if extra_content:
            content.update(extra_content)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.put(url, headers=self._headers(), json=content)
                if resp.is_success:
                    return resp.json().get("event_id")
                logger.warning("Matrix send_text %s -> %s: %s", room_id, resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("Matrix send_text failed: %s", e)
        return None

    def send_with_footer(
        self,
        room_id: str,
        body: str,
        *,
        trace_id: str = "",
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
        reply_to: str | None = None,
        thread_root: str | None = None,
        extra_content: dict | None = None,
        trace_meta_extra: dict | None = None,
    ) -> str | None:
        """Send a text message with a trace footer appended.

        This is the canonical send path for Skynet components -- DAGs, chat
        agents, MCP tools -- so every Matrix message shares the same
        ``<small>... | ...</small>`` trailer and the same
        ``dev.skynet.trace`` metadata shape. With no trace fields set the
        behaviour degrades to a plain ``send_text``.

        Args:
            room_id: target Matrix room.
            body: plain text body.
            trace_id/duration_s/.../service: footer fields (see
                ``skynet_matrix.trace_footer.format_trace_footer``).
            formatted_body: pre-rendered HTML body (without footer). If
                omitted and footer is non-empty, the plain body is used as
                HTML and the footer appended.
            reply_to/thread_root: Matrix relation (thread_root takes
                precedence).
            extra_content: additional keys merged into the event content.
            trace_meta_extra: extra keys merged into ``dev.skynet.trace``.

        Returns:
            event_id on success, None on failure.
        """
        combined_body, extra = build_footer_payload(
            body,
            trace_id=trace_id,
            duration_s=duration_s,
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            steps=steps,
            rag_sources=rag_sources,
            tools_used=tools_used,
            cost_usd=cost_usd,
            service=service,
            formatted_body=formatted_body,
            extra_content=extra_content,
            trace_meta_extra=trace_meta_extra,
        )
        return self.send_text(
            room_id,
            combined_body,
            reply_to=reply_to,
            thread_root=thread_root,
            extra_content=extra,
        )

    def edit_message(
        self,
        room_id: str,
        event_id: str,
        new_body: str,
    ) -> str | None:
        """Edit an existing message. Returns new event_id or None."""
        txn = self._next_txn()
        url = f"{self.homeserver_url}/_matrix/client/v3/rooms/{room_id}/send/m.room.message/{txn}"
        content = {
            "msgtype": "m.text",
            "body": f"* {new_body}",
            "m.new_content": {
                "msgtype": "m.text",
                "body": new_body,
            },
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": event_id,
            },
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.put(url, headers=self._headers(), json=content)
                if resp.is_success:
                    return resp.json().get("event_id")
                logger.warning("Matrix edit %s -> %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("Matrix edit failed: %s", e)
        return None

    def edit_with_footer(
        self,
        room_id: str,
        event_id: str,
        new_body: str,
        *,
        trace_id: str = "",
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
        thread_root: str | None = None,
        trace_meta_extra: dict | None = None,
    ) -> str | None:
        """Edit an existing message and (re-)attach the trace footer.

        Used by streaming live edits: the final edit carries the full
        footer + ``dev.skynet.trace`` metadata. ``thread_root`` keeps the
        edit inside an existing Matrix thread (MSC3440); omit for top-level
        edits.
        """
        combined_body, combined_formatted, trace_meta = build_edit_payload(
            new_body,
            trace_id=trace_id,
            duration_s=duration_s,
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            steps=steps,
            rag_sources=rag_sources,
            tools_used=tools_used,
            cost_usd=cost_usd,
            service=service,
            formatted_body=formatted_body,
            trace_meta_extra=trace_meta_extra,
        )

        txn = self._next_txn()
        url = f"{self.homeserver_url}/_matrix/client/v3/rooms/{room_id}/send/m.room.message/{txn}"
        new_content: dict[str, Any] = {
            "msgtype": "m.text",
            "body": combined_body,
        }
        if combined_formatted is not None:
            new_content["format"] = "org.matrix.custom.html"
            new_content["formatted_body"] = combined_formatted
        if thread_root:
            new_content["m.relates_to"] = {
                "rel_type": "m.thread",
                "event_id": thread_root,
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": thread_root},
            }

        content: dict[str, Any] = {
            "msgtype": "m.text",
            "body": f"* {combined_body}",
            "m.new_content": new_content,
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": event_id,
            },
        }
        if combined_formatted is not None:
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = f"* {combined_formatted}"
        if trace_meta is not None:
            content["dev.skynet.trace"] = trace_meta

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.put(url, headers=self._headers(), json=content)
                if resp.is_success:
                    return resp.json().get("event_id")
                logger.warning("Matrix edit_with_footer %s -> %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("Matrix edit_with_footer failed: %s", e)
        return None

    def send_reaction(
        self,
        room_id: str,
        event_id: str,
        emoji: str,
    ) -> str | None:
        """Send an emoji reaction to a message."""
        txn = self._next_txn()
        url = f"{self.homeserver_url}/_matrix/client/v3/rooms/{room_id}/send/m.reaction/{txn}"
        content = {
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": event_id,
                "key": emoji,
            },
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.put(url, headers=self._headers(), json=content)
                if resp.is_success:
                    return resp.json().get("event_id")
        except Exception as e:
            logger.warning("Matrix reaction failed: %s", e)
        return None

    def download_media(self, mxc_url: str) -> bytes | None:
        """Download media from mxc:// URL. Returns bytes or None."""
        if not mxc_url.startswith("mxc://"):
            return None
        server_media = mxc_url[6:]  # strip "mxc://"
        url = f"{self.homeserver_url}/_matrix/media/v1/download/{server_media}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url, headers=self._headers())
                if resp.is_success:
                    return resp.content
        except Exception as e:
            logger.warning("Matrix download failed: %s", e)
        return None
