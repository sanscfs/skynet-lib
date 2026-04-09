"""Synchronous Matrix HTTP client.

Replaces the duplicated Matrix client code across skynet-agent,
skynet-alert-bridge, skynet-sre, and skynet-profile-synthesis.
Uses httpx (not matrix-nio or aiohttp).
"""

from __future__ import annotations

import logging
import time
import uuid

import httpx

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
        extra_content: dict | None = None,
    ) -> str | None:
        """Send a text message. Returns the event_id or None on failure."""
        txn = self._next_txn()
        url = f"{self.homeserver_url}/_matrix/client/v3/rooms/{room_id}/send/m.room.message/{txn}"
        content: dict = {
            "msgtype": "m.text",
            "body": body,
        }
        if reply_to:
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
