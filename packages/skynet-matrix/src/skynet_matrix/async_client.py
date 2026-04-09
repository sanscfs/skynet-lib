"""Asynchronous Matrix HTTP client.

Async mirror of MatrixClient for components using asyncio
(skynet-alert-bridge, skynet-matrix-bridge, skynet-sre).
"""

from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15.0


class AsyncMatrixClient:
    """Async Matrix client using the Client-Server API v3."""

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
        self._client: httpx.AsyncClient | None = None

    def _next_txn(self) -> str:
        self._txn_counter += 1
        return f"skynet_{self._txn_counter}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.homeserver_url,
                headers=self._headers(),
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def send_text(
        self,
        room_id: str,
        body: str,
        *,
        reply_to: str | None = None,
        extra_content: dict | None = None,
    ) -> str | None:
        """Send a text message. Returns event_id or None."""
        txn = self._next_txn()
        url = f"/_matrix/client/v3/rooms/{room_id}/send/m.room.message/{txn}"
        content: dict = {"msgtype": "m.text", "body": body}
        if reply_to:
            content["m.relates_to"] = {"m.in_reply_to": {"event_id": reply_to}}
        if extra_content:
            content.update(extra_content)

        try:
            client = await self._get_client()
            resp = await client.put(url, json=content)
            if resp.is_success:
                return resp.json().get("event_id")
            logger.warning("Matrix async send_text %s -> %s", room_id, resp.status_code)
        except Exception as e:
            logger.warning("Matrix async send_text failed: %s", e)
        return None

    async def edit_message(self, room_id: str, event_id: str, new_body: str) -> str | None:
        """Edit an existing message."""
        txn = self._next_txn()
        url = f"/_matrix/client/v3/rooms/{room_id}/send/m.room.message/{txn}"
        content = {
            "msgtype": "m.text",
            "body": f"* {new_body}",
            "m.new_content": {"msgtype": "m.text", "body": new_body},
            "m.relates_to": {"rel_type": "m.replace", "event_id": event_id},
        }
        try:
            client = await self._get_client()
            resp = await client.put(url, json=content)
            if resp.is_success:
                return resp.json().get("event_id")
        except Exception as e:
            logger.warning("Matrix async edit failed: %s", e)
        return None

    async def send_reaction(self, room_id: str, event_id: str, emoji: str) -> str | None:
        """Send an emoji reaction."""
        txn = self._next_txn()
        url = f"/_matrix/client/v3/rooms/{room_id}/send/m.reaction/{txn}"
        content = {
            "m.relates_to": {"rel_type": "m.annotation", "event_id": event_id, "key": emoji},
        }
        try:
            client = await self._get_client()
            resp = await client.put(url, json=content)
            if resp.is_success:
                return resp.json().get("event_id")
        except Exception as e:
            logger.warning("Matrix async reaction failed: %s", e)
        return None

    async def sync(
        self,
        since: str | None = None,
        timeout_ms: int = 5000,
        *,
        room_filter: dict | None = None,
    ) -> dict:
        """Long-poll sync. Returns sync response dict."""
        params: dict = {"timeout": str(timeout_ms)}
        if since:
            params["since"] = since
        if room_filter:
            import json
            params["filter"] = json.dumps(room_filter)

        try:
            client = await self._get_client()
            resp = await client.get("/_matrix/client/v3/sync", params=params)
            if resp.is_success:
                return resp.json()
        except Exception as e:
            logger.warning("Matrix async sync failed: %s", e)
        return {}
