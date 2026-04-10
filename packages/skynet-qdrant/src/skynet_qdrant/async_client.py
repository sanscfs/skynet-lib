"""Asynchronous Qdrant HTTP client.

Async mirror of QdrantClient for components using asyncio
(skynet-matrix-bridge, skynet-memory-ui, skynet-profile-page).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0


class AsyncQdrantClient:
    """Async Qdrant client using the REST API."""

    def __init__(self, url: str = "http://qdrant.qdrant.svc:6333", timeout: float = _DEFAULT_TIMEOUT):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.url, timeout=self.timeout)
        return self._client

    async def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        client = await self._get_client()
        resp = await client.request(method, path, json=body)
        if resp.status_code >= 400:
            logger.error("Qdrant %s %s -> %s: %s", method, path, resp.status_code, resp.text[:300])
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def ensure_collection(
        self,
        name: str,
        dim: int = 512,
        distance: str = "Cosine",
        *,
        quantization: dict | None = None,
        replication_factor: int | None = None,
    ) -> bool:
        try:
            resp = await self._request("GET", f"/collections/{name}")
            existing_dim = resp.get("result", {}).get("config", {}).get("params", {}).get("vectors", {}).get("size")
            if existing_dim == dim:
                return False
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise

        payload: dict[str, Any] = {"vectors": {"size": dim, "distance": distance}}
        if quantization:
            payload["quantization_config"] = quantization
        if replication_factor:
            payload["replication_factor"] = replication_factor

        await self._request("PUT", f"/collections/{name}", payload)
        logger.info("Created Qdrant collection '%s' (dim=%d)", name, dim)
        return True

    async def upsert(self, collection: str, points: list[dict]) -> dict:
        batch_size = 100
        result = {}
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            result = await self._request("PUT", f"/collections/{collection}/points?wait=true", {"points": batch})
        return result

    async def update_vectors(self, collection: str, points: list[dict]) -> dict:
        """Update vectors in-place without touching payloads. Batched."""
        batch_size = 100
        result: dict = {}
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            result = await self._request(
                "PUT",
                f"/collections/{collection}/points/vectors?wait=true",
                {"points": batch},
            )
        return result

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 5,
        *,
        filter: dict | None = None,
        with_payload: bool = True,
        score_threshold: float | None = None,
    ) -> list[dict]:
        body: dict[str, Any] = {"vector": vector, "limit": limit, "with_payload": with_payload}
        if filter:
            body["filter"] = filter
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        resp = await self._request("POST", f"/collections/{collection}/points/search", body)
        return resp.get("result", [])

    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        *,
        offset: str | int | None = None,
        filter: dict | None = None,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> tuple[list[dict], str | None]:
        body: dict[str, Any] = {"limit": limit, "with_payload": with_payload, "with_vector": with_vector}
        if offset is not None:
            body["offset"] = offset
        if filter:
            body["filter"] = filter
        resp = await self._request("POST", f"/collections/{collection}/points/scroll", body)
        result = resp.get("result", {})
        return result.get("points", []), result.get("next_page_offset")

    async def scroll_all(
        self,
        collection: str,
        *,
        filter: dict | None = None,
        with_payload: bool = True,
        with_vector: bool = False,
        batch_size: int = 100,
    ) -> list[dict]:
        all_points = []
        offset = None
        while True:
            points, next_offset = await self.scroll(
                collection, limit=batch_size, offset=offset, filter=filter,
                with_payload=with_payload, with_vector=with_vector,
            )
            all_points.extend(points)
            if not next_offset:
                break
            offset = next_offset
        return all_points

    async def set_payload(self, collection: str, point_ids: list, payload: dict) -> dict:
        return await self._request(
            "POST", f"/collections/{collection}/points/payload?wait=true",
            {"payload": payload, "points": point_ids},
        )

    async def delete_points(self, collection: str, point_ids: list) -> dict:
        return await self._request(
            "POST", f"/collections/{collection}/points/delete?wait=true",
            {"points": point_ids},
        )

    async def count(self, collection: str, *, filter: dict | None = None) -> int:
        body: dict[str, Any] = {"exact": True}
        if filter:
            body["filter"] = filter
        resp = await self._request("POST", f"/collections/{collection}/points/count", body)
        return resp.get("result", {}).get("count", 0)

    async def get_point(self, collection: str, point_id, *, with_vector: bool = False) -> dict | None:
        try:
            resp = await self._request("GET", f"/collections/{collection}/points/{point_id}?with_vector={str(with_vector).lower()}")
            return resp.get("result")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def create_payload_index(self, collection: str, field_name: str, field_schema: str) -> dict:
        return await self._request("PUT", f"/collections/{collection}/index", {"field_name": field_name, "field_schema": field_schema})

    async def healthy(self) -> bool:
        try:
            await self._request("GET", "/collections")
            return True
        except Exception:
            return False
