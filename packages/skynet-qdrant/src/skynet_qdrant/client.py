"""Synchronous Qdrant HTTP client.

Replaces the inconsistent mix of qdrant-client, requests, and httpx
wrappers across Skynet components with a single thin httpx-based client.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0


class QdrantClient:
    """Synchronous Qdrant client using the REST API."""

    def __init__(self, url: str = "http://qdrant.qdrant.svc:6333", timeout: float = _DEFAULT_TIMEOUT):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.request(method, f"{self.url}{path}", json=body)
            if resp.status_code >= 400:
                logger.error("Qdrant %s %s -> %s: %s", method, path, resp.status_code, resp.text[:300])
            resp.raise_for_status()
            return resp.json() if resp.content else {}

    def ensure_collection(
        self,
        name: str,
        dim: int = 512,
        distance: str = "Cosine",
        *,
        quantization: dict | None = None,
        replication_factor: int | None = None,
    ) -> bool:
        """Create collection if it doesn't exist. Returns True if created."""
        try:
            resp = self._request("GET", f"/collections/{name}")
            existing_dim = resp.get("result", {}).get("config", {}).get("params", {}).get("vectors", {}).get("size")
            if existing_dim == dim:
                return False
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise

        payload: dict[str, Any] = {
            "vectors": {"size": dim, "distance": distance},
        }
        if quantization:
            payload["quantization_config"] = quantization
        if replication_factor:
            payload["replication_factor"] = replication_factor

        self._request("PUT", f"/collections/{name}", payload)
        logger.info("Created Qdrant collection '%s' (dim=%d)", name, dim)
        return True

    def upsert(self, collection: str, points: list[dict]) -> dict:
        """Upsert points. Each point: {"id": ..., "vector": [...], "payload": {...}}"""
        batch_size = 100
        result = {}
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            result = self._request("PUT", f"/collections/{collection}/points?wait=true", {"points": batch})
        return result

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 5,
        *,
        filter: dict | None = None,
        with_payload: bool = True,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Vector similarity search. Returns list of scored points."""
        body: dict[str, Any] = {
            "vector": vector,
            "limit": limit,
            "with_payload": with_payload,
        }
        if filter:
            body["filter"] = filter
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        resp = self._request("POST", f"/collections/{collection}/points/search", body)
        return resp.get("result", [])

    def scroll(
        self,
        collection: str,
        limit: int = 100,
        *,
        offset: str | int | None = None,
        filter: dict | None = None,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> tuple[list[dict], str | None]:
        """Scroll through points. Returns (points, next_offset)."""
        body: dict[str, Any] = {
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vector,
        }
        if offset is not None:
            body["offset"] = offset
        if filter:
            body["filter"] = filter
        resp = self._request("POST", f"/collections/{collection}/points/scroll", body)
        result = resp.get("result", {})
        return result.get("points", []), result.get("next_page_offset")

    def scroll_all(
        self,
        collection: str,
        *,
        filter: dict | None = None,
        with_payload: bool = True,
        with_vector: bool = False,
        batch_size: int = 100,
    ) -> list[dict]:
        """Scroll all points in a collection."""
        all_points = []
        offset = None
        while True:
            points, next_offset = self.scroll(
                collection,
                limit=batch_size,
                offset=offset,
                filter=filter,
                with_payload=with_payload,
                with_vector=with_vector,
            )
            all_points.extend(points)
            if not next_offset:
                break
            offset = next_offset
        return all_points

    def set_payload(self, collection: str, point_ids: list, payload: dict) -> dict:
        """Update payload fields on specific points."""
        return self._request(
            "POST",
            f"/collections/{collection}/points/payload?wait=true",
            {"payload": payload, "points": point_ids},
        )

    def delete_points(self, collection: str, point_ids: list) -> dict:
        """Delete points by IDs."""
        return self._request(
            "POST",
            f"/collections/{collection}/points/delete?wait=true",
            {"points": point_ids},
        )

    def count(self, collection: str, *, filter: dict | None = None) -> int:
        """Count points in collection."""
        body: dict[str, Any] = {"exact": True}
        if filter:
            body["filter"] = filter
        resp = self._request("POST", f"/collections/{collection}/points/count", body)
        return resp.get("result", {}).get("count", 0)

    def get_point(self, collection: str, point_id, *, with_vector: bool = False) -> dict | None:
        """Get a single point by ID."""
        try:
            resp = self._request("GET", f"/collections/{collection}/points/{point_id}?with_vector={str(with_vector).lower()}")
            return resp.get("result")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def create_payload_index(self, collection: str, field_name: str, field_schema: str) -> dict:
        """Create a payload index. field_schema: 'keyword', 'integer', 'float', 'datetime', 'text'."""
        return self._request(
            "PUT",
            f"/collections/{collection}/index",
            {"field_name": field_name, "field_schema": field_schema},
        )

    def create_snapshot(self, collection: str) -> dict:
        """Create a collection snapshot. Returns snapshot metadata."""
        return self._request("POST", f"/collections/{collection}/snapshots")

    def list_collections(self) -> list[str]:
        """List all collection names."""
        resp = self._request("GET", "/collections")
        return [c["name"] for c in resp.get("result", {}).get("collections", [])]

    def collection_info(self, collection: str) -> dict:
        """Get collection info (config, point count, etc.)."""
        resp = self._request("GET", f"/collections/{collection}")
        return resp.get("result", {})

    def healthy(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            self._request("GET", "/collections")
            return True
        except Exception:
            return False
