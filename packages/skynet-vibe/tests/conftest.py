"""Shared fixtures and deterministic fakes for the skynet-vibe test suite.

These fakes are intentionally trivial so the tests stay offline and fast.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest


def _hash_embed(text: str, dim: int = 16) -> list[float]:
    """Stable synthetic embedding. NOT semantic -- only for offline tests.

    Uses SHA-256 bytes so identical texts produce identical vectors and
    different texts produce unrelated vectors, then L2-normalizes.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for i in range(dim):
        b = digest[i % len(digest)]
        values.append((b / 255.0) - 0.5)
    # L2 normalize
    norm = sum(x * x for x in values) ** 0.5
    if norm == 0:
        return values
    return [x / norm for x in values]


@pytest.fixture
def hash_embedder():
    """Sync hash-based embedder for tests."""

    def _embed(text: str) -> list[float]:
        return _hash_embed(text)

    return _embed


@pytest.fixture
def async_hash_embedder():
    """Async hash-based embedder for tests."""

    async def _embed(text: str) -> list[float]:
        return _hash_embed(text)

    return _embed


class FakeQdrant:
    """In-memory Qdrant stub matching the subset VibeStore uses."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[Any, dict]] = {}

    def _coll(self, name: str) -> dict[Any, dict]:
        return self.collections.setdefault(name, {})

    async def upsert(self, collection: str, points: list[dict]) -> dict:
        coll = self._coll(collection)
        for p in points:
            coll[p["id"]] = {
                "id": p["id"],
                "vector": list(p["vector"]),
                "payload": dict(p.get("payload", {})),
            }
        return {"status": "ok"}

    async def get_point(self, collection: str, point_id, *, with_vector: bool = False) -> dict | None:
        coll = self._coll(collection)
        p = coll.get(point_id)
        if not p:
            return None
        out = {"id": p["id"], "payload": dict(p["payload"])}
        if with_vector:
            out["vector"] = list(p["vector"])
        return out

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
        from skynet_vibe.affinity import cosine

        coll = self._coll(collection)
        matches = []
        for p in coll.values():
            if not self._matches_filter(p["payload"], filter):
                continue
            score = cosine(vector, p["vector"])
            if score_threshold is not None and score < score_threshold:
                continue
            point = {"id": p["id"], "score": score, "vector": list(p["vector"])}
            if with_payload:
                point["payload"] = dict(p["payload"])
            matches.append(point)
        matches.sort(key=lambda pt: pt["score"], reverse=True)
        return matches[:limit]

    async def set_payload(self, collection: str, point_ids: list, payload: dict) -> dict:
        coll = self._coll(collection)
        for pid in point_ids:
            if pid in coll:
                coll[pid]["payload"].update(payload)
        return {"status": "ok"}

    def _matches_clause(self, payload: dict, clause: dict) -> bool:
        """Evaluate a single filter clause against a payload.

        Supports the subset of Qdrant filter DSL the skynet-vibe code
        actually emits:
          * ``{"key": K, "match": {"value": V}}`` — exact match
          * ``{"key": K, "range": {"gte": N, "lte": N, "gt": N, "lt": N}}``
          * ``{"must": [...]}`` / ``{"should": [...]}`` / ``{"must_not": [...]}``
            — nested sub-filter (same top-level semantics).
        """
        if "must" in clause or "should" in clause or "must_not" in clause:
            return self._matches_filter(payload, clause)
        key = clause.get("key")
        if key is None:
            return True
        actual = payload.get(key)
        if "match" in clause:
            expected = clause["match"].get("value")
            # Qdrant treats missing payload field as non-match.
            return actual == expected
        if "range" in clause:
            rng = clause["range"]
            if actual is None:
                return False
            try:
                if "gte" in rng and not (actual >= rng["gte"]):
                    return False
                if "gt" in rng and not (actual > rng["gt"]):
                    return False
                if "lte" in rng and not (actual <= rng["lte"]):
                    return False
                if "lt" in rng and not (actual < rng["lt"]):
                    return False
            except TypeError:
                return False
            return True
        return True

    def _matches_filter(self, payload: dict, filter: dict | None) -> bool:
        if not filter:
            return True
        if "must" in filter and isinstance(filter["must"], list):
            for clause in filter["must"]:
                if not self._matches_clause(payload, clause):
                    return False
        if "must_not" in filter and isinstance(filter["must_not"], list):
            for clause in filter["must_not"]:
                if self._matches_clause(payload, clause):
                    return False
        if "should" in filter and isinstance(filter["should"], list):
            # Qdrant: ``should`` requires at least one match (disjunction)
            # when present. Empty list -> trivially satisfied (skip).
            should = filter["should"]
            if should and not any(self._matches_clause(payload, clause) for clause in should):
                return False
        return True

    async def count(self, collection: str, *, filter: dict | None = None) -> int:
        coll = self._coll(collection)
        return sum(1 for p in coll.values() if self._matches_filter(p["payload"], filter))

    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        *,
        offset: Any = None,
        filter: dict | None = None,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> tuple[list[dict], Any]:
        coll = self._coll(collection)
        # Stable ordering: by id, so offset semantics are deterministic.
        ordered = sorted(
            (p for p in coll.values() if self._matches_filter(p["payload"], filter)),
            key=lambda pt: str(pt["id"]),
        )
        start = 0
        if offset is not None:
            for i, pt in enumerate(ordered):
                if str(pt["id"]) == str(offset):
                    start = i
                    break
        window = ordered[start : start + limit]
        out: list[dict] = []
        for pt in window:
            entry: dict[str, Any] = {"id": pt["id"]}
            if with_payload:
                entry["payload"] = dict(pt["payload"])
            if with_vector:
                entry["vector"] = list(pt["vector"])
            out.append(entry)
        next_offset = ordered[start + limit]["id"] if start + limit < len(ordered) else None
        return out, next_offset


@pytest.fixture
def fake_qdrant():
    return FakeQdrant()


@pytest.fixture
def fake_llm():
    """Returns an async LLM callable that mimics a structured rerank response."""

    async def _llm(prompt: str) -> str:
        if "Respond with JSON" in prompt:
            return '{"choice": 0, "reason": "top score aligns with current vibe"}'
        # For describe_current_vibe
        return "Leaning mellow and contemplative with strong recent signals."

    return _llm
