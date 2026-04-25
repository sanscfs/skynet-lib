"""Shared fixtures for skynet-orchestration tests.

Provides a ``FakeRedis`` with the subset of commands the package
uses (hashes, lists, pipelines, xadd) plus a deterministic ``cosine``
helper so gate tests don't need real embeddings.
"""

from __future__ import annotations

import pytest

from ._fake_redis import FakeRedis


@pytest.fixture(autouse=True)
def _hmac_secret(monkeypatch):
    """Inject a stable HMAC secret so token mint/verify works in tests."""
    monkeypatch.setenv("ORCHESTRATION_HMAC_SECRET", "test-secret-for-unit-tests")


@pytest.fixture
def redis():
    return FakeRedis()


@pytest.fixture
def cosine():
    """Trivial similarity: shared-token Jaccard.

    Good enough to drive gate logic without pulling skynet-embedding
    into the test environment. Returns a value in [0, 1].
    """

    def _sim(a: str, b: str) -> float:
        ta = set((a or "").lower().split())
        tb = set((b or "").lower().split())
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        return inter / union

    return _sim


@pytest.fixture
def specificity():
    """Token count + uppercase-anchor count -- structural, not lexical."""
    import re

    UPPER = re.compile(r"\b[A-Z][A-Za-z0-9_-]+")

    def _spec(q: str) -> float:
        tokens = (q or "").split()
        anchors = len(UPPER.findall(q or ""))
        return float(len(tokens)) + 2.0 * anchors

    return _spec
