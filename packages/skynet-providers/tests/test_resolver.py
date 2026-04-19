"""Resolver-only tests — chat.py needs mocked httpx which lives elsewhere."""

from __future__ import annotations

from skynet_providers.resolver import is_local_endpoint


def test_local_endpoint_matches_ollama_mesh():
    assert is_local_endpoint("http://100.64.0.4:11434")
    assert is_local_endpoint("http://localhost:11434")
    assert is_local_endpoint("http://127.0.0.1:8080")


def test_local_endpoint_rejects_cloud():
    assert not is_local_endpoint("https://api.mistral.ai/v1")
    assert not is_local_endpoint("https://openrouter.ai/api/v1")
    assert not is_local_endpoint("")
