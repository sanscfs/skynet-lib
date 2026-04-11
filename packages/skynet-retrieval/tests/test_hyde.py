"""Tests for HyDE query expansion."""

from __future__ import annotations

from skynet_retrieval.hyde import (
    ANCHOR_CHAR_BUDGET,
    CACHE_NAMESPACE,
    DEFAULT_HYDE_MODEL,
    SKELETON_CHAR_BUDGET,
    _build_prompt,
    _cache_key,
    hyde_expand,
)


class _StubCache:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.get_calls = 0
        self.set_calls: list[tuple[str, str, int]] = []

    def get(self, key):
        self.get_calls += 1
        return self.store.get(key)

    def set(self, key, value, ttl):
        self.set_calls.append((key, value, ttl))
        self.store[key] = value


# --- cache key ----------------------------------------------------------


def test_cache_key_stable_across_calls():
    k1 = _cache_key("hello", "skel", "anchor", "qwen3:4b")
    k2 = _cache_key("hello", "skel", "anchor", "qwen3:4b")
    assert k1 == k2
    assert k1.startswith(CACHE_NAMESPACE)


def test_cache_key_changes_on_skeleton_change():
    a = _cache_key("hello", "old", None, "qwen3:4b")
    b = _cache_key("hello", "new", None, "qwen3:4b")
    assert a != b


def test_cache_key_changes_on_model_change():
    a = _cache_key("hello", None, None, "qwen3:4b")
    b = _cache_key("hello", None, None, "qwen3:8b")
    assert a != b


# --- prompt building ----------------------------------------------------


def test_build_prompt_contains_query():
    p = _build_prompt("what about X", None, None)
    assert "User query: what about X" in p
    assert "Hypothetical passage:" in p


def test_build_prompt_truncates_skeleton():
    long_skel = "a" * (SKELETON_CHAR_BUDGET * 3)
    p = _build_prompt("q", long_skel, None)
    # The skeleton section must not exceed the budget.
    assert "a" * SKELETON_CHAR_BUDGET in p
    assert "a" * (SKELETON_CHAR_BUDGET + 1) not in p


def test_build_prompt_truncates_anchor_from_end():
    long_anchor = "xy" * ANCHOR_CHAR_BUDGET
    p = _build_prompt("q", None, long_anchor)
    # Tail of the anchor must be present; head may be cut.
    assert long_anchor[-ANCHOR_CHAR_BUDGET:] in p


# --- hyde_expand --------------------------------------------------------


def test_hyde_expand_calls_llm_once_and_caches():
    llm_calls = []

    def fake_llm(prompt, model):
        llm_calls.append((prompt, model))
        return "Hypothetical answer about plants."

    cache = _StubCache()
    out1 = hyde_expand("плани про рослинку", llm_client=fake_llm, skeleton="skel", cache=cache)
    out2 = hyde_expand("плани про рослинку", llm_client=fake_llm, skeleton="skel", cache=cache)
    assert out1 == "Hypothetical answer about plants."
    assert out2 == "Hypothetical answer about plants."
    assert len(llm_calls) == 1  # second call served from cache
    assert cache.get_calls == 2
    assert len(cache.set_calls) == 1
    assert cache.set_calls[0][2] == 86400  # default TTL
    assert llm_calls[0][1] == DEFAULT_HYDE_MODEL


def test_hyde_expand_empty_query_returns_empty():
    out = hyde_expand("", llm_client=lambda p, m: "x")
    assert out == ""


def test_hyde_expand_whitespace_query_returns_empty():
    out = hyde_expand("   \n  ", llm_client=lambda p, m: "x")
    assert out == ""


def test_hyde_expand_llm_failure_returns_empty():
    def bad_llm(prompt, model):
        raise RuntimeError("ollama down")

    cache = _StubCache()
    out = hyde_expand("q", llm_client=bad_llm, cache=cache)
    assert out == ""
    # No cache entry written on failure.
    assert cache.set_calls == []


def test_hyde_expand_llm_empty_returns_empty():
    out = hyde_expand("q", llm_client=lambda p, m: "   ")
    assert out == ""


def test_hyde_expand_no_cache_still_works():
    calls = []

    def fake_llm(prompt, model):
        calls.append(1)
        return "answer"

    out = hyde_expand("q", llm_client=fake_llm, cache=None)
    assert out == "answer"
    assert len(calls) == 1


def test_hyde_expand_cache_error_falls_back_to_llm():
    class _BrokenCache:
        def get(self, k):
            raise RuntimeError("redis down")

        def set(self, k, v, ttl):
            raise RuntimeError("still down")

    out = hyde_expand("q", llm_client=lambda p, m: "answer", cache=_BrokenCache())
    assert out == "answer"


def test_hyde_expand_passes_model_and_prompt_through():
    captured = {}

    def fake_llm(prompt, model):
        captured["prompt"] = prompt
        captured["model"] = model
        return "hit"

    hyde_expand(
        "why is my plant wilting",
        llm_client=fake_llm,
        skeleton="user: Bogdan, project: skynet",
        anchor="earlier you said drainage matters",
        model="qwen3:4b",
    )
    assert captured["model"] == "qwen3:4b"
    assert "why is my plant wilting" in captured["prompt"]
    assert "Bogdan" in captured["prompt"]
    assert "drainage matters" in captured["prompt"]
