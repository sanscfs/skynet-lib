"""Tests for the ``/model`` command argument parser.

Specifically guards against the 2026-04-25 regression where
``/model gemma4:e4b local`` (provider-after-model) was silently
swallowed and produced an override with model="gemma4:e4b local"
and provider=None — that landed every chat call on the default
Mistral upstream and 400'd until the override was cleared by hand.
"""

from __future__ import annotations

import pytest
from skynet_providers import InvalidModelOverride, ModelOverride, parse_model_args


def test_correct_provider_first():
    o = parse_model_args("local gpt-oss:20b")
    assert o.provider == "local"
    assert o.model == "gpt-oss:20b"
    assert o.tier is None


def test_tier_provider_model():
    o = parse_model_args("big phala openai/gpt-oss-120b")
    assert o.tier == "big"
    assert o.provider == "phala"
    assert o.model == "openai/gpt-oss-120b"


def test_reset():
    assert parse_model_args("reset") == ModelOverride()
    assert parse_model_args("default") == ModelOverride()
    assert parse_model_args("") == ModelOverride()
    assert parse_model_args("   ") == ModelOverride()


def test_bare_model_slug_kept_for_legacy():
    """Without a provider, the slug flows to the caller's LLM_BASE_URL."""
    o = parse_model_args("mistral-large-latest")
    assert o.provider is None
    assert o.model == "mistral-large-latest"


def test_provider_after_model_raises():
    """The exact 2026-04-25 incident shape."""
    with pytest.raises(InvalidModelOverride, match="appears after the model slug"):
        parse_model_args("gemma4:e4b local")


def test_provider_after_model_with_tier_raises():
    with pytest.raises(InvalidModelOverride, match="appears after the model slug"):
        parse_model_args("big gemma4:e4b local")


def test_typo_provider_raises():
    with pytest.raises(InvalidModelOverride, match="did you mean 'phala'"):
        parse_model_args("phalla openai/gpt-oss-120b")


def test_typo_provider_local():
    # 1-edit-distance typos (deletion / insertion / substitution) are caught.
    # ``locla`` would be a transposition (2 edits) so it falls through to
    # bare-model-slug; we test the deletion form which the heuristic actually
    # rejects.
    with pytest.raises(InvalidModelOverride, match="did you mean 'local'"):
        parse_model_args("locall gpt-oss:20b")


def test_legacy_author_slug_with_similar_letters_does_not_trigger_typo():
    """``mistralai/mistral-large-2512`` contains 'mistral' but mustn't be flagged."""
    # author/slug shape — full slug is ONE token, not two — so it's a bare
    # model slug, not a provider-position token.
    o = parse_model_args("mistralai/mistral-large-2512")
    assert o.provider is None
    assert o.model == "mistralai/mistral-large-2512"


def test_logging_emits_resolved_components(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="skynet_providers.model")
    parse_model_args("phala openai/gpt-oss-120b")
    assert any("parse_model_args" in r.message for r in caplog.records)
    assert any("provider=phala" in r.message for r in caplog.records)
