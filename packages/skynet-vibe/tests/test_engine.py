"""Tests for VibeEngine absorb + absorb_emoji + suggest + describe + explain."""

from __future__ import annotations

import pytest
from skynet_vibe import (
    PrototypeRegistry,
    Source,
    VibeEngine,
    VibeStore,
)
from skynet_vibe.emoji import clear_cache


@pytest.fixture
def engine(fake_qdrant, async_hash_embedder, fake_llm):
    store = VibeStore(fake_qdrant, collection="t")
    prototypes = PrototypeRegistry(async_hash_embedder)
    return VibeEngine(
        store=store,
        prototypes=prototypes,
        embedder=async_hash_embedder,
        llm_client=fake_llm,
        decay_half_life_days=45.0,
    )


@pytest.mark.asyncio
async def test_absorb_stores_signal(engine, fake_qdrant):
    sig = await engine.absorb(
        text="this album is exactly my evening mood",
        source=Source(type="chat", room_id="!music"),
        confidence=0.9,
        linked_rec_id="rec_1",
        context_text="raining tonight",
    )
    assert sig.id
    assert sig.vectors.content
    assert sig.vectors.context  # context_text was provided
    assert sig.vectors.user_state is None
    # It was written to the fake qdrant
    got = await engine.store.get(sig.id)
    assert got is not None
    assert got.linked_rec_id == "rec_1"


@pytest.mark.asyncio
async def test_absorb_emoji_uses_vibe_phrase(engine):
    clear_cache()
    sig = await engine.absorb_emoji(
        emoji="🔥", source=Source(type="reaction", room_id="!music"), linked_rec_id="rec_abc"
    )
    assert sig.extra_payload["emoji"] == "🔥"
    assert "intense" in sig.extra_payload["vibe_phrase"]
    assert sig.source.type == "reaction"
    assert sig.linked_rec_id == "rec_abc"


@pytest.mark.asyncio
async def test_absorb_emoji_unknown_raises(engine):
    from skynet_vibe import EmbeddingError

    clear_cache()
    with pytest.raises(EmbeddingError):
        await engine.absorb_emoji(emoji="🧿", source=Source(type="reaction"))


@pytest.mark.asyncio
async def test_suggest_returns_top_candidate(engine):
    """End-to-end suggest path with manually-controlled vectors.

    We build the store + prototype so the weighted target is the
    prototype centroid, then check that the candidate aligned with the
    centroid wins over its negation.
    """
    from datetime import datetime, timezone

    from skynet_vibe import FacetVectors, VibeSignal
    from skynet_vibe import Source as Src

    # Register domain prototype with a fixed centroid.
    await engine.prototypes.add("music", ["albums tracks melodies"])
    centroid = (await engine.prototypes.get("music")).centroid

    # Insert one signal whose content IS the centroid -> positive cosine with proto.
    sig = VibeSignal(
        id="seed-1",
        text_raw="aligned",
        vectors=FacetVectors(content=list(centroid)),
        source=Src(type="chat"),
        timestamp=datetime.now(timezone.utc),
        confidence=1.0,
    )
    await engine.store.put(sig)

    # Candidates: one aligned, one anti-aligned.
    close_vec = list(centroid)
    far_vec = [-x for x in centroid]

    candidates = [
        {"id": "rec_close", "title": "Close", "description": "aligned", "vector": close_vec},
        {"id": "rec_far", "title": "Far", "description": "anti", "vector": far_vec},
    ]

    # No context_text -> seed is prototype centroid; weighted sum stays aligned.
    result = await engine.suggest(
        candidates=candidates,
        domain="music",
        context_text=None,
        top_k=2,
    )
    assert result.candidate["id"] == "rec_close"
    assert result.rec_id
    assert result.reason
    assert result.top_contributing_signals
    # The seed signal should be in the contributors.
    contributor_ids = [s[0] for s in result.top_contributing_signals]
    assert "seed-1" in contributor_ids


@pytest.mark.asyncio
async def test_suggest_requires_domain_or_context(engine):
    with pytest.raises(ValueError):
        await engine.suggest(
            candidates=[{"id": "x", "vector": [0.1] * 16}], domain=None, context_text=None
        )


@pytest.mark.asyncio
async def test_suggest_empty_candidates_raises(engine):
    with pytest.raises(ValueError):
        await engine.suggest(candidates=[], domain="music", context_text="evening")


@pytest.mark.asyncio
async def test_explain_signal_returns_breakdown(engine):
    await engine.prototypes.add("music", ["albums tracks melodies"])
    sig = await engine.absorb(
        text="slow evening listening", source=Source(type="chat", room_id="!music")
    )
    breakdown = await engine.explain_signal(sig.id, domain="music", context_text="rain")
    # All components present
    for key in (
        "id",
        "text_raw",
        "source_type",
        "source_trust",
        "time_decay",
        "prototype_cosine",
        "prototype_term",
        "context_cosine",
        "context_term",
        "final_weight",
        "half_life_days",
        "context_alpha",
    ):
        assert key in breakdown, f"missing key {key!r}"
    assert breakdown["source_type"] == "chat"
    assert breakdown["source_trust"] == 1.0
    # No LLM was needed
    assert isinstance(breakdown["final_weight"], float)


@pytest.mark.asyncio
async def test_describe_current_vibe_empty_returns_canned(engine):
    text = await engine.describe_current_vibe(domain=None)
    # No domain -> we return the canned empty-pool text without hitting LLM
    assert "No vibe signals" in text
