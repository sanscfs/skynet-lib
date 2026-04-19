# skynet-vibe

Universal gradient signal system for preference capture and retrieval across
all Skynet services. One schema, one store, one engine -- for chat, emoji
reactions, wiki notes, consumption logs, Airflow DAGs, and any future source
of preference signal.

## Design philosophy

### Gradient, not binary
No `like`, `dislike`, `rating`, or `sentiment` fields. A signal is a vector in
embedding space; distance and direction carry the meaning. This aligns with
Skynet's broader memory-consolidation and decay philosophy (continuous
compression, logical-time decay) rather than tiered labels.

### Domain-as-prototype
`VibeSignal` deliberately has no `domain` field. Domain membership is
computed at retrieval time as cosine similarity against named prototype
centroids. A single signal about "minimalist atmosphere" can legitimately
contribute to `music`, `movies`, and `photography` recommendations
simultaneously with naturally different weights. New domains (or
sub-domains like `slow-cinema`, `electronic-music`) are added by defining
a new prototype -- zero retagging of existing signals.

### Facet vectors
`FacetVectors` bundles:

* `content` (required): embedding of the signal text itself.
* `context` (optional): embedding of the surrounding conversation / wiki
  page / batch that produced the signal.
* `user_state` (optional): embedding of the user's mood / time / location
  at capture time.

At retrieval, `content` is compared to the domain centroid for domain
affinity and optionally blended with the current context vector for a
"how well does this fit *now*" boost.

### Gradient weighting (run-time)
```
signal_weight = confidence
              * source_trust[source.type]
              * time_decay(timestamp, now, half_life_days)
              * max(0, cosine(content, prototype_centroid))
              * (1 + alpha * cosine(content, context_vector))
```

Every term gracefully degrades when inputs are absent -- no prototype gives
a 1.0 prototype term, no context gives a 1.0 context term. Negative cosines
against the prototype are clipped to zero so anti-vibe signals don't pull
recommendations backwards.

## Quick start

```python
from skynet_vibe import (
    PrototypeRegistry,
    Source,
    VibeEngine,
    VibeStore,
)
from skynet_qdrant import AsyncQdrantClient
from skynet_embedding import async_embed

# 1. Build dependencies
qdrant = AsyncQdrantClient(url="http://qdrant.qdrant.svc:6333")
store = VibeStore(qdrant, collection="user_profile_raw", sub_category="vibe_signal")

async def embedder(text: str) -> list[float]:
    return await async_embed(text)

async def llm(prompt: str) -> str:
    return await my_openrouter_client.complete(prompt)

prototypes = PrototypeRegistry(embedder)
await prototypes.load_defaults()  # seeds 11 starter domains

engine = VibeEngine(
    store=store,
    prototypes=prototypes,
    embedder=embedder,
    llm_client=llm,
    decay_half_life_days=45.0,
)

# 2. Absorb a chat signal
await engine.absorb(
    text="this album is exactly my evening mood",
    source=Source(type="chat", room_id="!skynet-music"),
    confidence=0.9,
    linked_rec_id="rec_20260419_142",
    context_text="raining outside, winding down",
)

# 3. Absorb an emoji reaction
await engine.absorb_emoji(
    emoji="🔥",
    source=Source(type="reaction", room_id="!skynet-music"),
    linked_rec_id="rec_20260419_142",
)

# 4. Recommend
result = await engine.suggest(
    candidates=[
        {"id": "album_a", "title": "Album A", "description": "...", "vector": [...]},
        {"id": "album_b", "title": "Album B", "description": "...", "text": "mellow piano"},
        # ...
    ],
    domain="music",
    context_text="rainy evening",
    top_k=5,
)
print(result.candidate, result.reason, result.vibe_summary)

# 5. Describe the current vibe
summary = await engine.describe_current_vibe(domain="music", window_days=14)

# 6. Diagnose a single signal's weight
breakdown = await engine.explain_signal(
    signal_id="abc-123", domain="music", context_text="rainy evening"
)
```

## Extension patterns

### Adding a new domain
Append to `config/default_prototypes.yaml` or call at runtime:

```python
await prototypes.add("workout", [
    "running lifting cardio fitness physical",
    "training reps sets pace stamina",
])
```

Existing signals immediately participate in the new domain via cosine --
no migration step.

### Adding a new source type
Source types are free-form strings. Add the new type + a trust value to
`SOURCE_TRUST` in `affinity.py` (or monkey-patch at boot) and start
emitting `Source(type="your_new_source", ...)`. The engine will pick it
up with no other code changes.

### Alternative storage
`VibeStore` depends only on a duck-typed Qdrant-like object
(`upsert` / `search` / `get_point` / `set_payload`). Pass any
implementation that exposes those methods; `skynet_qdrant.AsyncQdrantClient`
is the default but unit tests use an in-memory fake, and you can wire in
Weaviate / Milvus / Postgres + pgvector by writing a thin adapter.

## Relationship to `skynet-core.signals`

`skynet-core.signals` is an **event bus** (Redis streams, producer/consumer,
ephemeral impulses like `novelty`, `trait_drift`, `concern`). This package
is a **persistent memory store** for preference/opinion/vibe signals that
accumulate and feed retrieval. They overlap at the edges (a `trait_drift`
impulse might be worth persisting as a vibe signal) but serve different
purposes and should not be collapsed into one container.

## Default prototypes

The bundled `config/default_prototypes.yaml` defines 11 starter domains:

* `music`, `movies`, `books`, `food`, `travel`, `photography`
* `slow-cinema` (sub-domain of movies -- peer, not child)
* `electronic-music` (sub-domain of music -- peer)
* `workout`, `work-focus`, `evening-wind-down` (context/state domains)

Extend or override via `await registry.add(...)` at boot time.

## Storage layout

* Default Qdrant collection: `user_profile_raw` (existing Skynet convention;
  override in `VibeStore.__init__`).
* Payload field `category: "vibe_signal"` (configurable sub-category) keeps
  these records separate from legacy `cinema_preferences` /
  `music_preferences` / etc.
* Primary `vector` = content facet (so existing search tooling works).
* Context + user_state facets stored as payload arrays
  (`vector_context`, `vector_user_state`).

## Testing

```
uv run pytest packages/skynet-vibe/tests -v
```

36 tests, fully offline (deterministic hash embedder + JSON-producing fake LLM).
