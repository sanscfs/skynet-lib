"""
classify.py -- memory-class inference for logical-time decay.

Each memory point lives in exactly one of MEMORY_CLASSES. The class
determines which decay rate lambda applies during retrieval scoring
(see decay.py). The classification is deliberately a small set of
heuristics rather than an LLM call -- it runs on every retrieval so
must be cheap and deterministic.

Precedence, highest first:
  1. Explicit `memory_class` payload field (authoritative)
  2. Explicit `memory_tier` field (synonym honoured for legacy)
  3. Source / category / tags heuristics
  4. Default: "raw"

The heuristics are intentionally forgiving -- any ambiguous point
falls through to "raw", which has the fastest decay, so
mis-classification penalises the wrong points conservatively rather
than immortalising noise.
"""

from __future__ import annotations

from typing import Final

# Canonical class ids. Matches the tier names used in the rag-memory
# roadmap Phase 5 consolidation pipeline. "working" represents
# session-scoped state that should decay very fast if it leaks into
# persistent storage at all.
MEMORY_CLASSES: Final[tuple[str, ...]] = (
    "identity",
    "trait",
    "semantic",
    "episodic",
    "raw",
    "working",
    "observation",
)


def classify_memory(payload: dict) -> str:
    """Return the decay class for a memory payload.

    Priority order:
      1. Explicit `memory_class` on the payload
      2. Explicit `memory_tier` on the payload (legacy synonym)
      3. Heuristics derived from `source`, `category`, `tags`
      4. Default to "raw"
    """
    if not isinstance(payload, dict):
        return "raw"

    explicit = payload.get("memory_class")
    if isinstance(explicit, str) and explicit in MEMORY_CLASSES:
        return explicit

    tier = payload.get("memory_tier")
    if isinstance(tier, str):
        tier_lc = tier.lower()
        if tier_lc in MEMORY_CLASSES:
            return tier_lc
        # Legacy tier synonyms
        if tier_lc == "trait-semantic":
            return "trait"

    source = str(payload.get("source", "")).lower()
    category = str(payload.get("category", "")).lower()
    tags = payload.get("tags") or []
    tag_set = {str(t).lower() for t in tags if t}

    # Identity tier -- ground-truth about the user, maintained in wiki.
    # These are rare in Qdrant today but the hook is here for Phase 6.
    # NOTE: generic `source=wiki` goes to semantic below — only the
    # specific entities/user.md path counts as identity.
    if "identity" in category or source in ("identity", "wiki:entities/user.md"):
        return "identity"
    if "identity" in tag_set:
        return "identity"

    # Trait tier -- synthesised summaries and consolidated abstractions.
    # Consolidation output carries `source=consolidation`.
    if source in ("consolidation", "trait", "profile-synthesis"):
        return "trait"
    if "trait" in category or category in ("consolidated", "summary"):
        return "trait"
    if "trait" in tag_set:
        return "trait"

    # Semantic tier -- conceptual knowledge, wiki excerpts, tool catalogs.
    if source in ("wiki", "knowledge", "kb") or "wiki" in source:
        return "semantic"
    if category in ("knowledge", "semantic", "concept", "tool"):
        return "semantic"

    # Episodic tier -- dated events and past actions.
    if source in ("episodic", "skynet_episodic", "action", "event"):
        return "episodic"
    if "episodic" in category or category in ("action", "event", "daily"):
        return "episodic"
    if "episodic" in tag_set:
        return "episodic"

    # Working-memory leaks -- short-lived session state that somehow got
    # persisted. Decay these aggressively.
    if source in ("session", "working") or category in ("session", "working"):
        return "working"

    # Anything explicitly tagged as observation. These are the rawest
    # signals and sit just below `raw` -- same decay rate but the tier
    # exists for the Phase 6 wiki-as-truth separation where observed
    # facts are split from believed facts.
    if source in ("observation",) or category == "observation":
        return "observation"

    return "raw"
