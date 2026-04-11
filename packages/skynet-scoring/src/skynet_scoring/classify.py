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

This module also carries the salience heuristic (default_salience_for)
which mirrors the same precedence pattern: explicit payload field >
source-based base > confirmation/contradiction modifiers > neutral 0.5.
Salience is an orthogonal signal from memory_class; they modulate the
effective lambda together (see decay.compute_decay_factor_logical).
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


# --- Salience heuristics -----------------------------------------------
#
# Source-based base salience. Sources not listed here get the NEUTRAL
# default below. Ordering of lookups: exact match first, then prefix
# match (for families like `phone_app_*`).
#
# The rationale for each band:
#   0.85-0.95  authored by the user directly (wiki, feedback,
#              correction) or ground-truth identity markers
#   0.60-0.75  LLM-synthesised traits, direct chat observations
#              that were explicitly saved
#   0.40-0.55  work signals derived from chat/git/claude_code -- useful
#              but not universally salient
#   0.15-0.30  infrastructure metrics, phone telemetry, k8s events --
#              noise-prone by default
#
# All values are capped to [0, 1] after modifier application.

NEUTRAL_SALIENCE: Final[float] = 0.5

_BASE_SALIENCE_BY_SOURCE: Final[dict[str, float]] = {
    # High trust — ground truth about the user
    "wiki:entities/user.md": 0.95,
    "identity": 0.90,
    "wiki": 0.85,
    "feedback": 0.85,
    "correction": 0.85,
    "user_correction": 0.85,
    # Consolidated / synthesised knowledge
    "consolidation": 0.70,
    "trait": 0.70,
    "profile-synthesis": 0.70,
    # Direct chat observations
    "chat": 0.60,
    "skynet_chat": 0.60,
    "skynet-chat": 0.60,
    "direct_chat": 0.60,
    # Work signals
    "claude_code": 0.55,
    "claude_sessions": 0.55,
    "knowledge": 0.55,
    "kb": 0.55,
    # Imports / derived observations
    "google_takeout_gemini": 0.50,
    "git": 0.40,
    "health": 0.40,
    "gadgetbridge": 0.40,
    # Low-signal telemetry -- decay these fast
    "k8s": 0.25,
    "infrastructure": 0.25,
    "observation": 0.25,
    "session": 0.20,
    "working": 0.20,
    "phone_app_activity": 0.15,
    "app_usage": 0.15,
}

# Prefix-based fallbacks for source family namespacing.
_BASE_SALIENCE_BY_PREFIX: Final[tuple[tuple[str, float], ...]] = (
    ("wiki:", 0.80),  # any wiki path not entities/user.md
    ("phone_", 0.15),
    ("app_", 0.15),
    ("k8s_", 0.25),
    ("infra_", 0.25),
    ("skynet_", 0.50),  # skynet_episodic, skynet_knowledge, etc.
)

# Modifiers applied additively on top of the base salience. Signs
# encode the direction (confirmed = upward, contradicted/decay = down).
_CONFIRMED_BOOST: Final[float] = 0.10
_CONTRADICTED_PENALTY: Final[float] = 0.10
_DECAY_STRIKE_PENALTY: Final[float] = 0.15
# Threshold at which confirmations start boosting (below, we don't
# trust the count enough to lift the baseline).
_CONFIRMED_MIN: Final[int] = 3


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


def _base_salience_for_source(source: str) -> float:
    """Return the base (pre-modifier) salience for a source string.

    Exact match beats prefix match; unknown sources fall through to
    NEUTRAL_SALIENCE. `source` is expected lowercased -- callers should
    normalise before passing in.
    """
    if not source:
        return NEUTRAL_SALIENCE
    if source in _BASE_SALIENCE_BY_SOURCE:
        return _BASE_SALIENCE_BY_SOURCE[source]
    for prefix, val in _BASE_SALIENCE_BY_PREFIX:
        if source.startswith(prefix):
            return val
    return NEUTRAL_SALIENCE


def default_salience_for(payload: dict) -> float:
    """Heuristic salience for a payload lacking explicit salience.

    Returns a float in [0, 1]. Precedence:
      1. Explicit `salience` field on payload (clamped to [0, 1])
      2. Source-based base (exact then prefix match)
      3. Modifiers from confirmation / contradiction / strike counters
      4. NEUTRAL_SALIENCE default

    Used by decay.compute_decay_factor_logical when a point doesn't
    already carry `salience` in payload. Also called by the backfill
    DAG to seed `salience` on existing Qdrant points during the
    Phase 1 migration, so the formula MUST be pure and deterministic
    (no randomness, no external lookups).
    """
    if not isinstance(payload, dict):
        return NEUTRAL_SALIENCE

    explicit = payload.get("salience")
    if isinstance(explicit, (int, float)):
        return max(0.0, min(1.0, float(explicit)))

    source = str(payload.get("source", "")).lower()
    base = _base_salience_for_source(source)

    confirmed = int(payload.get("confirmed_count", 0) or 0)
    contradicted = int(payload.get("contradicted_count", 0) or 0)
    strikes = int(payload.get("decay_strikes", 0) or 0)

    modifier = 0.0
    if confirmed >= _CONFIRMED_MIN:
        modifier += _CONFIRMED_BOOST
    if contradicted > 0:
        modifier -= _CONTRADICTED_PENALTY
    if strikes > 0:
        modifier -= _DECAY_STRIKE_PENALTY

    result = base + modifier
    return max(0.0, min(1.0, result))
