"""Skynet Scoring -- memory decay, PageRank, and consolidation math.

Single source of truth for all scoring-related formulas used across
Skynet services. Previously duplicated inline in skynet-profile-synthesis
(scoring.py) and skynet-identity (main.py inline DECAY_STRIKE_PENALTY).
Centralising here ensures a constant change in one place propagates to
every consumer after a single rebuild.

Public API is re-exported from submodules so callers can keep their
imports short:

    from skynet_scoring import (
        # Carrying-capacity + PageRank (legacy, calendar-based)
        CARRYING_CAPACITY, TAU_BASE_DAYS, N_REF, DECAY_STRIKE_PENALTY,
        effective_halflife, retrieval_penalty,
        compute_direct_importance, compute_total_score,
        compute_pagerank, compute_graph_pagerank,
        dynamic_min_cluster_size, dynamic_similarity_threshold,
        SOURCE_WEIGHTS, source_weight_for,
        DEFAULT_SCORING_FIELDS, missing_scoring_fields,

        # Logical-time decay (Phase 1, new)
        LAMBDA_RAW, LAMBDA_EPISODIC, LAMBDA_TRAIT, LAMBDA_IDENTITY,
        DEFAULT_LAMBDAS, SALIENCE_MODULATION_STRENGTH,
        MEMORY_CLASSES, classify_memory,
        compute_decay_factor, compute_decay_factor_logical,
        compute_decay_factor_calendar,
    )
"""

from skynet_scoring.classify import (
    MEMORY_CLASSES,
    NEUTRAL_SALIENCE,
    classify_memory,
    default_salience_for,
)
from skynet_scoring.decay import (
    DEFAULT_LAMBDAS,
    LAMBDA_EPISODIC,
    LAMBDA_IDENTITY,
    LAMBDA_RAW,
    LAMBDA_TRAIT,
    SALIENCE_MODULATION_STRENGTH,
    compute_decay_factor,
    compute_decay_factor_calendar,
    compute_decay_factor_logical,
)
from skynet_scoring.scoring import (
    CARRYING_CAPACITY,
    DECAY_STRIKE_PENALTY,
    DEFAULT_SCORING_FIELDS,
    N_REF,
    SOURCE_WEIGHTS,
    TAU_BASE_DAYS,
    compute_direct_importance,
    compute_graph_pagerank,
    compute_pagerank,
    compute_total_score,
    dynamic_min_cluster_size,
    dynamic_similarity_threshold,
    effective_halflife,
    missing_scoring_fields,
    retrieval_penalty,
    source_weight_for,
)

__all__ = [
    # Legacy / calendar-time scoring
    "CARRYING_CAPACITY",
    "TAU_BASE_DAYS",
    "N_REF",
    "DECAY_STRIKE_PENALTY",
    "SOURCE_WEIGHTS",
    "DEFAULT_SCORING_FIELDS",
    "source_weight_for",
    "effective_halflife",
    "retrieval_penalty",
    "compute_direct_importance",
    "compute_total_score",
    "compute_pagerank",
    "compute_graph_pagerank",
    "missing_scoring_fields",
    "dynamic_min_cluster_size",
    "dynamic_similarity_threshold",
    # Logical-time decay (Phase 1)
    "LAMBDA_RAW",
    "LAMBDA_EPISODIC",
    "LAMBDA_TRAIT",
    "LAMBDA_IDENTITY",
    "DEFAULT_LAMBDAS",
    "SALIENCE_MODULATION_STRENGTH",
    "compute_decay_factor",
    "compute_decay_factor_logical",
    "compute_decay_factor_calendar",
    # Memory classification + salience
    "MEMORY_CLASSES",
    "NEUTRAL_SALIENCE",
    "classify_memory",
    "default_salience_for",
]
