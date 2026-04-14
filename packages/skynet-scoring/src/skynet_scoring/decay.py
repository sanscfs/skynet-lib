"""
decay.py -- logical-time exponential decay for memory scoring.

Phase 1 of docs/rag-memory-roadmap.md. The legacy scoring path
(scoring.py) multiplies freshness by `exp(-days / tau_eff)` where
`days` is wall-clock calendar time. That penalises the user for being
silent -- a memory that nobody queried for a month decays simply
because the sun kept rising.

This module introduces logical-time decay:

    recency = exp(-lambda * t_logical)

where `t_logical` is one of:

  * missed_opportunities (raw/episodic classes): the count of
    retrieval queries that matched this point semantically
    (cos > threshold) but didn't include it in top-K. A fact "ages"
    only when the system actually had a chance to use it.

  * ingestion_pressure (semantic/trait classes): the count of new
    points stored in the same cluster since last confirmation. A
    concept becomes stale only when newer knowledge about it arrives.

  * identity class: `lambda = 0`. Never decays except via explicit
    rewrite through the Phase 6 wiki-edit proposal flow.

Silence freezes the clock -- the user can go on holiday for a month
and nothing becomes obsolete. See feedback_decay_logical_time memory
for the full rationale.

All constants are env-tunable so the operator can retune lambdas
without a redeploy (identity/profile-synthesis read them at import
time, so a rolling restart picks up the new values).
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# --- Lambda constants (logical-time decay rates) -----------------------
#
# Half-life in the logical-time unit is ln(2) / lambda. The defaults
# below give:
#
#   raw       -> half-life ~69 missed opportunities
#   episodic  -> half-life ~139 missed opportunities
#   trait     -> half-life ~347 missed opportunities
#   identity  -> immortal
#
# Tune per service via env vars LAMBDA_RAW / LAMBDA_EPISODIC / ...
# Pass an explicit `lambdas` dict to override at call time.

LAMBDA_RAW = float(os.getenv("LAMBDA_RAW", "0.01"))
LAMBDA_EPISODIC = float(os.getenv("LAMBDA_EPISODIC", "0.005"))
LAMBDA_TRAIT = float(os.getenv("LAMBDA_TRAIT", "0.002"))
LAMBDA_IDENTITY = float(os.getenv("LAMBDA_IDENTITY", "0.0"))

# Memory classes not explicitly listed fall back to LAMBDA_RAW.
DEFAULT_LAMBDAS: dict[str, float] = {
    "identity": LAMBDA_IDENTITY,
    "trait": LAMBDA_TRAIT,
    "semantic": LAMBDA_TRAIT,  # same cohort as traits
    "episodic": LAMBDA_EPISODIC,
    "raw": LAMBDA_RAW,
    "observation": LAMBDA_RAW,  # same cohort as raw
    "working": float(os.getenv("LAMBDA_WORKING", "0.05")),
}

# How aggressively salience protects a memory from decay. Effective
# lambda becomes `lambda * (1 - salience * SALIENCE_MODULATION_STRENGTH)`.
# At strength 0.8 the salience knob spans a ~3x half-life range:
#
#   salience  lam_eff / lam   half-life vs default
#   0.0       1.00            0.6x  (shorter)
#   0.5 def   0.60            1.0x
#   0.9       0.28            2.15x (longer)
#   1.0       0.20            3.00x (longer)
#
# Half-life ratio = lam_eff(default) / lam_eff(target). The full span
# from salience=0 to salience=1 is 5x but the useful range (default 0.5
# upward) gives at most ~3x. Do not advertise "5x" externally.
SALIENCE_MODULATION_STRENGTH = float(os.getenv("SALIENCE_MODULATION_STRENGTH", "0.8"))

# Phase 9 — gradient compression dampens decay. Symmetric to the
# Phase 5 storage rule that high compression_level requires more
# pressure to budge: high compression_level should also decay slower
# in retrieval. A summary that compressed N raws is more "knowledge"
# than "event" and deserves a longer retention curve.
#
#   lam_eff = lam * 1 / (1 + compression_level * COMPRESSION_DAMPENING)
#
# At dampening 0.4:
#   level 0  → lam_eff / lam = 1.00 (raw, full decay)
#   level 1  → 0.71 (~40% longer half-life)
#   level 2  → 0.56 (~80% longer)
#   level 3  → 0.45 (~120% longer)
#
# This stacks multiplicatively with salience modulation: a salient
# level-2 summary decays slower than a non-salient level-2, which
# decays slower than a level-0 raw of equal salience.
COMPRESSION_DAMPENING = float(os.getenv("COMPRESSION_DAMPENING", "0.4"))

# Cosine threshold above which a retrieval "match" counts toward
# missed_opportunities if the point didn't make the final top-K.
# Lower -> more events count as misses -> faster decay of points
# that get retrieved-but-ignored.
MISSED_OPP_COS_THRESHOLD = float(os.getenv("MISSED_OPP_COS_THRESHOLD", "0.4"))


def _resolve_lambda(
    memory_class: str,
    lambdas: Optional[dict[str, float]] = None,
) -> float:
    """Return the base lambda for a memory class, honouring overrides."""
    src = lambdas if lambdas is not None else DEFAULT_LAMBDAS
    if memory_class in src:
        return src[memory_class]
    return src.get("raw", LAMBDA_RAW)


def _effective_lambda(lam: float, salience: float, compression_level: int = 0) -> float:
    """Modulate the base lambda by per-point salience AND compression level.

    Two independent multiplicative dampens stack here:
      1. Salience: explicit per-point importance flag → up to ~3x
         longer half-life when salience=1.
      2. Compression level: higher-level summaries decay slower
         (Phase 9 coupling — symmetric to the Phase 5 storage rule
         that higher levels are also harder to recompress).

    Clamps salience to [0, 1] and treats negative compression_level
    as 0 so a corrupt payload can't invert any sign.
    """
    if SALIENCE_MODULATION_STRENGTH > 0:
        s = max(0.0, min(1.0, float(salience)))
        lam = lam * (1.0 - s * SALIENCE_MODULATION_STRENGTH)
    if COMPRESSION_DAMPENING > 0:
        try:
            level = max(0, int(compression_level))
        except (TypeError, ValueError):
            level = 0
        if level > 0:
            lam = lam / (1.0 + level * COMPRESSION_DAMPENING)
    return lam


def compute_decay_factor_logical(
    payload: dict,
    *,
    memory_class: Optional[str] = None,
    lambdas: Optional[dict[str, float]] = None,
) -> float:
    """Return exp(-lambda_eff * missed_opportunities) in [0, 1].

    Reads `missed_opportunities` (int, default 0) from the payload.
    Salience is resolved via `default_salience_for` -- explicit
    `salience` on the payload wins, otherwise a source-based heuristic
    picks a base value (wiki/feedback high, phone_app_activity/k8s low)
    with additive modifiers for confirmation/contradiction/strike
    counters. See classify.default_salience_for for the full table.

    When `memory_class` is omitted we delegate to `classify_memory`.
    The result is ready to multiply onto a raw vector similarity score.

    Silence-safe: if the system hasn't performed any retrievals, the
    missed_opportunities counter stays at zero and the decay factor
    is exactly 1.0 regardless of memory_class, salience, or lambda.
    """
    from skynet_scoring.classify import classify_memory, default_salience_for

    if not isinstance(payload, dict):
        return 1.0

    if memory_class is None:
        memory_class = classify_memory(payload)

    lam = _resolve_lambda(memory_class, lambdas)
    if lam <= 0:
        return 1.0

    salience = default_salience_for(payload)
    compression_level = payload.get("compression_level") or 0
    lam_eff = _effective_lambda(lam, salience, compression_level)
    if lam_eff <= 0:
        return 1.0

    t = float(payload.get("missed_opportunities", 0) or 0)
    if t <= 0:
        return 1.0

    return math.exp(-lam_eff * t)


def compute_decay_factor_calendar(
    payload: dict,
    *,
    now: Optional[datetime] = None,
    tau_days: Optional[float] = None,
) -> float:
    """Legacy calendar-day decay extracted as a standalone function.

    Returns exp(-days_since_last_access / tau). When `tau_days` is
    None we read TAU_BASE_DAYS from the scoring module so both paths
    agree on the base constant.

    Exists only so consumers can A/B-test the old behaviour against
    the new logical-time path via the `time_basis` arg of
    `compute_decay_factor` below. No new code should call this
    directly.
    """
    from skynet_scoring.scoring import TAU_BASE_DAYS, _parse_iso

    if not isinstance(payload, dict):
        return 0.5

    last = payload.get("last_accessed") or payload.get("timestamp")
    dt = _parse_iso(last)
    if dt is None:
        return 0.5

    if now is None:
        now = datetime.now(timezone.utc)

    days = max(0.0, (now - dt).total_seconds() / 86400.0)
    tau = tau_days if tau_days is not None else TAU_BASE_DAYS
    if tau <= 0:
        return 1.0
    return math.exp(-days / tau)


def compute_decay_factor(
    payload: dict,
    *,
    time_basis: str = "logical",
    memory_class: Optional[str] = None,
    lambdas: Optional[dict[str, float]] = None,
    now: Optional[datetime] = None,
    tau_days: Optional[float] = None,
) -> float:
    """Unified decay factor dispatch.

    `time_basis` selects which clock is used:
      * "logical"  -- exp(-lambda * missed_opportunities)  [default]
      * "calendar" -- exp(-days / tau)                      [legacy]

    All other kwargs forward to the respective underlying function.
    Services gate the default via a DECAY_ENABLED env flag in their
    own config (identity/profile-synthesis); this function just
    implements whatever policy the caller requested.
    """
    if time_basis == "logical":
        return compute_decay_factor_logical(payload, memory_class=memory_class, lambdas=lambdas)
    if time_basis == "calendar":
        return compute_decay_factor_calendar(payload, now=now, tau_days=tau_days)
    raise ValueError(f"compute_decay_factor: unknown time_basis {time_basis!r} (expected 'logical' or 'calendar')")
