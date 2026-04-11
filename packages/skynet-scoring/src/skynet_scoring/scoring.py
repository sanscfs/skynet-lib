"""
scoring.py -- graph-based memory scoring (carrying-capacity + PageRank).

Ported verbatim from skynet-profile-synthesis/scoring.py during Phase 1
of docs/rag-memory-roadmap.md. The new logical-time decay functions
live in decay.py; consumers that want the Phase 1 behaviour should
call compute_decay_factor_logical instead of compute_direct_importance.

Implements a PageRank-inspired reinforcement / forgetting system:
  - direct importance (usage + freshness + source weight)
  - graph anchor score (PageRank over related_ids edges)
  - total score  = direct * 0.6 + graph * 0.4
  - e-based carrying-capacity pressure (new memories compress old ones)

Forgetting model (legacy / calendar-based — see decay.py for logical):
  New memories create consolidation pressure proportional to ln(N/N_0).
  The effective half-life tau_eff = tau_base * K / N, where K is the
  carrying capacity and N the current active memory count. Base e is
  the natural choice: dP/dN = 1/N -- each new memory's marginal
  pressure is inversely proportional to the total, giving the smoothest
  possible growth curve.

  Memories are NEVER hard-deleted. Instead they accumulate `decay_strikes`
  which penalise retrieval relevance by 0.8^strikes. A single access
  resets strikes to 0 ("resurrection").

All operations are pure Python -- the memory graph is small (<10k nodes),
so we deliberately avoid numpy/scipy to keep the container slim.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# --- Carrying-capacity constants ---

# Target active memory count — the "sweet spot" density where legacy
# calendar-time decay behaves at the base half-life.
CARRYING_CAPACITY = 10_000
# Base half-life in calendar days when N_active == CARRYING_CAPACITY.
# Legacy / calendar-based path only — logical-time decay in decay.py
# does NOT consume this constant.
TAU_BASE_DAYS = 90.0
# Reference density N_0 used by dynamic consolidation thresholds
# (ln(N/N_0) pressure shrinks min cluster size and similarity threshold
# as memory grows past the reference).
N_REF = 1000
# Multiplicative retrieval penalty per decay strike: 0.8^strikes.
# Independent signal from continuous recency decay — strikes are
# reactive (flagged by /prune), recency is proactive (time-based).
DECAY_STRIKE_PENALTY = 0.8


# --- Source weights ---

# Per-source trust multiplier applied to direct importance. Chat and
# claude_code observations are treated as full-trust; git / health
# observations are attenuated because they're derived signals.
SOURCE_WEIGHTS: dict[str, float] = {
    "chat": 1.0,
    "skynet_chat": 1.0,
    "claude_code": 1.0,
    "feedback": 1.0,
    "git": 0.7,
    "infrastructure": 0.3,
    "k8s": 0.3,
    "health": 0.5,
}


def source_weight_for(payload: dict) -> float:
    """Return the source weight for a point, falling back to 1.0."""
    explicit = payload.get("source_weight")
    if isinstance(explicit, (int, float)):
        return float(explicit)
    source = str(payload.get("source", "")).lower()
    for key, weight in SOURCE_WEIGHTS.items():
        if source.startswith(key):
            return weight
    return 1.0


# --- Time parsing ---


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp, returning None on any failure.

    Handles the common `...Z` suffix by normalising to `+00:00`. When the
    parsed datetime has no tzinfo we stamp it as UTC so downstream
    arithmetic with `datetime.now(timezone.utc)` stays consistent.
    """
    if not ts:
        return None
    if not isinstance(ts, str):
        return None
    try:
        # Python 3.11+ handles most ISO 8601 variants, including "Z".
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# --- Carrying-capacity helpers ---


def effective_halflife(n_active: int) -> float:
    """Effective half-life in days, inversely proportional to memory pressure.

    tau_eff = tau_base * K / N

    When N < K (plenty of room)  -> slower decay.
    When N = K (at capacity)     -> tau_base (90 days).
    When N > K (overcrowded)     -> faster decay, pushes consolidation.
    """
    return TAU_BASE_DAYS * CARRYING_CAPACITY / max(n_active, 1)


def retrieval_penalty(decay_strikes: int) -> float:
    """Multiplicative penalty on retrieval relevance: 0.8^strikes.

    0 strikes -> 1.0  (no penalty)
    1 strike  -> 0.8
    3 strikes -> 0.512
    5 strikes -> 0.328
    """
    if decay_strikes <= 0:
        return 1.0
    return DECAY_STRIKE_PENALTY**decay_strikes


# --- Direct importance ---


def compute_direct_importance(
    point: dict,
    now: datetime | None = None,
    n_active: int | None = None,
) -> float:
    """Usage + freshness + source weight => direct importance in [0, 1].

    `point` is a Qdrant point dict (`{"id": ..., "payload": {...}}`) or a
    raw payload dict. Handles both shapes gracefully.

    `n_active` is the current total active (non-archived) memory count.
    When provided, the half-life adapts via carrying-capacity pressure:
    tau_eff = tau_base * K / N.  When omitted, uses TAU_BASE_DAYS directly.

    NOTE: this function is the LEGACY calendar-time path and is kept for
    backwards compatibility. For logical-time decay (Phase 1 of the RAG
    roadmap) use `compute_decay_factor_logical` from the decay module.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    payload = point.get("payload") if "payload" in point else point
    if not isinstance(payload, dict):
        return 0.0

    base = float(payload.get("confidence", payload.get("importance", 0.5)) or 0.5)
    access = int(payload.get("access_count", 0) or 0)
    usage_boost = min(1.0 + access * 0.05, 1.5)

    last = payload.get("last_accessed") or payload.get("timestamp")
    dt = _parse_iso(last)
    if dt is not None:
        days = max(0.0, (now - dt).total_seconds() / 86400.0)
        tau = effective_halflife(n_active) if n_active else TAU_BASE_DAYS
        recency = math.exp(-days / tau)
    else:
        recency = 0.5

    sw = source_weight_for(payload)
    return max(0.0, min(base * usage_boost * recency * sw, 1.0))


# --- PageRank ---


def compute_pagerank(
    nodes: Iterable[str],
    edges: dict[str, list[str]],
    iterations: int = 20,
    damping: float = 0.85,
) -> dict[str, float]:
    """Pure-Python PageRank over a directed graph.

    Args:
        nodes: iterable of node ids (strings) -- defines the universe.
        edges: adjacency map {src -> [dst, ...]}. Edges to unknown nodes
               are ignored. Undirected `related_ids` should be passed as
               reciprocal edges by the caller.
        iterations: fixed number of power-iteration rounds.
        damping: classic PageRank damping factor.

    Returns:
        {node_id: score}, normalised so scores sum to ~1.
    """
    node_list = list({str(n) for n in nodes})
    n = len(node_list)
    if n == 0:
        return {}

    idx = {nid: i for i, nid in enumerate(node_list)}

    # Out-neighbours filtered to known nodes.
    out: list[list[int]] = [[] for _ in range(n)]
    for src, dsts in edges.items():
        si = idx.get(str(src))
        if si is None:
            continue
        for d in dsts or []:
            di = idx.get(str(d))
            if di is None or di == si:
                continue
            out[si].append(di)

    out_degree = [len(lst) for lst in out]

    # Initial uniform distribution.
    scores = [1.0 / n] * n
    base_rank = (1.0 - damping) / n

    for _ in range(max(1, iterations)):
        new_scores = [base_rank] * n
        dangling_mass = 0.0
        for i in range(n):
            if out_degree[i] == 0:
                dangling_mass += scores[i]
                continue
            share = damping * scores[i] / out_degree[i]
            for j in out[i]:
                new_scores[j] += share
        # Distribute dangling mass evenly (standard trick).
        if dangling_mass > 0.0:
            add = damping * dangling_mass / n
            new_scores = [s + add for s in new_scores]
        scores = new_scores

    # Normalise so that sum == 1 (robust against drift).
    total = sum(scores) or 1.0
    return {node_list[i]: scores[i] / total for i in range(n)}


def compute_graph_pagerank(points: list[dict], iterations: int = 20) -> dict[str, float]:
    """Build a graph from a list of Qdrant points and run PageRank.

    `related_ids` is treated as undirected: each edge (u, v) becomes
    both u->v and v->u so reciprocity doesn't require a second pass.
    """
    nodes: list[str] = []
    edges: dict[str, list[str]] = {}
    for p in points:
        pid = str(p.get("id", ""))
        if not pid:
            continue
        nodes.append(pid)
        payload = p.get("payload", {}) or {}
        related = payload.get("related_ids") or []
        clean = [str(r) for r in related if r and str(r) != pid]
        if clean:
            edges.setdefault(pid, []).extend(clean)
            for r in clean:
                edges.setdefault(r, []).append(pid)

    if not nodes:
        return {}
    return compute_pagerank(nodes, edges, iterations=iterations)


# --- Combined score ---


def compute_total_score(direct: float, graph: float) -> float:
    """Linear combination: 60% direct, 40% graph anchor."""
    return max(0.0, min(float(direct) * 0.6 + float(graph) * 0.4, 1.0))


# --- Default-value migration helper ---


DEFAULT_SCORING_FIELDS: dict[str, Any] = {
    "importance": 0.5,
    "graph_score": 0.0,
    "total_score": 0.5,
    "access_count": 0,
    "last_accessed": None,
    "source_weight": 1.0,
    "related_ids": [],
    "superseded_by": None,
    "archived": False,
    "version": 1,
    "decay_strikes": 0,
}


def missing_scoring_fields(payload: dict) -> dict[str, Any]:
    """Return the subset of DEFAULT_SCORING_FIELDS not yet present on `payload`."""
    out: dict[str, Any] = {}
    for k, v in DEFAULT_SCORING_FIELDS.items():
        if k not in payload:
            if k == "source_weight":
                out[k] = source_weight_for(payload)
            else:
                out[k] = v
    return out


# --- Dynamic consolidation thresholds ---


def dynamic_min_cluster_size(n_active: int) -> int:
    """Min similar memories to trigger consolidation, shrinks with ln(N/N_0).

    N=1000  -> 7  (gentle, few memories)
    N=5000  -> 3  (moderate pressure)
    N=10000 -> 2  (aggressive -- at carrying capacity)
    """
    if n_active <= N_REF:
        return 7
    return max(2, round(5 / math.log(n_active / N_REF + 1)))


def dynamic_similarity_threshold(n_active: int) -> float:
    """Cosine similarity threshold for clustering, lowers with ln(N/N_0).

    N=1000  -> 0.87  (strict, only near-duplicates)
    N=10000 -> 0.78  (looser, more consolidation)
    N=50000 -> 0.70  (floor)
    """
    base = 0.90
    if n_active <= N_REF:
        return base
    pressure = math.log(n_active / N_REF + 1) * 0.05
    return max(0.70, base - pressure)
