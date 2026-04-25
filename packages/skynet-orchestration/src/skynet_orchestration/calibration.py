"""History-driven calibration for estimates and adaptive thresholds.

Two responsibilities, both Redis-backed and self-tuning:

1. **Estimate baseline (k-NN over past invocations).** When a new
   AgentCall arrives, the server may want a baseline expectation
   for tokens / tool_calls / time before trusting the agent's
   self-estimate. We embed the query, look up the K nearest past
   *successful* invocations of the same target, and return the
   median actuals. Median (not mean) shrugs off the long-tailed
   outliers that diagnostic loops produce.

2. **Adaptive thresholds (per (caller, target) percentile).** The
   gates module enforces fixed comparisons; *which* fixed value to
   use is tuned here. The threshold for, e.g., justification cosine
   between music's reasons and SRE's tool description is calibrated
   from the observed cosines on past *accepted* invocations. We
   keep p25 (looseness) and p50 in Redis so the server can pick
   whichever side it wants.

Both pieces work without an LLM. The estimate baseline needs
embeddings (caller-injected), thresholds are pure float histograms.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Callable, Optional, Sequence

from .envelopes import WorkActuals, WorkEstimate

# ---------------------------------------------------------------------------
# Estimate baseline (k-NN history lookup)
# ---------------------------------------------------------------------------

CALIBRATION_KEY_PREFIX = "orchestration:calibration"


@dataclass(frozen=True)
class CalibrationRecord:
    """One past invocation's signature + outcome.

    Stored as a Redis HASH (one per record) and indexed by a sorted
    set per target. The query embedding is stored as a packed bytes
    blob; in tests the cosine_fn just gets raw text and computes its
    own similarity.
    """

    record_id: str  # `<target>:<invocation_id>`
    target: str
    query: str
    tokens_used: int
    tool_calls_made: int
    time_ms: int


def record_outcome(
    redis_client,
    target: str,
    invocation_id: str,
    query: str,
    actuals: WorkActuals,
    *,
    keep_last: int = 1000,
) -> None:
    """Append one successful invocation's outcome to the history.

    Stored as a Redis LIST trimmed to ``keep_last``. We use a list
    rather than a sorted set because k-NN here is implicit: the
    caller embeds + scores externally, we just supply candidates.
    """
    key = f"{CALIBRATION_KEY_PREFIX}:history:{target}"
    record = f"{invocation_id}\t{query}\t{actuals.tokens_used}\t{actuals.tool_calls_made}\t{actuals.time_ms}"
    pipe = redis_client.pipeline()
    pipe.rpush(key, record)
    pipe.ltrim(key, -keep_last, -1)
    pipe.execute()


def load_history(redis_client, target: str) -> list[CalibrationRecord]:
    """Read back the trimmed history for one target."""
    key = f"{CALIBRATION_KEY_PREFIX}:history:{target}"
    raw = redis_client.lrange(key, 0, -1) or []
    out: list[CalibrationRecord] = []
    for entry in raw:
        text = entry.decode() if isinstance(entry, bytes) else entry
        parts = text.split("\t")
        if len(parts) != 5:
            continue
        try:
            out.append(
                CalibrationRecord(
                    record_id=f"{target}:{parts[0]}",
                    target=target,
                    query=parts[1],
                    tokens_used=int(parts[2]),
                    tool_calls_made=int(parts[3]),
                    time_ms=int(parts[4]),
                )
            )
        except ValueError:
            continue
    return out


def baseline_estimate(
    query: str,
    history: Sequence[CalibrationRecord],
    *,
    similarity_fn: Callable[[str, str], float],
    k: int = 10,
    min_neighbours: int = 3,
) -> Optional[WorkEstimate]:
    """Build a WorkEstimate from the k nearest historical records.

    Returns ``None`` when there isn't enough history yet -- the
    server falls back to the agent's own estimate or to a heuristic
    in that case (cold-start handling lives in estimator.py).
    """
    if len(history) < min_neighbours:
        return None
    scored = sorted(
        ((similarity_fn(query, rec.query), rec) for rec in history),
        key=lambda x: -x[0],
    )[:k]
    if len(scored) < min_neighbours:
        return None
    tokens = int(median(r.tokens_used for _, r in scored))
    calls = int(median(r.tool_calls_made for _, r in scored))
    time_ms = int(median(r.time_ms for _, r in scored))
    confidence = scored[min_neighbours - 1][0]  # similarity of the k-th neighbour
    return WorkEstimate(
        tokens_needed=tokens,
        tool_calls_expected=calls,
        time_ms=time_ms,
        confidence=max(0.0, min(1.0, confidence)),
        complexity="unknown",
    )


# ---------------------------------------------------------------------------
# Adaptive thresholds (per-pair percentile snapshot)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdSnapshot:
    """p25/p50/p75 of a (caller, target, metric) measurement."""

    p25: float
    p50: float
    p75: float
    sample_size: int


def record_threshold_sample(
    redis_client,
    *,
    caller: str,
    target: str,
    metric: str,
    value: float,
    keep_last: int = 200,
) -> None:
    """Push a measurement onto the rolling window for that pair+metric."""
    key = f"{CALIBRATION_KEY_PREFIX}:thresh:{caller}:{target}:{metric}"
    pipe = redis_client.pipeline()
    pipe.rpush(key, f"{value:.6f}")
    pipe.ltrim(key, -keep_last, -1)
    pipe.execute()


def threshold_snapshot(
    redis_client,
    *,
    caller: str,
    target: str,
    metric: str,
    min_samples: int = 20,
) -> Optional[ThresholdSnapshot]:
    """Return a percentile snapshot or ``None`` if too few samples."""
    key = f"{CALIBRATION_KEY_PREFIX}:thresh:{caller}:{target}:{metric}"
    raw = redis_client.lrange(key, 0, -1) or []
    values: list[float] = []
    for entry in raw:
        text = entry.decode() if isinstance(entry, bytes) else entry
        try:
            values.append(float(text))
        except ValueError:
            continue
    if len(values) < min_samples:
        return None
    values.sort()
    n = len(values)

    def _pct(p: float) -> float:
        idx = max(0, min(n - 1, int(p * (n - 1))))
        return values[idx]

    return ThresholdSnapshot(
        p25=_pct(0.25),
        p50=_pct(0.50),
        p75=_pct(0.75),
        sample_size=n,
    )
