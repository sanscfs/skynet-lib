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

Logical-time decay (Phase 8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both stores were originally Redis LISTs trimmed by FIFO (``ltrim`` to
the last N entries). The problem: a busy hour can flush a week of
historical context. Per ``feedback_decay_logical_time.md`` we now use
a **per-key logical clock** — every new sample increments a counter,
older samples receive an exponential decay factor
``0.5 ** (delta_ticks / half_life)`` when their percentile/median is
read. Storage is a Redis ZSET (score = logical tick). The trim policy
sizes the window in ticks, not entries, so traffic spikes don't erase
older context — they just shrink each older sample's weight.

The half-life is set in the function defaults
(``half_life_ticks=200``) — operator-tune-free, fixed at 10x the
cold-start sample threshold so a sample meaningfully contributes for
~1000 ticks worth of subsequent traffic.

Backward compatibility: a key that was previously a LIST is migrated
to a ZSET on first read (function ``_migrate_list_to_zset`` below).
The migration is lazy and one-shot per key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import median
from typing import Callable, Optional, Sequence

from .envelopes import WorkActuals, WorkEstimate

log = logging.getLogger("skynet_orchestration.calibration")

# ---------------------------------------------------------------------------
# Estimate baseline (k-NN history lookup)
# ---------------------------------------------------------------------------

CALIBRATION_KEY_PREFIX = "orchestration:calibration"

# Default logical-time half-life (in ticks). A sample contributes with
# weight 0.5 after this many subsequent ticks; effectively one
# half-life ≈ "one cold-start corpus" of 200 samples.
_DEFAULT_HALF_LIFE_TICKS = 200
# Hard ceiling on the ZSET size — prevents unbounded growth on long-
# running pods. Trim window sized in ticks (5 half-lives ≈ 3% weight
# remaining); older entries are pruned with ``zremrangebyscore``.
_DEFAULT_WINDOW_TICKS = 5 * _DEFAULT_HALF_LIFE_TICKS


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


def _tick_key(base: str) -> str:
    """Key holding the per-store logical clock (a Redis INTEGER)."""
    return f"{base}:tick"


def _next_tick(redis_client, base: str) -> int:
    """Increment-and-return the logical clock for ``base``.

    Falls back to ``0`` if Redis can't increment (it shouldn't, but
    callers prefer "best effort" over crashing the request path).
    """
    try:
        val = redis_client.incr(_tick_key(base))
        return int(val or 0)
    except Exception as exc:  # noqa: BLE001
        log.warning("logical-tick incr failed for %s: %s", base, exc)
        return 0


def _key_type(redis_client, key: str) -> str:
    """Wrap ``redis_client.type`` so test fakes that omit it don't crash."""
    try:
        t = redis_client.type(key)
        return t.decode() if isinstance(t, bytes) else (t or "")
    except Exception:
        return ""


def _migrate_list_to_zset(redis_client, key: str, base: str) -> None:
    """One-shot lazy migration: convert a legacy LIST at ``key`` to a ZSET.

    Called once per key on first read after the upgrade. Rebuilds the
    ZSET by treating each LIST entry as one logical tick (so order is
    preserved, oldest entries decay first as if they had been written
    over time). Best-effort — on failure we log and leave the LIST
    in place so the next call can retry.
    """
    try:
        raw = redis_client.lrange(key, 0, -1) or []
        if not raw:
            try:
                redis_client.delete(key)
            except Exception:
                pass
            return
        items: list[tuple[float, str]] = []
        for i, entry in enumerate(raw):
            text = entry.decode() if isinstance(entry, bytes) else entry
            items.append((float(i + 1), text))
        # Drop the LIST first so we can re-create as ZSET.
        try:
            redis_client.delete(key)
        except Exception:
            pass
        for score, member in items:
            try:
                redis_client.zadd(key, {member: score})
            except Exception:  # noqa: BLE001
                continue
        # Seed the tick counter so subsequent writes don't collide
        # with migrated scores.
        try:
            redis_client.set(_tick_key(base), str(int(items[-1][0])))
        except Exception:
            pass
        log.info("calibration: migrated LIST→ZSET for key=%s (n=%d)", key, len(items))
    except Exception as exc:  # noqa: BLE001
        log.warning("LIST→ZSET migration failed for key=%s: %s", key, exc)


def record_outcome(
    redis_client,
    target: str,
    invocation_id: str,
    query: str,
    actuals: WorkActuals,
    *,
    keep_last: int = 1000,  # back-compat; ignored when ZSET path is active
    half_life_ticks: int = _DEFAULT_HALF_LIFE_TICKS,
    window_ticks: Optional[int] = None,
) -> None:
    """Append one successful invocation's outcome to the history.

    Stored as a Redis ZSET keyed by a per-target logical clock. New
    samples get a fresh tick (``INCR``), old samples decay by
    ``0.5 ** (delta_ticks / half_life_ticks)`` when read. The window
    (``zremrangebyscore`` floor) is sized in ticks, not entries, so a
    traffic spike doesn't evict older history.

    The ``keep_last`` parameter is kept for API compatibility — the
    new ZSET path uses ``window_ticks`` (default = 5 * half_life)
    instead. Callers who relied on the old LIST trim semantics will
    see history retained longer; this is intentional.
    """
    base_key = f"{CALIBRATION_KEY_PREFIX}:history:{target}"
    tick = _next_tick(redis_client, base_key)
    window = window_ticks if window_ticks is not None else (5 * half_life_ticks)

    record = f"{invocation_id}\t{query}\t{actuals.tokens_used}\t{actuals.tool_calls_made}\t{actuals.time_ms}"

    # Lazy migration: if the key still holds a LIST from before the
    # logical-time change, convert it before adding the new sample.
    key_type = _key_type(redis_client, base_key)
    if key_type == "list":
        _migrate_list_to_zset(redis_client, base_key, base_key)

    try:
        # Disambiguate identical record bytes (same query + same actuals)
        # by appending the tick — ZSET members are unique strings, so
        # repeated queries would otherwise overwrite their own history.
        member = f"{record}\t#{tick}"
        redis_client.zadd(base_key, {member: float(tick)})
        # Trim by score floor — anything older than (tick - window)
        # contributes <0.03 weight at the default 200/1000 settings.
        redis_client.zremrangebyscore(
            base_key,
            "-inf",
            f"({tick - window}",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("zadd outcome failed for target=%s: %s", target, exc)


def load_history(
    redis_client,
    target: str,
    *,
    half_life_ticks: int = _DEFAULT_HALF_LIFE_TICKS,
) -> list[CalibrationRecord]:
    """Read back the history for one target.

    Returns the records in chronological order (oldest tick first).
    Decay weights are NOT applied here — the consumer
    (:func:`baseline_estimate`) treats history as a candidate pool and
    picks the k-nearest by similarity; weighting comes in at the
    threshold-snapshot level where percentile semantics matter.
    Migrates a legacy LIST to a ZSET on first read.
    """
    base_key = f"{CALIBRATION_KEY_PREFIX}:history:{target}"
    key_type = _key_type(redis_client, base_key)
    if key_type == "list":
        _migrate_list_to_zset(redis_client, base_key, base_key)
        key_type = "zset"

    raw: list = []
    if key_type == "zset" or not key_type:  # missing key returns ""
        try:
            raw = redis_client.zrange(base_key, 0, -1) or []
        except Exception as exc:  # noqa: BLE001
            log.warning("zrange failed for target=%s: %s", target, exc)
            raw = []

    out: list[CalibrationRecord] = []
    for entry in raw:
        text = entry.decode() if isinstance(entry, bytes) else entry
        # Strip the disambiguating tick suffix added in record_outcome.
        if "\t#" in text:
            text = text.rsplit("\t#", 1)[0]
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
    keep_last: int = 200,  # back-compat; ignored on the ZSET path
    half_life_ticks: int = _DEFAULT_HALF_LIFE_TICKS,
    window_ticks: Optional[int] = None,
) -> None:
    """Push a measurement onto the per-pair logical-time history.

    ZSET-backed: each new sample gets a fresh logical tick (per-key
    ``INCR``), the floor of the kept window is computed in ticks. Old
    measurements decay by ``0.5 ** (delta_ticks / half_life_ticks)``
    when read in :func:`threshold_snapshot`.

    The ``keep_last`` parameter is kept for compatibility with older
    call sites; under the new model the trim is by tick window so the
    distribution doesn't get truncated by traffic spikes.
    """
    base_key = f"{CALIBRATION_KEY_PREFIX}:thresh:{caller}:{target}:{metric}"
    tick = _next_tick(redis_client, base_key)
    window = window_ticks if window_ticks is not None else (5 * half_life_ticks)

    # Lazy migration of legacy LIST data.
    key_type = _key_type(redis_client, base_key)
    if key_type == "list":
        _migrate_threshold_list_to_zset(redis_client, base_key)

    # Disambiguate identical values (cosine = 1.0 happens often when
    # gates evaluate the same reason against the same target description)
    # by appending the tick — otherwise the ZSET deduplicates them.
    member = f"{value:.6f}#{tick}"
    try:
        redis_client.zadd(base_key, {member: float(tick)})
        redis_client.zremrangebyscore(
            base_key,
            "-inf",
            f"({tick - window}",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "zadd threshold sample failed for %s/%s/%s: %s",
            caller,
            target,
            metric,
            exc,
        )


def _migrate_threshold_list_to_zset(redis_client, key: str) -> None:
    """Lazy LIST→ZSET migration for threshold metrics.

    Each LIST entry is a plain ``"%.6f"`` value; we re-add it under a
    synthetic tick (i+1) so the relative ordering survives, then seed
    the per-key tick counter at the new high watermark.
    """
    try:
        raw = redis_client.lrange(key, 0, -1) or []
        if not raw:
            try:
                redis_client.delete(key)
            except Exception:
                pass
            return
        try:
            redis_client.delete(key)
        except Exception:
            pass
        for i, entry in enumerate(raw, start=1):
            text = entry.decode() if isinstance(entry, bytes) else entry
            try:
                v = float(text)
            except ValueError:
                continue
            member = f"{v:.6f}#{i}"
            try:
                redis_client.zadd(key, {member: float(i)})
            except Exception:
                continue
        try:
            redis_client.set(_tick_key(key), str(len(raw)))
        except Exception:
            pass
        log.info(
            "calibration: migrated threshold LIST→ZSET for key=%s (n=%d)",
            key,
            len(raw),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("threshold LIST→ZSET migration failed for %s: %s", key, exc)


def _decay_weight(delta_ticks: float, *, half_life_ticks: float) -> float:
    """Return the exponential decay weight ``0.5 ** (delta / half_life)``.

    Bounded to ``[0.0, 1.0]`` so a clock skew (delta < 0 because two
    pods raced the INCR) doesn't produce a > 1 weight.
    """
    if half_life_ticks <= 0:
        return 1.0
    if delta_ticks <= 0:
        return 1.0
    # 2 ** (-x) == 0.5 ** x; use the natural form to keep precision.
    return 2.0 ** (-(delta_ticks / half_life_ticks))


def _weighted_percentile(
    pairs: list[tuple[float, float]],
    p: float,
) -> float:
    """Return the ``p``-th percentile of the weighted samples.

    ``pairs`` is ``[(value, weight), ...]`` sorted by value ascending.
    The implementation mirrors numpy's ``percentile(..., method="linear")``
    on a discrete weighted CDF — pick the first value whose cumulative
    weight crosses ``p * total_weight``.
    """
    if not pairs:
        return 0.0
    total = sum(w for _, w in pairs)
    if total <= 0.0:
        # All samples decayed past the precision floor; fall back to
        # the unweighted middle.
        return pairs[len(pairs) // 2][0]
    target = p * total
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= target:
            return v
    return pairs[-1][0]


def threshold_snapshot(
    redis_client,
    *,
    caller: str,
    target: str,
    metric: str,
    min_samples: int = 20,
    half_life_ticks: int = _DEFAULT_HALF_LIFE_TICKS,
) -> Optional[ThresholdSnapshot]:
    """Return a percentile snapshot or ``None`` if too few samples.

    Percentiles are computed over a logical-time-decayed weighting:
    each sample contributes ``0.5 ** (delta_ticks / half_life_ticks)``
    where ``delta_ticks`` is the gap between the sample's tick and the
    current per-key tick. Sample size for the threshold guard is the
    *raw* count (decay only affects the weighted percentile, never the
    "do we have enough data?" decision).
    """
    base_key = f"{CALIBRATION_KEY_PREFIX}:thresh:{caller}:{target}:{metric}"
    key_type = _key_type(redis_client, base_key)
    if key_type == "list":
        _migrate_threshold_list_to_zset(redis_client, base_key)
        key_type = "zset"

    raw: list = []
    if key_type == "zset" or not key_type:
        try:
            # withscores=True so we have ``[(member, score), ...]``.
            raw = redis_client.zrange(base_key, 0, -1, withscores=True) or []
        except Exception as exc:  # noqa: BLE001
            log.warning("zrange threshold failed for %s: %s", base_key, exc)
            raw = []

    if len(raw) < min_samples:
        return None

    # Current tick = highest score in the set. Reading INCR would race
    # with concurrent writes; the ZSET high watermark is canonical.
    cur_tick = max(float(score) for _, score in raw) if raw else 0.0

    weighted: list[tuple[float, float]] = []
    for member, score in raw:
        text = member.decode() if isinstance(member, bytes) else member
        # Strip the disambiguating tick suffix.
        val_str = text.split("#", 1)[0] if "#" in text else text
        try:
            v = float(val_str)
        except ValueError:
            continue
        delta = cur_tick - float(score)
        w = _decay_weight(delta, half_life_ticks=float(half_life_ticks))
        weighted.append((v, w))

    if len(weighted) < min_samples:
        return None

    weighted.sort(key=lambda x: x[0])

    return ThresholdSnapshot(
        p25=_weighted_percentile(weighted, 0.25),
        p50=_weighted_percentile(weighted, 0.50),
        p75=_weighted_percentile(weighted, 0.75),
        sample_size=len(weighted),
    )
