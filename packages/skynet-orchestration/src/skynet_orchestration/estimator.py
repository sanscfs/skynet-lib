"""Self-estimation interface + structural fallback.

When a sub-agent receives a task, it estimates how much work it is
*before* doing the work. The estimate, scaled by a buffer, becomes
the BudgetGrant.

We deliberately avoid keyword-based heuristics ("query contains
'investigate' → bigger budget"). Per the project codestyle (see
``feedback_no_phrase_heuristics.md`` / ``feedback_adaptive_not_hardcoded.md``)
phrase lists silently encode operator guesses and rot. The two
allowed paths here:

1. **Historical k-NN** -- if calibration has enough samples for
   this target, return their median (see :mod:`calibration`).

2. **Structural fallback** -- when history is too thin, derive an
   estimate from *structural* features of the query that don't
   pretend to know the language: token count, presence of named
   entities (regex on uppercase/IDs/paths), presence of error
   codes / numeric IDs. No vocabulary list.

3. **Caller-supplied estimator** -- a sub-agent that has its own
   tiny LLM judge can plug in by implementing :class:`Estimator`.
   The protocol stays the same -- the server never cares which
   path produced the estimate.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional, Protocol, Sequence

from .calibration import CalibrationRecord, baseline_estimate
from .envelopes import WorkEstimate

log = logging.getLogger("skynet_orchestration.estimator")


class Estimator(Protocol):
    """Plug-in interface for any estimator implementation."""

    def estimate(self, query: str) -> WorkEstimate: ...


# ---------------------------------------------------------------------------
# Structural fallback
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"\S+")
# Crude entity probes: anything that *looks* identifier-shaped.
# We are not parsing language, we are counting structural anchors.
_ENTITY_PROBES: tuple[re.Pattern, ...] = (
    re.compile(r"\b[A-Z][A-Za-z0-9_-]{2,}\b"),  # CamelCase or PascalCase ids
    re.compile(r"\b[a-z][a-z0-9-]+(?:-[a-z0-9-]+){1,}\b"),  # kebab-case (pod-name-12)
    re.compile(r"/[\w.-/]+"),  # paths
    re.compile(r"\b[0-9a-fA-F]{7,}\b"),  # SHA-ish
    re.compile(r"\b\d{2,}\b"),  # numeric IDs / counts
)


def structural_features(query: str) -> dict[str, int]:
    """Count cheap structural signals. No vocabulary, no language model.

    Exposed as a building block so callers can mix it with their own
    estimators (e.g., feed these counts as features to a tiny model).
    """
    tokens = _TOKEN_RE.findall(query or "")
    entity_count = sum(len(p.findall(query or "")) for p in _ENTITY_PROBES)
    return {
        "token_count": len(tokens),
        "char_count": len(query or ""),
        "entity_anchors": entity_count,
    }


def structural_fallback(query: str, *, base_tokens: int = 800) -> WorkEstimate:
    """Estimate from structural features alone.

    The intuition we *can* honestly defend without keyword lists:
    longer queries with more entity anchors tend to require more
    tool work. We scale ``base_tokens`` linearly in those signals
    and call confidence low because this isn't really learned.
    """
    feats = structural_features(query)
    # Each entity anchor adds ~30% of base; each 50 tokens of query
    # adds another 50% of base. Bounded so a 5000-token query
    # doesn't request 50k tokens.
    multiplier = 1.0 + 0.30 * feats["entity_anchors"] + 0.01 * feats["token_count"]
    multiplier = min(multiplier, 6.0)
    tokens = int(base_tokens * multiplier)
    return WorkEstimate(
        tokens_needed=tokens,
        tool_calls_expected=max(1, feats["entity_anchors"] // 2 + 1),
        time_ms=tokens * 30,  # crude tokens→latency
        confidence=0.30,  # honest: this is a fallback, not a model
        complexity="unknown",
    )


# ---------------------------------------------------------------------------
# Composite estimator: history first, structural fallback, caller override
# ---------------------------------------------------------------------------


class CompositeEstimator:
    """Estimator that consults history, then structural fallback.

    Construction:
      :param history: list of past CalibrationRecord for this target
      :param similarity_fn: ``(str, str) -> float`` cosine over
          embeddings; same one the gates use
      :param caller_estimator: optional :class:`Estimator` produced
          by the agent itself (e.g., a tiny LLM judge). If supplied,
          its result is *blended* with the history baseline rather
          than overriding it -- prevents a buggy/gaming agent from
          single-handedly inflating its budget.
    """

    def __init__(
        self,
        *,
        similarity_fn: Callable[[str, str], float],
        history: Sequence[CalibrationRecord] = (),
        caller_estimator: Optional[Estimator] = None,
        history_weight: float = 0.7,
    ):
        self._sim = similarity_fn
        self._history = list(history)
        self._caller = caller_estimator
        self._history_weight = history_weight

    def estimate(self, query: str) -> WorkEstimate:
        baseline = baseline_estimate(query, self._history, similarity_fn=self._sim)
        caller_est: Optional[WorkEstimate] = None
        if self._caller is not None:
            try:
                caller_est = self._caller.estimate(query)
            except Exception as exc:  # noqa: BLE001
                # The LLM judge endpoint is allowed to be flaky; the
                # composite never explodes — we just drop its opinion.
                log.debug("caller estimator raised: %s", exc)
                caller_est = None
        if baseline is None:
            structural = structural_fallback(query)
            if caller_est is None:
                return structural
            return _blend(structural, caller_est, weight=0.5)
        if caller_est is None:
            return baseline
        return _blend(baseline, caller_est, weight=self._history_weight)


def _blend(a: WorkEstimate, b: WorkEstimate, *, weight: float) -> WorkEstimate:
    """Linear blend: ``weight * a + (1 - weight) * b``."""
    w = max(0.0, min(1.0, weight))
    return WorkEstimate(
        tokens_needed=int(w * a.tokens_needed + (1 - w) * b.tokens_needed),
        tool_calls_expected=int(w * a.tool_calls_expected + (1 - w) * b.tool_calls_expected),
        time_ms=int(w * a.time_ms + (1 - w) * b.time_ms),
        confidence=max(a.confidence, b.confidence),
        complexity=a.complexity if a.complexity != "unknown" else b.complexity,
    )


# ---------------------------------------------------------------------------
# LLM-judge estimator (Phase 8) — POSTs to a configurable HTTP endpoint
# ---------------------------------------------------------------------------


class LLMJudgeEstimator:
    """An :class:`Estimator` that delegates to an HTTP endpoint.

    The endpoint is expected to return a JSON object with the
    :class:`WorkEstimate` fields (``tokens_needed``,
    ``tool_calls_expected``, ``time_ms``, ``confidence``,
    ``complexity``). On any failure (network error, malformed JSON,
    missing fields) :meth:`estimate` returns ``None`` so the
    :class:`CompositeEstimator` upstream can fall back to history /
    structural fallback.

    The library DOES NOT bundle an HTTP client by default — callers
    pass an ``httpx.Client`` (or compatible) so the dependency stays
    optional and tests can plug a fake. The chosen separation also
    keeps the estimator usable from sync handlers (the AgentServer
    is sync); async callers can wrap their own ``httpx.AsyncClient``
    in a thin adapter that exposes ``.post(...).json()``.

    Construction::

        from skynet_orchestration.estimator import LLMJudgeEstimator
        import httpx
        est = LLMJudgeEstimator(
            url="http://skynet-cache.skynet-cache.svc:8080/estimate",
            target="music",
            http_client=httpx.Client(timeout=4.0),
        )
        composite = CompositeEstimator(
            similarity_fn=cosine,
            history=history,
            caller_estimator=est,
        )
    """

    def __init__(
        self,
        *,
        url: str,
        target: str,
        http_client: Any,
        timeout: float = 4.0,
    ):
        self._url = url
        self._target = target
        self._http = http_client
        self._timeout = timeout

    def estimate(self, query: str) -> Optional[WorkEstimate]:  # type: ignore[override]
        """Return a :class:`WorkEstimate` from the endpoint, or ``None``.

        ``None`` is the documented "I have no opinion" answer — the
        :class:`CompositeEstimator` will then ignore the LLM judge for
        this request and use whatever pure structural / historical
        path was available.
        """
        if not self._url or not self._http:
            return None
        try:
            resp = self._http.post(
                self._url,
                json={"target": self._target, "query": query or ""},
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("LLM judge endpoint unreachable: %s", exc)
            return None
        try:
            status_ok = getattr(resp, "status_code", 200) == 200
        except Exception:  # noqa: BLE001
            status_ok = True
        if not status_ok:
            log.debug(
                "LLM judge returned non-200 status=%s for target=%s",
                getattr(resp, "status_code", "?"),
                self._target,
            )
            return None
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            log.debug("LLM judge returned invalid JSON: %s", exc)
            return None
        return _coerce_work_estimate(data)


def _coerce_work_estimate(data: Any) -> Optional[WorkEstimate]:
    """Best-effort: build a WorkEstimate from a loose dict.

    Returns ``None`` when required numeric fields are absent or
    invalid — the composite estimator will then ignore the LLM
    judge for this request and fall back to the historical /
    structural baseline.
    """
    if not isinstance(data, dict):
        return None
    try:
        tokens = int(data.get("tokens_needed", 0))
        tool_calls = int(data.get("tool_calls_expected", 1))
        time_ms = int(data.get("time_ms", 0))
        # Out-of-band ``null`` is fine — defaults are conservative.
        confidence = float(data.get("confidence", 0.3))
        complexity = data.get("complexity") or "unknown"
    except (TypeError, ValueError):
        return None
    if tokens <= 0 or time_ms < 0:
        return None
    return WorkEstimate(
        tokens_needed=tokens,
        tool_calls_expected=max(1, tool_calls),
        time_ms=time_ms,
        confidence=max(0.0, min(1.0, confidence)),
        complexity=complexity if complexity in ("low", "medium", "high", "unknown") else "unknown",
    )


# ---------------------------------------------------------------------------
# Estimate → grant conversion (with confidence-scaled buffer)
# ---------------------------------------------------------------------------


def grant_from_estimate(
    estimate: WorkEstimate,
    *,
    base_buffer: float = 1.5,
    low_confidence_extra: float = 0.5,
    extensions_remaining: int = 3,
):
    """Convert a WorkEstimate to a BudgetGrant with a confidence-scaled buffer.

    Imported lazily to keep the module dependency-light for tests.
    """
    from .envelopes import BudgetGrant

    confidence_factor = 1.0 - estimate.confidence  # 1 when no clue, 0 when certain
    factor = base_buffer + low_confidence_extra * confidence_factor
    scaled = estimate.scaled(factor)
    return BudgetGrant(
        tokens=scaled.tokens_needed,
        tool_calls=scaled.tool_calls_expected,
        time_ms=scaled.time_ms,
        extensions_remaining=extensions_remaining,
    )
