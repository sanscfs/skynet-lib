"""Prometheus instrumentation for the orchestration library.

Optional dep: ``prometheus-client`` (in the ``[server]`` extra).

The library does not require Prometheus at runtime — every helper
here degrades to a no-op if the import fails. That keeps the
unit-test environment dependency-free and lets services that don't
expose ``/metrics`` (yet) ship the same library version.

When the dep IS available the four metrics line up 1:1 with the
panel queries in
``infra-apps/grafana/dashboards/dashboard-orchestration.json``:

* ``skynet_orchestration_invocations_total{caller, target, purpose, status}``
  — Counter incremented once on every ``AgentServer.handle`` exit.
* ``skynet_orchestration_invocation_duration_seconds{caller, target, purpose}``
  — Histogram observing handler duration in seconds.
* ``skynet_orchestration_gate_rejections_total{caller, target, gate}``
  — Counter incremented on each gate rejection.
* ``skynet_orchestration_budget_exhausted_total{caller, target}``
  — Counter incremented when ``status == budget_exhausted``.

Why polarity-as-suffix on a single Counter instead of separate
``_accept`` / ``_reject`` metrics: keeps the dashboard PromQL simple
(``sum by (status)``) without forcing every consumer to know which
suffix variant to record.
"""

from __future__ import annotations

import logging

log = logging.getLogger("skynet_orchestration.metrics")

try:  # pragma: no cover -- the import path is exercised in CI but
    # not in the dependency-light unit-test runs.
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )

    _PROM_AVAILABLE = True

    INVOCATIONS_TOTAL = Counter(
        "skynet_orchestration_invocations_total",
        "Total agent-to-agent invocations handled by an AgentServer.",
        ["caller", "target", "purpose", "status"],
    )

    INVOCATION_DURATION_SECONDS = Histogram(
        "skynet_orchestration_invocation_duration_seconds",
        "Wall-clock duration of an AgentServer.handle() call.",
        ["caller", "target", "purpose"],
        # Buckets shaped for typical handler latencies: a sub-agent
        # ranges from <100ms (cached recommend) to ~5min (deep SRE
        # diagnostic loops).
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
    )

    GATE_REJECTIONS_TOTAL = Counter(
        "skynet_orchestration_gate_rejections_total",
        "Calls rejected at one of the four gates (cycle / repeat / justification / convergence) or token check.",
        ["caller", "target", "gate"],
    )

    BUDGET_EXHAUSTED_TOTAL = Counter(
        "skynet_orchestration_budget_exhausted_total",
        "Calls whose handler returned status=budget_exhausted.",
        ["caller", "target"],
    )

except Exception as _exc:  # noqa: BLE001
    _PROM_AVAILABLE = False
    INVOCATIONS_TOTAL = None  # type: ignore[assignment]
    INVOCATION_DURATION_SECONDS = None  # type: ignore[assignment]
    GATE_REJECTIONS_TOTAL = None  # type: ignore[assignment]
    BUDGET_EXHAUSTED_TOTAL = None  # type: ignore[assignment]
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"  # type: ignore[assignment]

    def generate_latest() -> bytes:  # type: ignore[no-redef]
        return b""

    log.debug("prometheus-client not installed; metrics are no-ops: %s", _exc)


# ---------------------------------------------------------------------------
# Helpers — every function tolerates the no-op state.
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Return whether prometheus-client is importable in this process."""
    return _PROM_AVAILABLE


def record_invocation(
    *,
    caller: str,
    target: str,
    purpose: str,
    status: str,
    duration_seconds: float,
) -> None:
    """Stamp one completed AgentServer.handle() exit.

    Increments the invocation counter, observes the duration histogram,
    and (when applicable) bumps the budget_exhausted counter. Safe to
    call even when prometheus-client isn't installed — the function
    is then a no-op. Each label is coerced to a non-empty string so
    Prometheus doesn't trip over ``None``.
    """
    if not _PROM_AVAILABLE:
        return
    caller = caller or "unknown"
    target = target or "unknown"
    purpose = purpose or "unknown"
    status = status or "unknown"
    try:
        INVOCATIONS_TOTAL.labels(  # type: ignore[union-attr]
            caller=caller, target=target, purpose=purpose, status=status
        ).inc()
        INVOCATION_DURATION_SECONDS.labels(  # type: ignore[union-attr]
            caller=caller, target=target, purpose=purpose
        ).observe(max(0.0, duration_seconds))
        if status == "budget_exhausted":
            BUDGET_EXHAUSTED_TOTAL.labels(  # type: ignore[union-attr]
                caller=caller, target=target
            ).inc()
    except Exception as exc:  # noqa: BLE001
        log.debug("metrics record_invocation failed: %s", exc)


def record_rejection(*, caller: str, target: str, gate: str) -> None:
    """Stamp one gate rejection (cycle / repeat / justification /
    convergence / token).

    Same no-op-on-missing-dep contract as :func:`record_invocation`.
    """
    if not _PROM_AVAILABLE:
        return
    try:
        GATE_REJECTIONS_TOTAL.labels(  # type: ignore[union-attr]
            caller=caller or "unknown",
            target=target or "unknown",
            gate=gate or "unknown",
        ).inc()
    except Exception as exc:  # noqa: BLE001
        log.debug("metrics record_rejection failed: %s", exc)


def latest_text() -> tuple[bytes, str]:
    """Return ``(text_format_bytes, content_type)`` for ``GET /metrics``.

    When prometheus-client is missing the body is empty and the content
    type is the standard plain-text — the caller can still respond 200
    so the FastAPI router doesn't 500 on ``/metrics`` scraping.
    """
    if not _PROM_AVAILABLE:
        return b"# prometheus-client not installed\n", "text/plain; charset=utf-8"
    try:
        return generate_latest(), CONTENT_TYPE_LATEST
    except Exception as exc:  # noqa: BLE001
        log.debug("metrics generate_latest failed: %s", exc)
        return b"", "text/plain; charset=utf-8"


__all__ = [
    "BUDGET_EXHAUSTED_TOTAL",
    "GATE_REJECTIONS_TOTAL",
    "INVOCATIONS_TOTAL",
    "INVOCATION_DURATION_SECONDS",
    "is_available",
    "latest_text",
    "record_invocation",
    "record_rejection",
]
