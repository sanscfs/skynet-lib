"""OpenTelemetry tracing setup -- no-op if OTEL_EXPORTER_OTLP_ENDPOINT is unset.

Consolidates the identical _setup_tracing() boilerplate from all 12 Skynet
components into a single function.
"""

from __future__ import annotations

import logging
import os

from opentelemetry import trace

logger = logging.getLogger(__name__)

# Map of instrumentor names to (module_path, class_name)
_INSTRUMENTORS = {
    "fastapi": ("opentelemetry.instrumentation.fastapi", "FastAPIInstrumentor"),
    "httpx": ("opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
    "requests": ("opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
    "redis": ("opentelemetry.instrumentation.redis", "RedisInstrumentor"),
}

_initialized = False


def setup_tracing(
    service_name: str,
    *,
    instrumentors: list[str] | None = None,
    endpoint: str | None = None,
) -> None:
    """Initialize OpenTelemetry tracing with OTLP gRPC exporter.

    Args:
        service_name: The OTEL service name (e.g. "skynet-agent").
        instrumentors: List of libraries to auto-instrument.
            Supported: "fastapi", "httpx", "requests", "redis".
            If None, instruments nothing (manual spans only).
        endpoint: OTLP endpoint override. Defaults to OTEL_EXPORTER_OTLP_ENDPOINT env var.
    """
    global _initialized
    if _initialized:
        return

    endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        return

    try:
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        resource = Resource.create({SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _initialized = True
    except Exception:
        logger.debug("Failed to initialize OTel tracing", exc_info=True)
        return

    for name in instrumentors or []:
        entry = _INSTRUMENTORS.get(name)
        if not entry:
            logger.debug("Unknown instrumentor: %s", name)
            continue
        try:
            mod = __import__(entry[0], fromlist=[entry[1]])
            getattr(mod, entry[1])().instrument()
        except Exception:
            logger.debug("Failed to instrument %s", name, exc_info=True)


def get_tracer(name: str) -> trace.Tracer:
    """Return a tracer instance. Works whether or not tracing was initialized."""
    return trace.get_tracer(name)
