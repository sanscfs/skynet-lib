"""Standardized logging setup for Skynet components."""

from __future__ import annotations

import logging


def setup_logging(
    service_name: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> logging.Logger:
    """Configure root logging and return a named logger.

    Call once at module/application startup. Returns a logger named
    after the service so callers can immediately do:

        log = setup_logging("skynet-agent")
        log.info("started")
    """
    logging.basicConfig(level=level, format=fmt, force=True)
    return logging.getLogger(service_name)
