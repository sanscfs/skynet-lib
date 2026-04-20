"""Integrity-error detection shared across capture modules.

Avoids importing asyncpg in test paths — a fake pool can raise any exception
with a matching class name or ``sqlstate`` attribute and it will be recognised.
"""

from __future__ import annotations


def looks_like_integrity_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is a DB uniqueness / integrity violation.

    Matches asyncpg ``UniqueViolationError`` (SQLSTATE 23505), generic
    ``IntegrityError`` from DB-API 2 drivers, and plain-string messages.
    """
    name = type(exc).__name__
    if name in (
        "UniqueViolationError",
        "IntegrityError",
        "IntegrityConstraintViolationError",
    ):
        return True
    sqlstate = getattr(exc, "sqlstate", "") or ""
    if sqlstate.startswith("23"):
        return True
    msg = str(exc).lower()
    if "duplicate key" in msg:
        return True
    if "unique" in msg and "violat" in msg:
        return True
    return False
