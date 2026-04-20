"""Unit tests for shared integrity-error detection."""

from __future__ import annotations

from skynet_capture.common.errors import looks_like_integrity_error


class UniqueViolationError(Exception):
    pass


class IntegrityError(Exception):
    pass


class _Sqlstate:
    def __init__(self, code: str) -> None:
        self.sqlstate = code


def test_class_name_unique_violation():
    assert looks_like_integrity_error(UniqueViolationError("dup"))


def test_class_name_integrity_error():
    assert looks_like_integrity_error(IntegrityError("dup"))


def test_sqlstate_23505():
    exc = _Sqlstate("23505")
    assert looks_like_integrity_error(exc)  # type: ignore[arg-type]


def test_sqlstate_23000_family():
    exc = _Sqlstate("23000")
    assert looks_like_integrity_error(exc)  # type: ignore[arg-type]


def test_message_duplicate_key():
    assert looks_like_integrity_error(Exception("ERROR: duplicate key value violates unique constraint"))


def test_message_unique_violation():
    assert looks_like_integrity_error(Exception("unique constraint violates unique"))


def test_generic_error_not_matched():
    assert not looks_like_integrity_error(ValueError("something else"))
