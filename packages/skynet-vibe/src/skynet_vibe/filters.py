"""Qdrant filter helpers for the vibe signal pool.

The vibe "pool" in ``user_profile_raw`` is *heterogeneous* after the
2026-04-20 schema backfill:

* ~13k **retrofitted legacy records** (e.g. ``gemini_facts``,
  ``phone_telemetry``, ``git_history``, ``cinema_preferences``, ...)
  that kept their original ``category`` but gained
  ``signal_version=2`` + ``source_v2``.
* A handful of **new skynet-vibe writes** tagged ``category=vibe_signal``
  (that's what :class:`skynet_vibe.VibeStore` stamps on ``put()``).

Filtering by ``category == vibe_signal`` alone would see only the second
bucket (7 points at time of writing). Filtering by ``signal_version == 2``
alone would miss future writers that forget to stamp signal_version.

:func:`vibe_signal_filter` returns a **disjunctive** Qdrant filter that
accepts EITHER:

* ``signal_version >= 2`` (range match — spans the backfilled pool), OR
* ``category == "vibe_signal"`` (exact match — skynet-vibe writes).

This is the single source of truth for "is this payload vibe-compatible";
every consumer (VibeStore, profile-synthesis aggregator, skynet-movies /
skynet-music pool_stats fallback) imports this helper rather than
reconstructing the clause inline.
"""

from __future__ import annotations

from typing import Any

# Default sub-category that :class:`skynet_vibe.VibeStore.put` stamps on
# new records. Kept in sync with VibeStore's default constructor arg so
# callers that don't override ``sub_category`` get a clean match.
DEFAULT_VIBE_CATEGORY = "vibe_signal"


def vibe_signal_filter(
    extra: dict[str, Any] | None = None,
    *,
    vibe_category: str = DEFAULT_VIBE_CATEGORY,
) -> dict[str, Any]:
    """Return a Qdrant filter matching any vibe-signal-compatible record.

    The base filter is a ``should`` disjunction (Qdrant semantics:
    "match at least one"):

        signal_version >= 2   OR   category == vibe_category

    Parameters
    ----------
    extra:
        Optional extra filter to AND with the base disjunction. If it
        carries a ``must`` list, those clauses are AND'd in. Shortcut
        ``{key: value}`` pairs are also supported (treated as
        ``must`` match clauses). If ``extra`` contains its own
        ``should`` / ``must_not`` / ``should`` keys, they are preserved
        and composed alongside the base.
    vibe_category:
        Override the category name. Useful if a caller constructs a
        VibeStore with a non-default ``sub_category``.

    Returns
    -------
    A Qdrant filter dict. Examples::

        >>> vibe_signal_filter()
        {
            "should": [
                {"key": "signal_version", "range": {"gte": 2}},
                {"key": "category", "match": {"value": "vibe_signal"}},
            ]
        }

        >>> vibe_signal_filter({"must": [{"key": "timestamp",
        ...                               "range": {"gte": "2026-04-01"}}]})
        {
            "must": [
                {"key": "timestamp", "range": {"gte": "2026-04-01"}},
                {"should": [ ...base disjunction... ]},
            ]
        }
    """
    base_should: list[dict[str, Any]] = [
        {"key": "signal_version", "range": {"gte": 2}},
        {"key": "category", "match": {"value": vibe_category}},
    ]

    if not extra:
        return {"should": base_should}

    # Normalise ``extra`` into a ``must`` list. Shortcut ``{k: v}`` maps
    # to ``[{"key": k, "match": {"value": v}}]`` for each pair that is
    # not already a reserved filter key.
    must_clauses: list[dict[str, Any]] = []
    reserved = {"must", "should", "must_not"}
    if "must" in extra and isinstance(extra["must"], list):
        must_clauses.extend(extra["must"])
    for key, value in extra.items():
        if key in reserved:
            continue
        must_clauses.append({"key": key, "match": {"value": value}})

    # Nest the disjunction as a single clause inside ``must``. Qdrant
    # accepts a ``should`` object as a valid ``must`` member, forming
    # ``(extra clauses) AND (signal_version>=2 OR category==vibe_signal)``.
    must_clauses.append({"should": base_should})

    out: dict[str, Any] = {"must": must_clauses}
    # Forward any ``should`` / ``must_not`` the caller passed through,
    # side-by-side with the ``must`` we built. This lets advanced
    # callers compose additional disjunctions without losing the base.
    if "should" in extra and isinstance(extra["should"], list):
        # Merge caller's ``should`` OUTSIDE the base disjunction by
        # AND'ing it via a sub-filter in ``must`` -- Qdrant doesn't
        # merge two top-level ``should`` arrays semantically.
        out["must"].append({"should": list(extra["should"])})
    if "must_not" in extra and isinstance(extra["must_not"], list):
        out["must_not"] = list(extra["must_not"])
    return out


__all__ = ["DEFAULT_VIBE_CATEGORY", "vibe_signal_filter"]
