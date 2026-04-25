"""Shared budget pool semantics."""

from __future__ import annotations

from skynet_orchestration import budget
from skynet_orchestration.envelopes import BudgetGrant, WorkActuals


def _cap() -> budget.RootBudget:
    return budget.RootBudget(tokens=10_000, tool_calls=20, time_ms=120_000)


def test_init_root_idempotent(redis):
    """Re-initialising a root preserves the existing pool."""
    budget.init_root(redis, "root1", _cap())
    # Spend some.
    budget.try_reserve(redis, "root1", BudgetGrant(tokens=2000, tool_calls=4, time_ms=20_000))
    rem = budget.remaining(redis, "root1")
    assert rem.tokens == 8000
    # Re-init shouldn't reset.
    budget.init_root(redis, "root1", _cap())
    rem2 = budget.remaining(redis, "root1")
    assert rem2.tokens == 8000


def test_try_reserve_succeeds_within_pool(redis):
    budget.init_root(redis, "root1", _cap())
    ok = budget.try_reserve(redis, "root1", BudgetGrant(tokens=4000, tool_calls=10, time_ms=60_000))
    assert ok
    assert budget.remaining(redis, "root1").tokens == 6000


def test_try_reserve_fails_and_rolls_back_on_overdraw(redis):
    """If any field would go negative, the whole reservation is undone."""
    budget.init_root(redis, "root1", _cap())
    # Tokens fit but time_ms doesn't.
    ok = budget.try_reserve(redis, "root1", BudgetGrant(tokens=1000, tool_calls=2, time_ms=999_999))
    assert not ok
    rem = budget.remaining(redis, "root1")
    assert rem.tokens == 10_000
    assert rem.time_ms == 120_000


def test_refund_returns_unused(redis):
    """Refund credits unused budget back to the pool."""
    budget.init_root(redis, "root1", _cap())
    grant = BudgetGrant(tokens=4000, tool_calls=10, time_ms=60_000)
    budget.try_reserve(redis, "root1", grant)
    actuals = WorkActuals(tokens_used=1000, tool_calls_made=2, time_ms=10_000)
    budget.refund(redis, "root1", actuals=actuals, granted=grant)
    rem = budget.remaining(redis, "root1")
    # We reserved 4000 tokens but only used 1000 -- 3000 should come back.
    assert rem.tokens == 10_000 - 1000
    assert rem.tool_calls == 20 - 2
    assert rem.time_ms == 120_000 - 10_000


def test_refund_no_credit_when_overspent(redis):
    """If actuals exceed grant (extension scenario), refund is zero."""
    budget.init_root(redis, "root1", _cap())
    grant = BudgetGrant(tokens=1000, tool_calls=2, time_ms=10_000)
    budget.try_reserve(redis, "root1", grant)
    actuals = WorkActuals(tokens_used=2000, tool_calls_made=5, time_ms=20_000)
    budget.refund(redis, "root1", actuals=actuals, granted=grant)
    rem = budget.remaining(redis, "root1")
    # No refund -- pool stays at post-reservation level.
    assert rem.tokens == 10_000 - 1000


def test_grant_extension_capped(redis):
    """Per-tree extension counter prevents unbounded grow-on-demand."""
    budget.init_root(redis, "root1", _cap())
    # Tiny extensions -- we're testing the counter, not the math.
    additional = BudgetGrant(tokens=100, tool_calls=1, time_ms=1_000)
    granted = 0
    for _ in range(15):
        if budget.grant_extension(redis, "root1", additional, max_extensions_per_tree=5):
            granted += 1
    assert granted == 5
