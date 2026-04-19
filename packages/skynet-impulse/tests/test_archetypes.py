"""Thompson-sampling bandit behavior + persistence."""

from __future__ import annotations

import random

import pytest
from skynet_impulse.archetypes import (
    Archetype,
    ArchetypeBandit,
    default_archetypes,
)


def test_archetype_name_is_colonized():
    a = Archetype("novelty", "playful", "short")
    assert a.name == "novelty:playful:short"


def test_bandit_rejects_empty_init():
    with pytest.raises(ValueError):
        ArchetypeBandit([])


def test_bandit_sample_respects_trigger_filter():
    arches = [
        Archetype("novelty", "curious", "short"),
        Archetype("repeat", "direct", "short"),
        Archetype("*", "reflective", "medium"),
    ]
    b = ArchetypeBandit(arches, rng=random.Random(42))
    # Force heavy bias toward wildcard + novelty via repeated failures on "repeat".
    for _ in range(50):
        b.update(arches[1], 0.0)
    chosen = b.sample(trigger_kind="novelty")
    # Never returns the "repeat" archetype when filtering to "novelty".
    assert chosen.trigger_kind in ("novelty", "*")


def test_bandit_filter_falls_back_when_no_match():
    arches = [Archetype("novelty", "curious", "short")]
    b = ArchetypeBandit(arches, rng=random.Random(1))
    # No archetype matches "unknown" kind -> fall back to full set instead of raising.
    chosen = b.sample(trigger_kind="unknown")
    assert chosen == arches[0]


def test_bandit_converges_to_best_arm():
    """Thompson sampling should concentrate on the best arm after many rewards."""
    arches = [
        Archetype("novelty", "good", "short"),
        Archetype("novelty", "bad", "short"),
    ]
    b = ArchetypeBandit(arches, rng=random.Random(7))
    # Simulate 500 trials: "good" rewards 70%, "bad" rewards 10%.
    rng = random.Random(13)
    for _ in range(500):
        chosen = b.sample(trigger_kind="novelty")
        if chosen.name.endswith("good:short"):
            reward = 1.0 if rng.random() < 0.7 else 0.0
        else:
            reward = 1.0 if rng.random() < 0.1 else 0.0
        b.update(chosen, reward)
    good_mean = b.posterior_mean(arches[0])
    bad_mean = b.posterior_mean(arches[1])
    assert good_mean > bad_mean, (good_mean, bad_mean)
    # Good arm should be sampled far more often than bad -- count fractional rewards.
    good_pulls = b.state()["archetypes"][0]["s"] + b.state()["archetypes"][0]["f"]
    bad_pulls = b.state()["archetypes"][1]["s"] + b.state()["archetypes"][1]["f"]
    assert good_pulls > bad_pulls * 2


def test_bandit_reward_clamping():
    a = Archetype("novelty", "x", "short")
    b = ArchetypeBandit([a], rng=random.Random(0))
    b.update(a, 5.0)  # out of range, should clamp to 1.0
    b.update(a, -2.0)  # clamp to 0.0
    state = b.state()
    # After one success + one failure plus priors, s = 1 + 1 = 2, f = 1 + 1 = 2
    assert state["archetypes"][0]["s"] == pytest.approx(2.0)
    assert state["archetypes"][0]["f"] == pytest.approx(2.0)


def test_bandit_roundtrip_state():
    a = Archetype("novelty", "curious", "short")
    b = Archetype("novelty", "playful", "short")
    bandit = ArchetypeBandit([a, b], rng=random.Random(3))
    bandit.update(a, 1.0)
    bandit.update(a, 1.0)
    bandit.update(b, 0.0)
    state = bandit.state()
    restored = ArchetypeBandit.restore(state, rng=random.Random(3))
    assert restored.state() == state


def test_default_archetypes_cross_product():
    a = default_archetypes()
    assert len(a) == 3 * 4 * 2


def test_bandit_new_archetype_after_init_seeded_with_prior():
    b = ArchetypeBandit([Archetype("a", "b", "c")])
    new = Archetype("x", "y", "z")
    b.update(new, 0.5)
    state = b.state()
    names = [x["trigger_kind"] + ":" + x["tone"] + ":" + x["length"] for x in state["archetypes"]]
    assert "x:y:z" in names
