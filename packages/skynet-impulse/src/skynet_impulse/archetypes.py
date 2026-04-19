"""Thompson-sampling bandit over (trigger_kind, tone, length) archetypes.

New in skynet-impulse -- not in the original agent module. Implements §7 of
the research doc: coarse archetype space (trigger_kind x tone x length) with
a Beta(s, f) posterior per archetype and Thompson sampling at speak time.

Math recap:

- Every archetype carries two pseudo-counts: ``s`` (successes) and ``f``
  (failures). Both start at ``prior_alpha`` / ``prior_beta`` so the posterior
  is a proper distribution from day one.
- At selection, we draw ``score ~ Beta(s+1, f+1)`` per archetype and pick
  the argmax. High-variance (under-sampled) archetypes win frequently early
  via wide posteriors -- natural exploration.
- On reward r in [0, 1]: ``s += r; f += 1 - r``. Fractional rewards are
  supported so a "user replied within 2h with vaguely positive sentiment"
  can contribute 0.7 rather than a hard 0/1.

Filtering by ``trigger_kind`` is done at sample-time: callers pass the
current trigger (``"novelty"`` / ``"repeat"`` / ``"uncertain"`` / etc.) and
only archetypes with matching ``trigger_kind`` (or ``"*"`` wildcard) are
considered. This prevents the bandit from picking a "repeat-intensity"
archetype on a "novelty" trigger.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Iterable, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Archetype:
    """One question-shape the bot can take.

    - ``trigger_kind``: which detector / drive elicits this archetype.
      Use ``"*"`` for archetypes that can follow any trigger.
    - ``tone``: free-text label the compose prompt consumes verbatim.
    - ``length``: ``"short" | "medium" | "long"``; consumed by compose to
      choose a token budget.

    ``name`` is derived from the three fields and doubles as the bandit
    posterior's key in Redis. Keep components short -- they end up in
    Redis key paths.
    """

    trigger_kind: str
    tone: str
    length: str

    @property
    def name(self) -> str:
        return f"{self.trigger_kind}:{self.tone}:{self.length}"


@dataclass
class _PosteriorCounts:
    s: float  # success pseudo-count
    f: float  # failure pseudo-count


class ArchetypeBandit:
    """Beta-Bernoulli Thompson-sampling bandit over a fixed archetype set.

    State (the posterior counts) is held in-memory. Consumers persist it
    by calling ``state()`` and reloading via ``restore()``; the library
    stays storage-agnostic so the same bandit works from Redis, Postgres,
    or a JSON file on disk.
    """

    def __init__(
        self,
        archetypes: Iterable[Archetype],
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        rng: random.Random | None = None,
    ):
        self._archetypes = list(archetypes)
        if not self._archetypes:
            raise ValueError("ArchetypeBandit needs at least one archetype")
        # Seed independent Beta(α, β) for each archetype.
        self._counts: dict[str, _PosteriorCounts] = {
            a.name: _PosteriorCounts(prior_alpha, prior_beta) for a in self._archetypes
        }
        self._by_name = {a.name: a for a in self._archetypes}
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._rng = rng or random.Random()

    @property
    def archetypes(self) -> list[Archetype]:
        return list(self._archetypes)

    def _candidates(self, trigger_kind: str | None) -> list[Archetype]:
        """Archetypes matching ``trigger_kind`` (or any "*" catch-alls)."""
        if trigger_kind is None:
            return list(self._archetypes)
        return [a for a in self._archetypes if a.trigger_kind == trigger_kind or a.trigger_kind == "*"]

    def sample(
        self,
        context: dict | None = None,
        *,
        trigger_kind: str | None = None,
    ) -> Archetype:
        """Thompson draw. Returns the archetype with the largest sampled score.

        ``context`` is reserved for future contextual bandits; currently
        unused but part of the public contract so we don't break callers
        when we upgrade to LinUCB / Thompson-with-features later.

        If ``trigger_kind`` is provided, only archetypes with a matching
        ``trigger_kind`` (or ``"*"``) are considered. Falls back to the
        full set when no archetype matches (so a new trigger doesn't
        hard-crash the bot).
        """
        candidates = self._candidates(trigger_kind) or list(self._archetypes)
        best: Optional[Archetype] = None
        best_score = -math.inf
        for a in candidates:
            counts = self._counts[a.name]
            # Beta samples come from the underlying PRNG so tests can seed it.
            score = self._rng.betavariate(counts.s + 1, counts.f + 1)
            if score > best_score:
                best_score = score
                best = a
        assert best is not None
        return best

    def update(self, archetype: Archetype, reward: float) -> None:
        """Bayesian update from a ``reward`` in [0, 1]. Clamp silently."""
        r = max(0.0, min(1.0, float(reward)))
        counts = self._counts.get(archetype.name)
        if counts is None:
            # Archetype was added after init; seed with priors and continue.
            counts = _PosteriorCounts(self._prior_alpha, self._prior_beta)
            self._counts[archetype.name] = counts
            self._by_name[archetype.name] = archetype
            self._archetypes.append(archetype)
        counts.s += r
        counts.f += 1.0 - r

    def posterior_mean(self, archetype: Archetype) -> float:
        """E[p] under the current Beta posterior. Diagnostic only."""
        c = self._counts[archetype.name]
        denom = c.s + c.f
        return c.s / denom if denom > 0 else 0.5

    def state(self) -> dict:
        """Serialize to JSON-safe dict. Use with ``restore`` for persistence."""
        return {
            "prior_alpha": self._prior_alpha,
            "prior_beta": self._prior_beta,
            "archetypes": [
                {
                    "trigger_kind": a.trigger_kind,
                    "tone": a.tone,
                    "length": a.length,
                    "s": self._counts[a.name].s,
                    "f": self._counts[a.name].f,
                }
                for a in self._archetypes
            ],
        }

    @classmethod
    def restore(cls, state: dict, *, rng: random.Random | None = None) -> "ArchetypeBandit":
        archetypes = [Archetype(a["trigger_kind"], a["tone"], a["length"]) for a in state.get("archetypes", [])]
        bandit = cls(
            archetypes,
            prior_alpha=state.get("prior_alpha", 1.0),
            prior_beta=state.get("prior_beta", 1.0),
            rng=rng,
        )
        for a_dict, arch in zip(state.get("archetypes", []), archetypes):
            bandit._counts[arch.name] = _PosteriorCounts(
                a_dict.get("s", bandit._prior_alpha),
                a_dict.get("f", bandit._prior_beta),
            )
        return bandit


# --- Convenience factories -------------------------------------------------


def default_archetypes(
    trigger_kinds: Iterable[str] = ("novelty", "repeat", "uncertain"),
    tones: Iterable[str] = ("curious", "playful", "direct", "casual"),
    lengths: Iterable[str] = ("short", "medium"),
) -> list[Archetype]:
    """Cross-product ``(trigger_kind x tone x length)`` archetypes.

    Defaults give 3 * 4 * 2 = 24 archetypes -- enough room for the bandit to
    differentiate but small enough to converge within a few hundred ticks.
    Research doc warns that 90 is too many at ~3 msg/day so we keep "long"
    out of the default set.
    """
    return [Archetype(trigger, tone, length) for trigger in trigger_kinds for tone in tones for length in lengths]


__all__ = ["Archetype", "ArchetypeBandit", "default_archetypes"]
