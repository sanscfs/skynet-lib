# skynet-impulse

Domain-agnostic autonomous-impulse engine. Extracted and generalised from the
original `skynet_agent.modules.impulse` package so music / movies / main-chat
agents can all run curiosity loops on the same signal bus
(`skynet_core.impulses`) without sharing state or speaking over each other.

## What it does

Each tick:

1. Drain pending `Signal`s from the domain's Redis consumer group.
2. Apply the configured signal -> drive wirings, then decay.
3. Compare the dominant drive against a rolling p75 baseline of recent
   dominant values (with cold-start fallback + epsilon-greedy exploration).
4. Check per-anchor refractory, rate limit, and staleness.
5. If still eligible, ask a cheap LLM gate ("should I fire?"). On yes,
   sample an `Archetype` from a Thompson-sampling bandit over
   `(trigger_kind, tone, length)`.
6. Compose the message via a larger LLM.
7. Return a `TickResult` -- the caller posts the message, emits the
   `spoke` self-feedback signal, and bumps the bandit reward asynchronously.

## Install

```bash
# Inside the skynet-lib workspace:
uv add skynet-impulse

# Optional extras:
uv add "skynet-impulse[scipy]"   # Poisson tail via scipy.stats
uv add "skynet-impulse[ml]"      # + sklearn for UncertaintySamplingDetector
```

## Minimal example (music domain)

```python
import redis.asyncio as redis_async
from openai import OpenAI
from skynet_core import impulses as bus
from skynet_impulse import (
    ImpulseEngine, EngineConfig, Drive, SignalToDrive,
    DefaultHttpGateClient, DefaultOpenAIComposeClient,
    ArchetypeBandit, default_archetypes,
)

# 1. Wire up drives.
config = EngineConfig(
    domain="music",
    drives=[
        Drive("boredom", decay_rate=0.9, growth_per_tick=0.08),
        Drive("curiosity", decay_rate=0.85),
        Drive("concern", decay_rate=0.80),
        Drive("need_to_share", decay_rate=0.90),
    ],
    signal_to_drive=[
        SignalToDrive("novelty", "curiosity", multiplier=0.30),
        SignalToDrive("novelty", "need_to_share", multiplier=0.15),
        SignalToDrive("concern", "concern", multiplier=0.40),
        SignalToDrive("resolution", "concern", multiplier=-0.30),
        SignalToDrive("spoke", "need_to_share", dampen_multiply=0.6),
    ],
    exclude_from_trigger=["boredom"],
    rate_limit_per_day=3,
    voice_hint="Пиши як музичний друг, без жаргону критика.",
)

# 2. Inject clients.
gate = DefaultHttpGateClient(voice_hint=config.voice_hint)
compose = DefaultOpenAIComposeClient(
    OpenAI(),
    model="mistralai/mistral-large-2512",
    voice_hint=config.voice_hint,
)
bandit = ArchetypeBandit(default_archetypes())

engine = ImpulseEngine(
    config,
    bus=bus,
    redis=redis_async.from_url("redis://redis:6379/0"),
    gate_llm=gate,
    compose_llm=compose,
    archetype_bandit=bandit,
)

# 3. Tick loop (run as background task).
await engine.start()
while True:
    result = await engine.tick()
    if result.fired:
        await matrix.send(room_id, result.message)
        # ... later, reward the bandit based on user reply ...
    await asyncio.sleep(config.tick_interval_seconds)
```

## Detectors (producers)

The engine consumes `Signal`s; the detectors produce them. Run these in your
collection DAGs / analyzer jobs and call `emit_signal` with the outputs:

### Centroid-distance novelty

```python
from skynet_impulse.detectors import CentroidNoveltyDetector

det = CentroidNoveltyDetector(
    centroid=user_profile_centroid,        # np.ndarray
    threshold_cos=0.4,
    anchor_prefix="music:artist:",
)
sigs = await det.detect(
    vector=new_artist_embedding,
    signal_id="Autechre",
    source="collectors",
)
for sig in sigs:
    emit_signal(sig.kind, sig.source, sig.salience,
                anchor=sig.anchor, payload=sig.payload)
```

### Poisson repeat-intensity

```python
from skynet_impulse.detectors import PoissonRepeatDetector

det = PoissonRepeatDetector(
    lambda_baseline=2.5,          # EMA of per-artist listens per day
    p_threshold=0.05,
    anchor_prefix="music:artist:",
)
sigs = await det.detect(
    anchor_name="Autechre",
    observed_count=11,            # today's listens
    historical_count=420,         # lifetime
)
```

### Uncertainty sampling

```python
from skynet_impulse.detectors import UncertaintySamplingDetector
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X_train, y_train)
det = UncertaintySamplingDetector(
    clf, uncertainty_threshold=0.6, anchor_prefix="music:artist:",
)
sigs = await det.detect(X=candidate_artists_X, signal_ids=candidate_names)
```

## Consumer-specific customisation

Each domain scopes its Redis keys and consumer group automatically via
`EngineConfig.domain`. Override prompts by passing custom
`system_template` / `user_template` / `voice_hint` to the gate/compose
clients. Override the cross-product of default archetypes with your own
list of `Archetype`s for exotic domains.

## Architecture notes

- Signal bus is `skynet_core.impulses` -- one stream, many consumer groups.
- Baseline / refractory / rate-limit state lives in Redis under
  `skynet:impulses:{domain}:*` so two engine replicas stay consistent.
- The engine never speaks directly. It returns a `TickResult`; your caller
  is responsible for delivery and for emitting `spoke` self-feedback.
- Bandit state is held in memory; persist via `state()` / `restore()`.

## Adaptive half-life calibration

As of `2026.4.21` the per-anchor refractory is **self-calibrating**. The
historical `EngineConfig.refractory_cap_ticks` magic number (default: 4
signal events) is deprecated — setting it emits a `DeprecationWarning`
and the value is ignored by the new cooldown path. The engine now
learns an effective half-life per domain from observed anchor-repeat
intervals on its signal stream.

### How it works

Every signal that arrives on the stream is fed through the calibrator:

1. **Observe**: if the signal's anchor was seen before, compute
   `gap = global_signal_count - last_seen[anchor]` (in signal events,
   not wall-clock time — logical-time decay).
2. **Update both EMAs, per observation**:
   - `h_fast = 0.3 * gap + 0.7 * h_fast` (effective window ~10 obs)
   - `h_slow = 0.05 * gap + 0.95 * h_slow` (effective window ~50 obs)
3. **Blend** them by their disagreement ratio:
   `weight_fast = min(|h_fast - h_slow| / max(h_slow, 1), 1)`
   so when the two EMAs disagree (regime change) the blend tracks
   `h_fast`, and when they agree (stable) the blend tracks `h_slow`.
4. **Decay penalties** for every active anchor using
   `factor = exp(-ln(2) / effective_half_life)`.
5. **Gate**: if the candidate anchor's penalty is ≥ `0.1`
   (`PENALTY_NOISE_FLOOR`) the engine skips the fire with
   `skip_reason = "penalty:<value>"`.

After a successful fire the anchor's penalty is set back to `1.0` and
decays over the next `~effective_half_life × 3.32` signals back below
the noise floor (one half-life halves the penalty; after ~3.3 half-
lives it's at 0.1).

### Persistence

State lives in three Redis hashes, scoped per domain:

| Key | Contents |
|-----|----------|
| `skynet:impulse:calibration:<domain>` | `h_fast`, `h_slow`, `global_signal_count`, `observations` |
| `skynet:impulse:last_seen:<domain>`   | `anchor -> signal_count` for gap computation |
| `skynet:impulse:penalties:<domain>`   | `anchor -> current penalty float` |

Saves happen every `PERSIST_EVERY_N_SIGNALS = 10` events and after
every fire, so a pod restart loses at most 10 signals worth of learning.
Persistence failures are logged and swallowed — they never block
signal processing.

### Safety clamps

The effective half-life is always clamped to
`[HALF_LIFE_MIN, HALF_LIFE_MAX] = [5, 500]`, so even a pathological
burst of gap=1 signals can't collapse the cooldown to zero, and a
single corrupt observation of gap=10000 can't inflate it beyond 500.

### Diagnostics

`engine.state().half_life` returns:

```python
{
    "h_fast": 27.5,
    "h_slow": 42.1,
    "effective": 35.8,
    "disagreement": 0.346,
    "observations": 184,
    "global_signal_count": 1923,
}
```

`engine.state().active_penalties` exposes the per-anchor penalty dict
for diagnostic UIs.

`engine.state().half_life_human()` returns a one-liner for
`/vibe/status` and `!vibe-status`:

```
half-life: 36 signals (fast=28, slow=42, blend=35%)
184 anchor-repeat observations accumulated
```

### Why no median, no batching?

A single low-alpha EMA is already a smoothing filter — there is no
additional value in layering a median on top. The two-speed design
gives us outlier detection for free: a single extreme gap spikes
`h_fast` briefly, `h_slow` barely moves, and the disagreement ratio
warns the operator that the recent activity pattern shifted. After a
handful of normal observations `h_fast` reconverges with `h_slow` and
the transient clears. This behaviour is tested in
`tests/test_calibration.py::test_single_outlier_transient_recovers`.

### Tuning

Don't. The constants
(`HALF_LIFE_ALPHA_FAST`, `HALF_LIFE_ALPHA_SLOW`,
`HALF_LIFE_MIN`, `HALF_LIFE_MAX`, `HALF_LIFE_PRIOR_FRACTION`,
`PENALTY_NOISE_FLOOR`, `PENALTY_EVICTION_FLOOR`,
`PERSIST_EVERY_N_SIGNALS`) are intentionally **not** exposed as env
vars. They are well-grounded ML-convention values; if you feel the
urge to knob them the right answer is almost always "gather more data
and let the EMA adapt" rather than "tune the EMA".

## Wire contract

`signals.py` re-exports from `skynet_core.impulses` -- do NOT fork the
`Signal` dataclass. Any divergence would corrupt the Redis stream schema.
