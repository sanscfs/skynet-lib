# Prototype derivation notes

**Last regenerated**: 2026-04-19 (second pass, user-calibrated sampling)
**LLM**: `mistralai/mistral-large-2512` via OpenRouter (external backend)
**Entry point**: `scripts/extract_prototypes.py`
**Wall-clock**: ~6 min (Qdrant scroll + 9 cluster chunks + 1 merge)

## Sampling strategy

The chat corpus is **skewed**: a one-time historical dump
(`google_takeout_gemini`, ~4k-5k messages) dwarfs the live streams
(`skynet_chat` ~150, `claude_sessions` ~100). A single fixed sample (e.g. 1500
uniform) drowns the user's live voice in the historical dump and produces
prototypes that describe who the user was on Gemini a year ago, not who they
are now in Skynet.

### Formula (`sample_per_bucket`)

```python
LIVE_SOURCES  = {"skynet_chat", "claude_sessions", "matrix_dm", "matrix_chat"}
LIVE_FULL_UNTIL = 500   # take the whole live bucket below this
HISTORICAL_CAP  = 1000  # hard cap per historical bucket

def sample_per_bucket(buckets: dict[str, int]) -> dict[str, int]:
    out = {}
    for source, count in buckets.items():
        if source in LIVE_SOURCES:
            out[source] = count if count <= LIVE_FULL_UNTIL \
                          else LIVE_FULL_UNTIL + (count - LIVE_FULL_UNTIL) // 2
        else:
            out[source] = min(count, HISTORICAL_CAP)
    return out
```

Rationale:

- **Live buckets** (`skynet_chat`, `claude_sessions`) are precious and small.
  Take them in full up to 500 messages; half-subsample above that so they
  don't dominate a future fat live stream either.
- **Historical dumps** are hard-capped at 1000 to cover breadth without
  drowning out live voice. New dumps beyond that threshold are
  recency-weighted (see below) rather than uniformly truncated.

### Recency weighting (`recency_weighted_sample`)

When a bucket exceeds its allotment, we downsample with **exponential-decay
weights** rather than uniform random, so newer messages dominate without
excluding the historical tail:

```python
weight(msg) = exp(-age_days / half_life_days)    # default half_life = 90d
```

Drawing is `random.choices`-style (with-replacement under the hood, then
deduped and topped up by descending weight so the final pick is unique).
Timestamps that fail to parse get the **median** weight of valid candidates
so they neither dominate nor disappear.

Seed (`seed=42`) is fixed so a re-run on the same corpus is reproducible.

**Spot-check from this run**: on a synthetic pool spanning 0-500 days old,
the pool median age was 247 days, the picked median age 72 days — a ~3.4x
pull toward recent voice, exactly as intended. On the real
`google_takeout_gemini` bucket (ingested via backfill in one wave), most
timestamps point to the ingestion window, so the recency weights collapse
to near-uniform; the diversity that matters there comes from the hard cap
plus chunked clustering rather than from recency.

## Today's allotment

| Source | Raw bucket (>=50 chars) | Allotment |
|--------|-------------------------|-----------|
| `skynet_chat` (live) | 148 | 148 (full, <=500) |
| `claude_sessions` (live) | 87 | 87 (full, <=500) |
| `google_takeout_gemini` (historical) | 4118 | 1000 (capped) |
| **Total** | **4353** | **1235** |

*(`skynet_chat` raw count is 149 pre-filter; 1 row shorter than 50 chars was dropped. `claude_sessions` raw count is 108 pre-filter; 21 short rows dropped. The task brief quoted pre-filter counts; above counts are post-filter.)*

## Backend selection

Pluggable backends in `BACKENDS` dict:

| Backend | Kind | Model | PII redaction |
|---------|------|-------|---------------|
| `ollama_gemma4` | local Ollama | `gemma4:31b` | not required (on-cluster) |
| `ollama_qwen3_32b` | local Ollama | `qwen3:32b` | not required |
| `openrouter_mistral_large` | external | `mistralai/mistral-large-2512` | **required** |

The entry point `extract_prototypes(sample, primary_backend=..., external_fallback=...)`:

1. Runs the primary backend in chunks.
2. Evaluates a **quality gate** after clustering:
   - `parse_failure_rate > 0.3` (more than 30% chunks produced unparseable JSON)
   - OR `num_domains < min_domains` (default 8)
3. If either trips, re-runs the entire clustering via the external fallback.
4. Environment override: `EXTRACTION_USE_EXTERNAL=true` forces external from
   the start (useful for local-development dry runs where Ollama is not
   reachable).

**Today's run**: `primary_backend=openrouter_mistral_large` directly, as the
user requested Mistral for the first pass ("для початку містрал").

**Future DAG default**: `primary_backend=ollama_gemma4` with Mistral as
automatic fallback. Local-first keeps the raw corpus on-cluster (no PII
redaction needed for local) and only escalates when local quality is
insufficient.

## PII policy

When the chosen backend is `external` (`requires_redaction=True`),
`redact()` strips:

- **tokens** (`sk-...`, `ghp_...`, `hf_...`) → `[token]`
- **emails** (`foo@bar.baz`) → `[email]`
- **IPs** (`1.2.3.4`) → `[ip]`
- **phones** (`+380...` etc.) → `[phone]`

Counts are accumulated in the run stats. **This run redacted**: 28 phones,
20 IPs, 2 emails.

Local backends skip redaction (raw text gives the LLM maximum signal and
the data never leaves the cluster network).

## Final domain count: 17 (+ 3 baseline)

| # | Domain |
|---|--------|
| 1 | `ai-infrastructure-optimization` |
| 2 | `personal-cognitive-architecture` |
| 3 | `horizontal-networks-philosophy` |
| 4 | `personal-memory-identity-systems` |
| 5 | `self-hosted-infrastructure-management` |
| 6 | `autonomous-ai-agent-development` |
| 7 | `personal-identity-knowledge-graph` |
| 8 | `neuroscience-consciousness-research` |
| 9 | `indoor-gardening-automation` |
| 10 | `philosophy-metaphysics-consciousness` |
| 11 | `advanced-mathematical-physics` |
| 12 | `botanical-cultivation-research` |
| 13 | `personal-tech-optimization` |
| 14 | `herbal-vaporization-techniques` |
| 15 | `cybersecurity-post-quantum-cryptography` |
| 16 | `quantum-cosmology-physics` |
| 17 | `linux-devops-ai-systems` |

Baseline (hand-seeded, not from corpus): `movies`, `music`, `books`.

## Diff vs prior 14-domain extraction (2026-04-19 morning, 478 sampled)

**New/split domains** (not present in prior pass):

- `horizontal-networks-philosophy` — теза "вертикаль програє, горизонталь
  виграє" висвітлилась як окремий кластер завдяки ширшій вибірці з takeout.
- `personal-memory-identity-systems` — self-hosted memory / markdown+vector
  practice окремо від агентів.
- `personal-identity-knowledge-graph` — user's meta-commentary about own
  profile ("що зараз воно назбирало про мене") was its own signal strong
  enough for a split.
- `neuroscience-consciousness-research` — OPM-MEG / GWT / OpenBCI / gamma
  waves: this showed up only with the wider takeout sample.
- `indoor-gardening-automation` & `botanical-cultivation-research` — the
  single prior domain `autonomous-physical-systems-and-bio-digital-integration`
  split into engineering (climate/ventilation) vs chemistry (nutrients,
  coco, harmala prep).
- `herbal-vaporization-techniques` — TM2/damiana/blue-lotus content became
  dense enough for its own prototype.
- `personal-tech-optimization` — M4 Max, Sony XM5, Onyx Boox, etc. — was
  previously folded into devops.
- `cybersecurity-post-quantum-cryptography` — PQC / lattice / YubiKey.
- `quantum-cosmology-physics` — inflation, Penrose process, Planckian foam.
- `linux-devops-ai-systems` — Qwen3/Ollama/podman + Arch dual-boot (split
  from broader devops).

**Merged / absorbed domains**:

- `cognitive-architecture-and-attention-management`,
  `cognitive-load-and-neurobiological-constraints`,
  `epistemology-and-intuitive-knowledge-systems` → all folded into
  `personal-cognitive-architecture`. The new merge is more compact.
- `embodied-cognition-and-physical-reset-protocols`,
  `physical-anchoring-and-sensory-calibration` — dropped. The wider sample
  diluted their signal (gym/travel talk is a small fraction of the 1235
  corpus); they didn't survive the merge step. If they matter for routing,
  re-add via `PrototypeRegistry.add()` at runtime.
- `existential-cinema-and-temporal-aesthetics` — **dropped**. Movie talk
  didn't cluster in this sample; cinema preferences will route through the
  hand-seeded `movies` baseline instead. This is the expected mode once
  skynet-movies owns film signal.
- `ai-personality-and-autonomous-agents` → renamed / refined into
  `autonomous-ai-agent-development` (more engineering-leaning).
- `autonomous-ai-ecosystem-architecture`, `devops-and-gitops-infrastructure`
  → split between `ai-infrastructure-optimization`,
  `self-hosted-infrastructure-management`, `linux-devops-ai-systems`.
- `mathematical-philosophy-and-abstract-systems`,
  `philosophy-of-excess-and-nonlinear-thinking`,
  `existential-philosophy-and-consciousness-transitions` → merged into
  `philosophy-metaphysics-consciousness` + `advanced-mathematical-physics`.
- `self-quantification-and-biometric-systems` — **dropped**. The earlier
  sample was heavy on "як я спав?" debugging; with more takeout the vector
  got diluted. Likely to reappear once more live sleep-data messages accrue.

Overall: the new extraction is **broader in coverage** (17 vs 14), trades
some live-chat idiosyncrasies (gym, sleep-tracking, cinema) for
deeper-thematic slices mined from takeout (neuroscience, PQC, cosmology,
vaping). This is the expected shape once live buckets stay small and the
takeout dominates 80% of the sample — tuning `LIVE_FULL_UNTIL` upward or
the cap downward will shift the balance back toward live signal.

## Future DAG plan

A periodic DAG (suggested cadence: weekly) should:

1. Call `sample_from_qdrant(sources=[...])` — auto-calibrates per-bucket
   allotment from current Qdrant counts.
2. Call `extract_prototypes(sample, primary_backend="ollama_gemma4",
   external_fallback="openrouter_mistral_large")`.
3. Render YAML via `render_yaml()` and open a PR on
   `sanscfs/skynet-lib` against `skynet-vibe/src/skynet_vibe/config/default_prototypes.yaml`
   preserving the baseline block verbatim.
4. Human review before merge (topic drift, seed-phrase quality).

## Stats file (machine-readable run summary)

```json
{
  "backend_used": "openrouter_mistral_large",
  "sample_size": 1235,
  "bucket_sizes": {
    "skynet_chat": 148,
    "claude_sessions": 87,
    "google_takeout_gemini": 4118
  },
  "allotments": {
    "skynet_chat": 148,
    "claude_sessions": 87,
    "google_takeout_gemini": 1000
  },
  "source_breakdown": {
    "skynet_chat": 148,
    "claude_sessions": 87,
    "google_takeout_gemini": 1000
  },
  "domain_count": 17,
  "redaction": {"phone": 28, "ip": 20, "email": 2},
  "half_life_days": 90.0
}
```
