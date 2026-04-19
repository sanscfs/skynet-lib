"""Extract domain prototypes from chat corpus via LLM clustering.

Pluggable backends:
- ``ollama_gemma4``     -> local Ollama on 100.64.0.4 (model ``gemma4:31b``),
                          DEFAULT, no PII redaction (data stays on-cluster).
- ``ollama_qwen3_32b``  -> local Ollama alt, same privacy envelope.
- ``openrouter_mistral_large`` -> external (OpenRouter), REQUIRES PII redaction.

Designed as the entry point for a future periodic DAG: call
``extract_prototypes(sample, ...)`` with ``primary_backend`` set to a local
backend; the function will fall back to ``external_fallback`` automatically
if the local backend trips a quality gate (parse-failure rate or too few
final domains). The wrapper DAG is responsible for:
  1. Sampling fresh messages from ``user_profile_raw``.
  2. Calling ``extract_prototypes(...)``.
  3. Writing the resulting YAML to ``default_prototypes.yaml`` while
     preserving the baseline block (movies / music / books) at the bottom.
  4. Opening a PR on sanscfs/skynet-lib for human review.

Usage (manual, one-shot):

    python extract_prototypes.py \\
        --sources skynet_chat,claude_sessions,google_takeout_gemini \\
        --backend openrouter_mistral_large \\
        --output /tmp/prototypes.yaml

or with explicit per-bucket quotas (bypass auto-calibration):

    python extract_prototypes.py \\
        --source skynet_chat:149,claude_sessions:108,google_takeout_gemini:1000 \\
        --backend openrouter_mistral_large \\
        --output /tmp/prototypes.yaml

The script MUST be run inside the cluster (e.g. skynet-analyzer pod) so the
OpenRouter key never leaves the cluster and Qdrant is reachable via its
internal service name.

Policy: ``redact_pii=True`` is forced whenever the chosen backend is
external. Local backends skip redaction (raw text gives the LLM the best
signal and the data never leaves our network).

Sampling strategy (auto-calibrated, see ``sample_per_bucket``):
  * live sources (skynet_chat, claude_sessions, matrix_*) taken in full up
    to ``LIVE_FULL_UNTIL`` (500); above that they're half-subsampled.
  * historical dumps (everything else) are hard-capped at ``HISTORICAL_CAP``
    (1000) per bucket.
  * when a bucket is downsampled, ``recency_weighted_sample`` picks with
    exponential-decay weights (``half_life_days=90`` default) so newer
    messages dominate without excluding the historical tail.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger("extract_prototypes")


# ---------------------------------------------------------------------------
# Per-bucket dynamic sampling (calibrated to user's distribution)
# ---------------------------------------------------------------------------
#
# The chat corpus is skewed: a one-time historical dump (``google_takeout_gemini``
# ~4k-5k messages) dwarfs the live streams (``skynet_chat``, ``claude_sessions``
# currently ~100-200 each). Fixed uniform sampling drowns the live voice in the
# historical dump. ``sample_per_bucket`` returns per-source counts matched to
# that distribution: live buckets are taken in full up to ``LIVE_FULL_UNTIL``
# and half-subsampled above, historical dumps are hard-capped at
# ``HISTORICAL_CAP``. When a bucket ends up downsampled, ``recency_weighted_sample``
# biases the pick toward newer messages so the extracted prototypes track the
# user's current voice rather than ancient history.

LIVE_SOURCES: set[str] = {"skynet_chat", "claude_sessions", "matrix_dm", "matrix_chat"}
HISTORICAL_CAP = 1000   # per-bucket cap for one-time historical dumps
LIVE_FULL_UNTIL = 500   # below this, take the whole live bucket


def sample_per_bucket(buckets: dict[str, int]) -> dict[str, int]:
    """Return per-source sample counts for this user's distribution.

    Live small buckets: full up to ``LIVE_FULL_UNTIL``, then half-subsample
    after. Historical dumps (``google_takeout_gemini`` etc.) are capped at
    ``HISTORICAL_CAP``.
    """
    out: dict[str, int] = {}
    for source, count in buckets.items():
        if count <= 0:
            out[source] = 0
            continue
        if source in LIVE_SOURCES:
            if count <= LIVE_FULL_UNTIL:
                out[source] = count   # take the whole bucket
            else:
                out[source] = LIVE_FULL_UNTIL + (count - LIVE_FULL_UNTIL) // 2
        else:   # historical / dumps
            out[source] = min(count, HISTORICAL_CAP)
    return out


def _parse_ts(ts: Any) -> datetime | None:
    """Best-effort timestamp parse. Accept ISO strings (optionally Z-suffixed)
    and numeric epoch (s or ms). Returns naive UTC datetime or None."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        # Heuristic: values >1e12 are ms, else seconds.
        val = float(ts)
        if val > 1e12:
            val /= 1000.0
        try:
            return datetime.fromtimestamp(val, tz=timezone.utc).replace(tzinfo=None)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return None
        # Normalize trailing Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    return None


def recency_weighted_sample(
    candidates: list[dict[str, Any]],
    target_count: int,
    *,
    half_life_days: float = 90.0,
    seed: int | None = 42,
) -> list[dict[str, Any]]:
    """Weighted sample by recency. Newer messages more likely to be picked.

    Uses ``random.choices`` with exponential-decay weights so older messages
    are underweighted but not excluded. Messages with unparsable timestamps
    get a neutral weight equal to the median so they neither dominate nor
    disappear. Returns ``min(target_count, len(candidates))`` items.
    """
    if not candidates:
        return []
    if target_count >= len(candidates):
        return list(candidates)

    now = datetime.utcnow()
    weights: list[float] = []
    for c in candidates:
        ts = _parse_ts(c.get("timestamp"))
        if ts is None:
            weights.append(float("nan"))
            continue
        age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
        weights.append(math.exp(-age_days / half_life_days))

    # Replace NaN (missing timestamps) with the median of valid weights so
    # they have a neutral chance rather than 0 or infinity.
    valid = [w for w in weights if not math.isnan(w)]
    if valid:
        median = sorted(valid)[len(valid) // 2]
    else:
        median = 1.0
    weights = [median if math.isnan(w) else w for w in weights]

    rng = random.Random(seed)
    # Without replacement: draw indices weighted, dedupe, top up if needed.
    picked_idx: list[int] = []
    seen: set[int] = set()
    # Over-sample to account for collisions then trim.
    attempts = 0
    max_attempts = target_count * 8
    while len(picked_idx) < target_count and attempts < max_attempts:
        idx = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
        if idx not in seen:
            seen.add(idx)
            picked_idx.append(idx)
        attempts += 1
    # Fallback: fill any remainder deterministically by descending weight.
    if len(picked_idx) < target_count:
        order = sorted(range(len(candidates)), key=lambda i: -weights[i])
        for i in order:
            if i not in seen:
                picked_idx.append(i)
                seen.add(i)
                if len(picked_idx) >= target_count:
                    break
    return [candidates[i] for i in picked_idx]


# ---------------------------------------------------------------------------
# PII redaction (external-backend only)
# ---------------------------------------------------------------------------

# Order matters: high-specificity patterns first (token, email, ip) so the
# greedier phone regex doesn't cannibalize digits inside an API key or IP.
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "token": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "ip": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "phone": re.compile(r"\+?\d[\d\s\-()]{8,}\d"),
}


def redact(text: str, counter: dict[str, int] | None = None) -> str:
    """Strip PII. ``counter`` (optional) accumulates per-kind hit counts."""
    out = text
    for kind, pat in _PII_PATTERNS.items():
        def _sub(m: re.Match[str], _k: str = kind) -> str:
            if counter is not None:
                counter[_k] = counter.get(_k, 0) + 1
            return f"[{_k}]"

        out = pat.sub(_sub, out)
    return out


# ---------------------------------------------------------------------------
# Qdrant sampling
# ---------------------------------------------------------------------------


def _scroll_all_for_source(
    client: httpx.Client,
    qdrant_url: str,
    collection: str,
    src: str,
    *,
    min_len: int,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    """Scroll all points for a given ``source`` value, keeping payloads with
    ``len(text) > min_len``. Returns a flat list of ``{source, text, timestamp,
    category}`` dicts."""
    out: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        body: dict[str, Any] = {
            "limit": page_size,
            "with_payload": {"include": ["source", "text", "timestamp", "category"]},
            "with_vector": False,
            "filter": {"must": [{"key": "source", "match": {"value": src}}]},
        }
        if offset is not None:
            body["offset"] = offset
        r = client.post(
            f"{qdrant_url}/collections/{collection}/points/scroll",
            json=body,
        )
        r.raise_for_status()
        d = r.json()["result"]
        for p in d["points"]:
            t = (p["payload"].get("text") or "").strip()
            if len(t) <= min_len:
                continue
            out.append(
                {
                    "source": src,
                    "text": t,
                    "timestamp": p["payload"].get("timestamp"),
                    "category": p["payload"].get("category"),
                }
            )
        offset = d.get("next_page_offset")
        if offset is None:
            break
    return out


def sample_from_qdrant(
    *,
    qdrant_url: str,
    collection: str = "user_profile_raw",
    sources: list[str] | None = None,
    quotas: dict[str, int] | None = None,
    min_len: int = 50,
    half_life_days: float = 90.0,
    seed: int | None = 42,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Pull all points per source (text length > ``min_len``), then apply
    ``sample_per_bucket`` to decide per-bucket allotments, then apply
    ``recency_weighted_sample`` for buckets that need downsampling.

    Either pass ``sources`` (auto-discover per-bucket sizes and calibrate) or
    ``quotas`` (explicit per-source caps — used when a caller already knows
    exactly how many to take).

    Returns ``(sample, bucket_sizes, allotments)``.
    """
    if not sources and not quotas:
        raise ValueError("sample_from_qdrant: pass either sources= or quotas=")
    if quotas is not None and sources is None:
        sources = list(quotas.keys())
    assert sources is not None

    # Step 1: pull the raw bucket for each source.
    buckets_raw: dict[str, list[dict[str, Any]]] = {}
    with httpx.Client(timeout=60.0) as client:
        for src in sources:
            buckets_raw[src] = _scroll_all_for_source(
                client, qdrant_url, collection, src, min_len=min_len,
            )
            logger.info("scrolled source=%s -> %d messages", src, len(buckets_raw[src]))

    bucket_sizes = {s: len(msgs) for s, msgs in buckets_raw.items()}

    # Step 2: decide per-bucket allotment (explicit quotas override).
    if quotas is not None:
        allotments = {s: min(quotas.get(s, 0), bucket_sizes.get(s, 0)) for s in sources}
    else:
        allotments = sample_per_bucket(bucket_sizes)

    # Step 3: apply recency-weighted downsampling where needed.
    out: list[dict[str, Any]] = []
    for src in sources:
        allotted = allotments.get(src, 0)
        pool = buckets_raw.get(src, [])
        if allotted <= 0 or not pool:
            continue
        if allotted >= len(pool):
            out.extend(pool)
            continue
        picked = recency_weighted_sample(
            pool, allotted, half_life_days=half_life_days, seed=seed,
        )
        out.extend(picked)
        # Spot-check log: compare median age of pool vs picked so we can
        # confirm the weighting nudged toward newer messages.
        def _median_age(items: list[dict[str, Any]]) -> float | None:
            ages = []
            now = datetime.utcnow()
            for m in items:
                ts = _parse_ts(m.get("timestamp"))
                if ts is not None:
                    ages.append((now - ts).total_seconds() / 86400.0)
            if not ages:
                return None
            ages.sort()
            return ages[len(ages) // 2]
        logger.info(
            "recency-weighted %s: pool=%d median_age=%s days, picked=%d median_age=%s days",
            src, len(pool), _median_age(pool), len(picked), _median_age(picked),
        )
    return out, bucket_sizes, allotments


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass
class BackendSpec:
    name: str
    kind: str  # "ollama" or "openrouter"
    base_url: str
    model: str
    requires_redaction: bool = False
    api_key_env: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


BACKENDS: dict[str, BackendSpec] = {
    "ollama_gemma4": BackendSpec(
        name="ollama_gemma4",
        kind="ollama",
        base_url="http://100.64.0.4:11434",
        model="gemma4:31b",
        requires_redaction=False,
        options={"temperature": 0.2, "num_ctx": 16384},
    ),
    "ollama_qwen3_32b": BackendSpec(
        name="ollama_qwen3_32b",
        kind="ollama",
        base_url="http://100.64.0.4:11434",
        model="qwen3:32b",
        requires_redaction=False,
        options={"temperature": 0.2, "num_ctx": 16384},
    ),
    "openrouter_mistral_large": BackendSpec(
        name="openrouter_mistral_large",
        kind="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="mistralai/mistral-large-2512",
        requires_redaction=True,
        api_key_env="LLM_API_KEY",
        options={"temperature": 0.2},
    ),
}


def _call_llm(
    backend: BackendSpec,
    prompt: str,
    *,
    timeout: float = 300.0,
) -> str:
    if backend.kind == "ollama":
        r = httpx.post(
            f"{backend.base_url}/api/generate",
            json={
                "model": backend.model,
                "prompt": prompt,
                "stream": False,
                "options": backend.options,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    if backend.kind == "openrouter":
        key = os.environ.get(backend.api_key_env or "")
        if not key:
            raise RuntimeError(f"Missing ${backend.api_key_env} for {backend.name}")
        r = httpx.post(
            f"{backend.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": backend.model,
                "temperature": backend.options.get("temperature", 0.2),
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise ValueError(f"unknown backend kind: {backend.kind}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CLUSTER_PROMPT = """You are extracting DOMAIN PROTOTYPES from a user's chat corpus.

Cluster the messages below into 2-5 coherent SPECIFIC domains. A domain is a stable area of interest — NOT a mood, NOT a single event.

For each domain emit:
- "name": slug (kebab-case, 3-6 words, specific not generic)
- "description": one-line English description of what the domain is about
- "seed_phrases": 5-7 vivid phrases PULLED FROM THE MESSAGES (keep original language — Ukrainian/English/mixed is fine). Phrases must be concrete, not abstract. They will later be embedded as the domain centroid, so they must be representative of real user language.

Rules:
- NEVER include personal names, emails, phones, API keys.
- Do NOT invent domains that are not supported by the messages.
- Keep author/title/place names — those are signal.
- Return ONLY a JSON array, no prose wrapper.

Messages:
{messages}

JSON array:"""


_MERGE_PROMPT = """You are consolidating domain proposals from multiple clustering chunks.

Input: a list of domain proposals, each with "name", "description", "seed_phrases".

Merge overlapping domains. Keep specific over generic. Split only when distinct enough to warrant separate routing. Target: {target_min}-{target_max} final domains.

For each FINAL domain emit:
- "name": kebab-case slug
- "description": one-line English description
- "seed_phrases": 5-7 best phrases (dedupe, keep most vivid, original language preserved)

NEVER include personal names, emails, phones, API keys.
Return ONLY a JSON array.

Proposals:
{proposals}

Final JSON array:"""


# ---------------------------------------------------------------------------
# Parsing + clustering
# ---------------------------------------------------------------------------


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_json_array(raw: str) -> list[dict[str, Any]] | None:
    raw = raw.strip()
    # Strip fenced code blocks if present
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    m = _JSON_ARRAY_RE.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    cleaned: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        desc = str(item.get("description") or "").strip()
        phrases = item.get("seed_phrases") or []
        if not name or not phrases:
            continue
        cleaned.append(
            {
                "name": name,
                "description": desc,
                "seed_phrases": [str(p).strip() for p in phrases if str(p).strip()],
            }
        )
    return cleaned


def _chunk(lst: list[Any], n: int) -> list[list[Any]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def run_clustering(
    sample: list[dict[str, Any]],
    *,
    backend: str,
    chunk_size: int = 150,
    target_min: int = 15,
    target_max: int = 20,
    redact_pii: bool | None = None,
    pii_counter: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Cluster ``sample`` through ``backend`` in chunks; merge into final set.

    Returns (proposals, num_chunks, parse_failures).
    ``redact_pii`` None -> derived from backend.requires_redaction.
    """
    spec = BACKENDS[backend]
    if redact_pii is None:
        redact_pii = spec.requires_redaction

    chunks = _chunk(sample, chunk_size)
    proposals: list[dict[str, Any]] = []
    parse_fails = 0

    for i, chunk in enumerate(chunks, 1):
        lines: list[str] = []
        for j, m in enumerate(chunk, 1):
            src = m.get("source") or "?"
            txt = m.get("text") or ""
            if redact_pii:
                txt = redact(txt, pii_counter)
            txt = txt.replace("\n", " ").strip()
            if len(txt) > 800:
                txt = txt[:800] + "..."
            lines.append(f"[{j}] ({src}) {txt}")
        prompt = _CLUSTER_PROMPT.format(messages="\n".join(lines))
        logger.info("cluster chunk %d/%d via %s (%d msgs)", i, len(chunks), backend, len(chunk))
        try:
            raw = _call_llm(spec, prompt)
        except Exception as e:
            logger.warning("cluster chunk %d failed: %s", i, e)
            parse_fails += 1
            continue
        parsed = _parse_json_array(raw)
        if parsed is None:
            logger.warning("cluster chunk %d: could not parse JSON", i)
            parse_fails += 1
            continue
        proposals.extend(parsed)

    if not proposals:
        return [], len(chunks), parse_fails

    merge_input = json.dumps(proposals, ensure_ascii=False, indent=2)
    merge_prompt = _MERGE_PROMPT.format(
        target_min=target_min, target_max=target_max, proposals=merge_input
    )
    logger.info("merging %d proposals -> %d-%d final via %s",
                len(proposals), target_min, target_max, backend)
    try:
        raw = _call_llm(spec, merge_prompt, timeout=600.0)
    except Exception as e:
        logger.warning("merge step failed: %s; returning unmerged proposals", e)
        return proposals, len(chunks), parse_fails
    merged = _parse_json_array(raw)
    if merged is None:
        logger.warning("merge step: could not parse JSON; returning unmerged")
        return proposals, len(chunks), parse_fails + 1
    return merged, len(chunks), parse_fails


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def extract_prototypes(
    sample: list[dict[str, Any]],
    *,
    primary_backend: str = "ollama_gemma4",
    external_fallback: str = "openrouter_mistral_large",
    min_domains: int = 8,
    max_parse_failure_rate: float = 0.3,
    target_min: int = 15,
    target_max: int = 20,
    chunk_size: int = 150,
    pii_counter: dict[str, int] | None = None,
    force_external: bool | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Run clustering. Try primary; fall back to external if quality gate fails.

    Returns (proposals, backend_used).

    ``force_external`` (or env ``EXTRACTION_USE_EXTERNAL=true``) bypasses the
    primary and calls the external backend directly.
    """
    if force_external is None:
        force_external = os.environ.get("EXTRACTION_USE_EXTERNAL", "").lower() in {"1", "true", "yes"}

    if force_external:
        logger.info("EXTRACTION_USE_EXTERNAL=true -> using %s directly", external_fallback)
        proposals, _, _ = run_clustering(
            sample, backend=external_fallback,
            chunk_size=chunk_size, target_min=target_min, target_max=target_max,
            pii_counter=pii_counter,
        )
        return proposals, external_fallback

    proposals, n_chunks, parse_fails = run_clustering(
        sample, backend=primary_backend,
        chunk_size=chunk_size, target_min=target_min, target_max=target_max,
        pii_counter=pii_counter,
    )
    failure_rate = parse_fails / max(n_chunks, 1)
    too_few = len(proposals) < min_domains

    if failure_rate > max_parse_failure_rate or too_few:
        logger.warning(
            "primary %s failed quality gate (parse_fail_rate=%.2f domains=%d); "
            "falling back to %s",
            primary_backend, failure_rate, len(proposals), external_fallback,
        )
        proposals, _, _ = run_clustering(
            sample, backend=external_fallback,
            chunk_size=chunk_size, target_min=target_min, target_max=target_max,
            pii_counter=pii_counter,
        )
        return proposals, external_fallback
    return proposals, primary_backend


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------

_BASELINE_BLOCK = """# ---------------------------------------------------------------------------
# Baseline domain prototypes for recommendation routing.
#
# These do NOT come from chat corpus — they are hand-seeded so domain-specific
# bots (skynet-movies, skynet-music, future skynet-books) always have a base
# prototype to filter against. The chat-derived prototypes above enrich the
# vocabulary; these ensure the bots can always route.
#
# A signal about Tarkovsky registers at high cosine to BOTH
# `existential-cinema-and-temporal-aesthetics` AND `movies` — this is the
# intended overlap (multi-domain via cosine).
# ---------------------------------------------------------------------------

movies:
  - "films cinema directors cinematography screenplay actors"
  - "visual language editing pacing tone atmosphere"
  - "arthouse horror noir drama documentary"
  - "narrative structure shot composition mise en scène"
  - "watched rewatched theatre streaming letterboxd"

music:
  - "albums tracks artists listening melodies genres"
  - "rhythm beat production mixing recording bass drums"
  - "live performance concert playlist playback discography"
  - "ambient electronic jazz classical rock experimental"
  - "listened saved rediscovered looped headphones"

books:
  - "books reading authors novels essays poetry"
  - "fiction nonfiction philosophy theory memoir"
  - "pages prose voice narrative style"
  - "translated originally published annotated"
  - "read finished abandoned rereading bookmarked"
"""


def _yaml_escape(s: str) -> str:
    # Always quote; escape double quotes and backslashes
    return s.replace("\\", "\\\\").replace('"', '\\"')


def render_yaml(proposals: list[dict[str, Any]], *, header: str | None = None) -> str:
    """Render final YAML with chat-derived domains + baseline block appended."""
    lines: list[str] = []
    if header:
        lines.append(header.rstrip() + "\n")
    for p in proposals:
        name = p["name"].strip()
        desc = p.get("description", "").strip()
        if desc:
            lines.append(f"# {desc}")
        lines.append(f"{name}:")
        for phrase in p["seed_phrases"]:
            lines.append(f'  - "{_yaml_escape(phrase)}"')
        lines.append("")
    lines.append(_BASELINE_BLOCK.rstrip() + "\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_quotas(spec: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition(":")
        out[k.strip()] = int(v)
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL") or
                    f"http://{os.environ.get('QDRANT_HOST', 'qdrant.qdrant.svc')}:{os.environ.get('QDRANT_PORT', '6333')}")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--sources",
                       help="comma list of source values; per-bucket counts auto-calibrated "
                            "(e.g. skynet_chat,claude_sessions,google_takeout_gemini)")
    group.add_argument("--source",
                       help="comma list source:quota (explicit per-source caps; bypasses auto-calibration)")
    ap.add_argument("--backend", default="ollama_gemma4", choices=list(BACKENDS))
    ap.add_argument("--fallback", default="openrouter_mistral_large", choices=list(BACKENDS))
    ap.add_argument("--target-min", type=int, default=15)
    ap.add_argument("--target-max", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=150)
    ap.add_argument("--min-domains", type=int, default=8)
    ap.add_argument("--max-parse-failure-rate", type=float, default=0.3)
    ap.add_argument("--half-life-days", type=float, default=90.0,
                    help="recency weighting half-life for downsampled buckets")
    ap.add_argument("--seed", type=int, default=42, help="random seed for recency sampling")
    ap.add_argument("--output", required=True, help="where to write the final YAML")
    ap.add_argument("--stats-output", default=None, help="optional JSON file with stats")
    args = ap.parse_args(argv)

    if args.sources:
        sources_list = [s.strip() for s in args.sources.split(",") if s.strip()]
        sample, bucket_sizes, allotments = sample_from_qdrant(
            qdrant_url=args.qdrant_url,
            sources=sources_list,
            half_life_days=args.half_life_days,
            seed=args.seed,
        )
    else:
        quotas = _parse_quotas(args.source or "")
        logger.info("explicit quotas: %s", quotas)
        sample, bucket_sizes, allotments = sample_from_qdrant(
            qdrant_url=args.qdrant_url,
            quotas=quotas,
            half_life_days=args.half_life_days,
            seed=args.seed,
        )
    logger.info("bucket_sizes=%s, allotments=%s", bucket_sizes, allotments)
    logger.info("sampled %d messages (%s)",
                len(sample),
                ", ".join(f"{s}={sum(1 for m in sample if m['source']==s)}"
                         for s in allotments))

    pii_counter: dict[str, int] = {}
    proposals, used_backend = extract_prototypes(
        sample,
        primary_backend=args.backend,
        external_fallback=args.fallback,
        min_domains=args.min_domains,
        max_parse_failure_rate=args.max_parse_failure_rate,
        target_min=args.target_min,
        target_max=args.target_max,
        chunk_size=args.chunk_size,
        pii_counter=pii_counter,
    )

    header = (
        "# Default domain prototypes for skynet-vibe.\n"
        "#\n"
        f"# Generated via {used_backend} from calibrated per-bucket chat corpus sample\n"
        "# (see docs/prototypes_derivation.md for sampling strategy + model details).\n"
        "#\n"
        "# Each top-level key is a domain; the value is a list of seed phrases\n"
        "# embedded and mean-centroided by PrototypeRegistry.load_from_config().\n"
    )
    yaml_out = render_yaml(proposals, header=header)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(yaml_out)
    logger.info("wrote %d chat-derived domains to %s", len(proposals), args.output)

    stats = {
        "backend_used": used_backend,
        "sample_size": len(sample),
        "bucket_sizes": bucket_sizes,
        "allotments": allotments,
        "source_breakdown": {s: sum(1 for m in sample if m["source"] == s) for s in allotments},
        "domain_count": len(proposals),
        "redaction": pii_counter,
        "half_life_days": args.half_life_days,
    }
    if args.stats_output:
        with open(args.stats_output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
