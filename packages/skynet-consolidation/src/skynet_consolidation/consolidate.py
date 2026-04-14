"""
consolidate.py — turn a clique of related memories into a single
summary with explicit provenance.

The caller hands us:
  - `members`: list of `{id, text, ...}` dicts (one per clique point)
  - `structural_edges`: optional list of typed relationships between
    them (supersedes/contradicts/elaborates/caused_by/example_of)
  - `llm_client`: a callable that wraps the deployment's preferred
    chat model

We return a `ConsolidationResult` whose shape mirrors what the DAG
writes to Qdrant:
  - `summary`: the merged text (<= ~1000 chars)
  - `discarded_ids`: member ids whose content is fully captured by
    the summary OR superseded by a newer member — these should
    become `archived=true` once the summary point is persisted
  - `contradictions`: pairs where members disagree; the LLM flags
    them so the caller can surface them separately rather than
    silently smoothing over disagreement
  - `confidence`: self-reported LLM confidence in [0, 1]; DAG uses
    it to route high-confidence results to auto-archive vs a human
    review queue

Design notes:
  - The prompt is deliberate: "write a single-paragraph, first-
    person-if-the-source-is-first-person summary that preserves
    the user's own vocabulary. Do not invent facts. If two members
    conflict, mention the conflict explicitly — do not pick a
    winner silently." This keeps the consolidation faithful to the
    raw evidence and lets Phase 6 wiki promotion consume the output.
  - Inputs are truncated at `MEMBER_TEXT_CAP` chars per member so a
    pathological 20kB wall of text in one member doesn't blow the
    prompt window.
  - `summarise_prompt` is exposed verbatim so the atlas formulas
    catalog can render it and so tests can pin on the exact shape.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


#: Max chars per member text passed to the LLM. Bigger cliques
#: dominate the prompt if members are unbounded; 1500 gives a
#: comfortable window while keeping the total prompt under the
#: context limit of most value-tier models.
MEMBER_TEXT_CAP = 1500

#: Max number of members in a single consolidation call. The
#: caller (DAG) should have already down-sampled if needed; this
#: is a defence-in-depth clamp so a misconfigured clique (1000
#: points) doesn't run a 200k-token prompt by accident.
MAX_MEMBERS = 30


LlmClient = Callable[[str, str], str]


@dataclass(slots=True, frozen=True)
class Contradiction:
    """One conflict the LLM flagged within the clique.

    Both ids must be member ids; `explanation` is a one-line
    description of what disagrees. The DAG surfaces these in the
    Atlas review queue so a human can decide which side wins
    before the summary fully archives the sources.
    """

    id_a: Any
    id_b: Any
    explanation: str


@dataclass(slots=True)
class ConsolidationResult:
    """Output of `consolidate_clique`.

    - `summary` is "" on LLM failure; caller should treat empty
      summaries as "skip this clique for now".
    - `discarded_ids` is a subset of the input member ids; the DAG
      archives these AFTER persisting the summary point with
      `consolidated_from = [...]` back-references.
    - `contradictions` may be non-empty even when confidence is
      high — conflict is a signal to surface, not a failure.
    - `confidence` ∈ [0.0, 1.0]. Caller decides the auto-commit
      threshold; the library doesn't pre-judge.
    - `raw_response` is kept for telemetry / debugging when the
      parser falls back.
    """

    summary: str
    discarded_ids: list[Any] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    confidence: float = 0.0
    raw_response: str = ""

    def to_payload(self) -> dict:
        """Serialise for Qdrant payload on the DAG side."""
        return {
            "summary": self.summary,
            "discarded_ids": [str(i) for i in self.discarded_ids],
            "contradictions": [
                {"id_a": str(c.id_a), "id_b": str(c.id_b), "explanation": c.explanation} for c in self.contradictions
            ],
            "confidence": round(max(0.0, min(1.0, self.confidence)), 4),
        }


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_confidence(raw: Any) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN guard
        return 0.0
    return max(0.0, min(1.0, v))


def _coerce_string(raw: Any, cap: int = 2000) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.strip()[:cap]


def _coerce_id_list(raw: Any, valid_ids: set) -> list[Any]:
    """Filter LLM-proposed ids down to the input member id set.

    Hallucinated ids are silently dropped — safer than trusting
    the LLM to echo UUIDs perfectly. `valid_ids` contains both
    string and original representations so str/int ambiguities
    round-trip correctly.
    """
    if not isinstance(raw, list):
        return []
    out: list[Any] = []
    for entry in raw:
        if entry in valid_ids:
            out.append(entry)
        elif isinstance(entry, (int, float)) and str(entry) in valid_ids:
            out.append(str(entry))
        elif isinstance(entry, str) and entry in valid_ids:
            out.append(entry)
    return out


def _coerce_contradictions(raw: Any, valid_ids: set) -> list[Contradiction]:
    if not isinstance(raw, list):
        return []
    out: list[Contradiction] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        a = entry.get("id_a")
        b = entry.get("id_b")
        explanation = entry.get("explanation") or entry.get("reason") or ""
        if a not in valid_ids or b not in valid_ids:
            continue
        if not isinstance(explanation, str):
            explanation = str(explanation)
        out.append(Contradiction(id_a=a, id_b=b, explanation=explanation.strip()[:240]))
    return out


def _parse_response(text: str, valid_ids: set) -> ConsolidationResult:
    raw = (text or "").strip()
    if not raw:
        return ConsolidationResult(summary="", confidence=0.0, raw_response="")

    parsed: dict | None = None
    candidates = [raw] + [m.group(0) for m in _JSON_OBJECT_RE.finditer(raw)]
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            parsed = data
            break
    if parsed is None:
        logger.debug("consolidation response unparseable: %s", raw[:120])
        return ConsolidationResult(summary="", confidence=0.0, raw_response=raw[:2000])

    summary = _coerce_string(parsed.get("summary"), cap=2000)
    discarded = _coerce_id_list(parsed.get("discarded_ids"), valid_ids)
    contradictions = _coerce_contradictions(parsed.get("contradictions"), valid_ids)
    confidence = _coerce_confidence(parsed.get("confidence"))
    return ConsolidationResult(
        summary=summary,
        discarded_ids=discarded,
        contradictions=contradictions,
        confidence=confidence,
        raw_response=raw[:2000],
    )


def summarise_prompt(
    members: list[dict],
    structural_edges: list[dict] | None = None,
) -> str:
    """Build the prompt for a consolidation call.

    Kept standalone so tests can pin on its shape and the atlas
    formulas catalog can render it verbatim alongside the
    classify_edge prompt.
    """
    lines = [
        "You are consolidating a tight cluster of closely-related memories",
        "from a personal knowledge base into a single summary note.",
        "Write ONE paragraph that:",
        "- preserves the user's own vocabulary and voice",
        "- does NOT invent facts; if members disagree, say so explicitly",
        "- captures what is common across members + what is novel in any one",
        "- stays under 1000 characters",
        "Then identify which input ids are FULLY captured by the summary",
        "(safe to archive) versus which still carry unique signal and must",
        "stay live. Flag contradictions as pairs.",
        "",
        "Return STRICT JSON:",
        "{",
        '  "summary": "<paragraph>",',
        '  "discarded_ids": ["<id>", ...],',
        '  "contradictions": [{"id_a": "<id>", "id_b": "<id>", "explanation": "..."}],',
        '  "confidence": 0.0-1.0',
        "}",
        "",
        "Cluster members:",
    ]
    for m in members:
        mid = m.get("id")
        text = (m.get("text") or "")[:MEMBER_TEXT_CAP]
        lines.append(f"- id={mid}: {text}")

    if structural_edges:
        lines.append("")
        lines.append("Already-known relationships between members:")
        for e in structural_edges:
            src = e.get("source_id")
            dst = e.get("id")
            kind = e.get("kind") or "?"
            lines.append(f"- {src} -[{kind}]-> {dst}")

    lines.append("")
    lines.append("JSON:")
    return "\n".join(lines)


def consolidate_clique(
    members: Iterable[dict],
    llm_client: LlmClient,
    *,
    model: str,
    structural_edges: list[dict] | None = None,
) -> ConsolidationResult:
    """Summarise a clique's members into one coherent note.

    Returns a ConsolidationResult on every path — LLM exceptions,
    empty responses, unparseable JSON all collapse to "empty
    summary, confidence 0" so the caller writes nothing and tries
    again next cycle without branching. The LLM is never told about
    the confidence auto-commit threshold; the DAG enforces that
    policy based on the returned value.
    """
    mat = [m for m in members if isinstance(m, dict) and m.get("id") is not None]
    if len(mat) < 2:
        return ConsolidationResult(summary="", confidence=0.0, raw_response="insufficient members")
    if len(mat) > MAX_MEMBERS:
        logger.debug("clamping clique members %d -> %d", len(mat), MAX_MEMBERS)
        mat = mat[:MAX_MEMBERS]

    # Build a valid-id set that accepts the original + string form
    # so the LLM's round-tripped ids (often str(int)) resolve.
    valid_ids: set = set()
    for m in mat:
        i = m.get("id")
        valid_ids.add(i)
        valid_ids.add(str(i))

    prompt = summarise_prompt(mat, structural_edges=structural_edges)
    try:
        response = llm_client(prompt, model) or ""
    except Exception as e:  # noqa: BLE001
        logger.warning("consolidate_clique llm_client raised: %s", e)
        return ConsolidationResult(summary="", confidence=0.0, raw_response=f"llm error: {e}"[:200])

    return _parse_response(response, valid_ids)
