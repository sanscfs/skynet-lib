"""
classify.py -- LLM-powered typed-edge classification.

Phase 4 of docs/rag-memory-roadmap.md. The `similar_ids` graph built
by the nightly DAG in Phase 2 only knows cosine similarity -- point B
is "somewhat like" point A. That's enough for neighbourhood retrieval
but hides the semantic relationship: is B a newer version of A
(supersedes), a counter-example (contradicts), a deeper explanation
(elaborates), something A caused (caused_by), or a concrete instance
of A's general claim (example_of)?

This module calls a caller-supplied LLM to classify one pair at a
time and returns a structured result with a confidence score and a
one-line reason. The library stays backend-agnostic -- callers
inject an `llm_client(prompt, model) -> str` callable that already
knows how to talk to Ollama / OpenRouter / a stub in tests. The
output goes into a SEPARATE payload field (`structural_edges`) from
`similar_ids`, so the base similarity graph stays pristine and
Phase 4 can be rolled back by just ignoring the new field.

Design notes:

- Pair direction matters. "A supersedes B" is NOT the same as "B
  supersedes A". Callers classify the *directed* pair and write the
  edge to the source's payload with a target pointer. Reversed
  relationships get their own classification call.
- The LLM may legitimately say "none of these apply" -- the pair
  is similar but not structurally related. We expose a sixth
  category `RELATED` (= fallback, only similarity) so calls return
  a result for every pair, and the caller filters if they only
  want typed edges.
- Low-confidence classifications are kept in the output but the
  caller is expected to honour a `min_confidence` gate before
  writing them to payload. We don't apply the gate in this library
  because payload policy is a deployment concern.
- Prompt is small and deterministic: the LLM just produces strict
  JSON. Non-JSON outputs and unknown kinds both degrade to
  `RELATED` with confidence 0.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Literal

from skynet_graph.similarity import EdgeKind

logger = logging.getLogger(__name__)


#: Kinds that the LLM is allowed to emit. `related` is the fallback
#: for "similar but no typed relationship" -- keeping it in the enum
#: means every call returns a valid kind and callers don't need to
#: special-case parse failures.
STRUCTURAL_KINDS: tuple[str, ...] = (
    EdgeKind.SUPERSEDES,
    EdgeKind.CONTRADICTS,
    EdgeKind.ELABORATES,
    EdgeKind.CAUSED_BY,
    EdgeKind.EXAMPLE_OF,
    "related",
)

#: Sentinel returned when the LLM can't decide. Not in the set of
#: "structural" kinds (see STRUCTURAL_EDGE_KINDS in traversal.py) so
#: callers filtering with that set drop the RELATED edges naturally.
RELATED_KIND = "related"


@dataclass(slots=True, frozen=True)
class ClassifiedEdge:
    """One LLM classification result.

    `kind` is one of STRUCTURAL_KINDS. `confidence` is [0.0, 1.0] --
    the LLM's self-reported confidence (parsed from the JSON response,
    clamped). `reason` is a one-line human-readable justification,
    useful for debugging in the atlas UI and for the Matrix approval
    flow when high-stakes edges need owner review.
    """

    kind: str
    confidence: float
    reason: str

    def to_payload(self) -> dict:
        """Serialise for Qdrant payload. Matches SimilarityEdge shape
        so the traversal code can walk both fields uniformly: reader
        only needs `id` + `kind` to decide how to weight the edge.
        """
        return {
            "kind": self.kind,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


# JSON-in-free-text extraction regex. DeepSeek / Qwen occasionally
# wrap the JSON in markdown fences or prefix it with "Here's the
# classification:" — we tolerate that rather than demanding pristine
# output.
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _coerce_kind(raw: object) -> str:
    """Map any LLM-output value to one of STRUCTURAL_KINDS or `related`.

    Accepts the six valid strings (case-insensitively). Anything else
    -- missing key, unknown string, wrong type, None -- collapses to
    `related` so the caller always gets a usable kind.
    """
    if not isinstance(raw, str):
        return RELATED_KIND
    s = raw.strip().lower()
    for k in STRUCTURAL_KINDS:
        if s == k:
            return k
    return RELATED_KIND


def _coerce_confidence(raw: object) -> float:
    """Clamp LLM confidence to [0.0, 1.0]. Missing/garbage -> 0.0."""
    try:
        v = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN guard
        return 0.0
    return max(0.0, min(1.0, v))


def _parse_response(text: str) -> ClassifiedEdge:
    """Turn the LLM's raw response text into a ClassifiedEdge.

    Strategy:
      1. Try full-string JSON parse.
      2. Fall back to a regex that extracts the first {...} block and
         retries JSON parsing. Handles responses that wrap the object
         in markdown or prose.
      3. Any failure -> RELATED with confidence 0 and a reason that
         preserves the raw response head so debugging is possible.
    """
    raw = (text or "").strip()
    if not raw:
        return ClassifiedEdge(kind=RELATED_KIND, confidence=0.0, reason="empty response")

    parsed: dict | None = None
    for candidate in (raw, *(m.group(0) for m in _JSON_OBJECT_RE.finditer(raw))):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            parsed = data
            break
    if parsed is None:
        return ClassifiedEdge(
            kind=RELATED_KIND,
            confidence=0.0,
            reason=f"unparseable: {raw[:80]}",
        )

    kind = _coerce_kind(parsed.get("kind"))
    confidence = _coerce_confidence(parsed.get("confidence"))
    reason = parsed.get("reason") or ""
    if not isinstance(reason, str):
        reason = str(reason)
    reason = reason.strip()[:200]

    return ClassifiedEdge(kind=kind, confidence=confidence, reason=reason)


def _build_prompt(point_a_text: str, point_b_text: str) -> str:
    """Build the classification prompt.

    Kept as a small standalone function so tests can assert on the
    prompt structure and the atlas formulas catalog can render it
    verbatim under /formulas.
    """
    return (
        "You classify the RELATIONSHIP between two notes from a personal "
        "knowledge base. Respond with strict JSON and nothing else. "
        "Allowed kinds:\n"
        "- supersedes: A is a newer, more correct version of B; if forced "
        "to keep one, keep A\n"
        "- contradicts: A and B make incompatible claims\n"
        "- elaborates: A deepens, explains, or extends B without replacing it\n"
        "- caused_by: A describes something that happened because of B\n"
        "- example_of: A is a concrete instance of a general claim in B\n"
        "- related: A and B share topic but none of the above applies\n"
        "\n"
        "Return exactly this JSON shape:\n"
        '{"kind": "...", "confidence": 0.0-1.0, "reason": "<one line>"}\n'
        "\n"
        f"A: {point_a_text.strip()[:1200]}\n"
        f"B: {point_b_text.strip()[:1200]}\n"
        "JSON:"
    )


LlmClient = Callable[[str, str], str]
"""Signature of the caller-supplied LLM client.

Takes `(prompt, model)` and returns the raw text. Should NOT raise --
callers are expected to wrap their HTTP client with a broad
try/except and return "" on failure. An empty return is treated as
"classification not available" and yields RELATED with confidence 0.
"""


def classify_edge(
    point_a_text: str,
    point_b_text: str,
    llm_client: LlmClient,
    *,
    model: str,
) -> ClassifiedEdge:
    """Classify the directed A→B relationship between two notes.

    Returns a ClassifiedEdge in every case -- failures and
    "just similar" both collapse to kind=RELATED so the caller can
    pass the result directly to payload-write code without branching.

    Empty / whitespace-only inputs short-circuit to RELATED with
    confidence 0 (there's nothing to classify).
    """
    a = (point_a_text or "").strip()
    b = (point_b_text or "").strip()
    if not a or not b:
        return ClassifiedEdge(kind=RELATED_KIND, confidence=0.0, reason="empty input")

    prompt = _build_prompt(a, b)
    try:
        response = llm_client(prompt, model) or ""
    except Exception as e:  # noqa: BLE001
        logger.warning("classify_edge llm_client raised: %s", e)
        return ClassifiedEdge(kind=RELATED_KIND, confidence=0.0, reason=f"llm error: {e}"[:200])
    return _parse_response(response)


Direction = Literal["a_to_b", "b_to_a"]


def classify_pair_bidirectional(
    point_a_text: str,
    point_b_text: str,
    llm_client: LlmClient,
    *,
    model: str,
) -> dict[Direction, ClassifiedEdge]:
    """Classify both directions of a pair in two calls.

    Convenience wrapper for callers that want the full directed
    relationship (A→B AND B→A). Most kinds are asymmetric
    (supersedes, caused_by, example_of) so running one direction
    only miss half the edges. Cost doubles, so the DAG should use
    this sparingly on high-cos pairs.
    """
    return {
        "a_to_b": classify_edge(point_a_text, point_b_text, llm_client, model=model),
        "b_to_a": classify_edge(point_b_text, point_a_text, llm_client, model=model),
    }
