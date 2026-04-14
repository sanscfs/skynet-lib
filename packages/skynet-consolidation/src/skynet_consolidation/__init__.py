"""Skynet Consolidation — merge mature cliques into summary points.

Phase 5 of docs/rag-memory-roadmap.md. The similarity + classification
pipelines (Phase 2 + 4) detect when multiple memories say the same
thing. This package is the *decision layer* that turns that signal
into action: reduce a tight clique of raw memories to a single
well-worded summary point, preserving provenance so nothing is
actually lost.

Design constraints:

- Stateless and backend-agnostic. Callers inject the LLM client
  (`llm_client(prompt, model) -> str`) exactly like the Phase 4
  classify_edge primitive — same deployment pattern (OpenRouter via
  `skynet.airflow_helpers.llm_chat` in DAGs, Ollama + fallback in
  services).
- Every call returns a `ConsolidationResult` dataclass even on LLM
  failure, so the DAG never branches on exceptions.
- LLM response JSON parser is forgiving: markdown fences, prose
  prefixes, unknown keys all degrade to a conservative "low-
  confidence, no discards" result.

Public API:

    from skynet_consolidation import (
        ConsolidationResult,
        Contradiction,
        consolidate_clique,
    )
"""

from skynet_consolidation.consolidate import (
    ConsolidationResult,
    Contradiction,
    consolidate_clique,
)

__all__ = [
    "ConsolidationResult",
    "Contradiction",
    "consolidate_clique",
]
