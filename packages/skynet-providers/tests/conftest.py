"""Shared fixtures for skynet-providers tests.

Makes the in-worktree package importable so edits in ``src/`` are
tested instead of whatever is in site-packages. Mirrors the conftest
already shipped by ``skynet-matrix``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Prefer the in-tree skynet-matrix over the installed wheel so the SSE
# streaming hooks (``emit_llm_start_if_live`` etc., added in 2026.4.24.5)
# are actually importable when chat.py does its optional-import dance.
_MATRIX_SRC = _SRC.parent.parent / "skynet-matrix" / "src"
if _MATRIX_SRC.exists() and str(_MATRIX_SRC) not in sys.path:
    sys.path.insert(0, str(_MATRIX_SRC))
