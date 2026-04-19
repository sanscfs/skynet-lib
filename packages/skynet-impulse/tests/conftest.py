"""Shared fixtures for skynet-impulse tests.

The engine/homeostat/baseline interact with Redis via the standard client's
hash/list/zset commands; the ``FakeRedis`` fixture below implements just
those ops to keep tests offline. We also register the ``test_detectors``
subdir on the Python path so pytest can discover it as a package.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Register the tests directory itself so ``test_detectors`` imports work
# without a full conftest hierarchy.
sys.path.insert(0, str(Path(__file__).parent))
