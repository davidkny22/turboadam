from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

TRITON_CACHE = SRC.parent / ".tmp" / "triton_test_cache"
TRITON_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TRITON_CACHE_DIR", str(TRITON_CACHE))
