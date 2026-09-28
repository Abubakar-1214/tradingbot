"""pytest bootstrap — put the repo root and this tests dir on sys.path.

The verify_* gates are run as scripts with an explicit sys.path.insert; pytest
needs the same so every test can ``from core.config import ...`` etc.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent

for _p in (REPO, TESTS):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
