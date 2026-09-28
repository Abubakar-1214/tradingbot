"""CRLF-safe targeted patch for live/live_trade_mt5.py (edit_file fails on CRLF).

Inserts a sys.path bootstrap so ``python live/live_trade_mt5.py`` resolves
``core``/``live``/``models`` regardless of CWD.  Idempotent: if the bootstrap
is already present it is a no-op.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "live" / "live_trade_mt5.py"

BOOTSTRAP = (
    "import sys\n"
    "import time\n"
    "from datetime import datetime, timedelta, timezone\n"
    "from pathlib import Path\n"
    "from typing import Dict, List, Optional, Protocol\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "\n"
    "# Allow direct execution from any CWD (``python live/live_trade_mt5.py``):\n"
    "# put the repo root on sys.path so ``core`` / ``live`` / ``models`` resolve.\n"
    "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n"
    "\n"
    "from core.config import ("
)

text = TARGET.read_text(encoding="utf-8")
old = (
    "import time\n"
    "from datetime import datetime, timedelta, timezone\n"
    "from pathlib import Path\n"
    "from typing import Dict, List, Optional, Protocol\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "\n"
    "from core.config import ("
)

if "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))" in text:
    print("[OK] bootstrap already present — no-op")
else:
    if old not in text:
        raise SystemExit(f"[FAIL] expected import block not found in {TARGET}")
    new = text.replace(old, BOOTSTRAP, 1)
    TARGET.write_text(new, encoding="utf-8")
    print("[OK] bootstrap inserted")

# verify
check = TARGET.read_text(encoding="utf-8")
assert "import sys" in check and "sys.path.insert(0" in check
print("[OK] verify: sys.path bootstrap present")
