"""Patch live_trade_mt5.py to wire the SL/TP dual-mode startup guard.

Two targeted replacements (CRLF-safe, exact-match, asserts each hit):
1. Extend the `from core.config import (...)` block with
   `enforce_sltp_compatibility`.
2. Insert `enforce_sltp_compatibility(cfg)` at the TOP of
   `enforce_model_promotion_gate` so demo + live startup paths both enforce
   the guard (it raises ValueError on mismatch — never a silent discard).
"""

import sys
from pathlib import Path

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
TARGET = REPO / "live" / "live_trade_mt5.py"

text = TARGET.read_text(encoding="utf-8")

# --- replacement 1: extend core.config import block ------------------------- #
OLD_IMPORT = """from core.config import (
    AppConfig,
    TradingMode,
    load_config,
)"""
NEW_IMPORT = """from core.config import (
    AppConfig,
    TradingMode,
    enforce_sltp_compatibility,
    load_config,
)"""

# --- replacement 2: guard call at top of enforce_model_promotion_gate ------- #
OLD_GATE = """def enforce_model_promotion_gate(cfg: AppConfig) -> None:
    if cfg.model.signal_source != "model":
        return"""
NEW_GATE = """def enforce_model_promotion_gate(cfg: AppConfig) -> None:
    # SL/TP dual-mode guard first: a model trained with SL/TP actions MUST run
    # in SLTP_MODE=model (and vice versa).  Raises ValueError on mismatch —
    # model-decided SL/TP output is NEVER silently discarded.
    enforce_sltp_compatibility(cfg)
    if cfg.model.signal_source != "model":
        return"""

n_import = text.count(OLD_IMPORT)
n_gate = text.count(OLD_GATE)
if n_import != 1:
    raise SystemExit(f"FATAL: expected exactly 1 import block hit, got {n_import}")
if n_gate != 1:
    raise SystemExit(f"FATAL: expected exactly 1 gate-body hit, got {n_gate}")

text = text.replace(OLD_IMPORT, NEW_IMPORT)
text = text.replace(OLD_GATE, NEW_GATE)

TARGET.write_text(text, encoding="utf-8")
print(f"patched {TARGET.name}: import block (1 hit), gate body (1 hit)")
