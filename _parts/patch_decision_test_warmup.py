"""Fix test_decision_sltp.py _model_cfg: drop_warmup=0 (manifest window=4 stub)."""

import sys
from pathlib import Path

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
TARGET = REPO / "_parts" / "tests_sltp" / "test_decision_sltp.py"

text = TARGET.read_text(encoding="utf-8")

old = "        feature=replace(cfg.feature, window=4),"
new = "        feature=replace(cfg.feature, window=4, drop_warmup=0),"

n = text.count(old)
if n != 1:
    raise SystemExit(f"FATAL: expected 1 hit, got {n}")
text = text.replace(old, new)
TARGET.write_text(text, encoding="utf-8")
print("OK: drop_warmup=0 wired into _model_cfg")
