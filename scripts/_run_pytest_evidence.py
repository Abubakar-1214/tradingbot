"""Run the full pytest suite and record evidence to artifacts/pytest_final.txt.

Executes ``python -m pytest tests env -q`` from the repo root, captures stdout +
stderr + exit code into a single artifact file for FIXES.md evidence, and prints
the tail so the session log shows the result.  Exits with pytest's exit code.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
out_path = ART / "pytest_final.txt"

t0 = time.time()
proc = subprocess.run(
    [sys.executable, "-m", "pytest", "tests", "env", "-q"],
    cwd=str(ROOT),
    capture_output=True,
    text=True,
)
dt = time.time() - t0
combined = proc.stdout + proc.stderr
out_path.write_text(
    f"command: {sys.executable} -m pytest tests env -q\n"
    f"exit_code: {proc.returncode}\n"
    f"duration_sec: {dt:.1f}\n"
    f"--- output ---\n{combined}\n",
    encoding="utf-8",
)
print(f"exit_code={proc.returncode} duration={dt:.1f}s artifact={out_path}")
print(combined[-2500:])
sys.exit(proc.returncode)
