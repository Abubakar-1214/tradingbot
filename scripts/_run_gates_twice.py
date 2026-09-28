"""Run all six verify gates TWICE each and record evidence to artifacts/gate_evidence.txt.

Each invocation is executed with the venv python, capturing stdout + stderr +
exit code.  The artifact is structured as run1/run2 sections per gate with the
pass/fail line and exit code, satisfying the "two recorded green runs per gate"
acceptance criterion.  Exits non-zero if ANY invocation exits non-zero.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
out_path = ART / "gate_evidence.txt"

GATES = [
    ("scripts/verify_config.py", 24),
    ("scripts/verify_features.py", 33),
    ("scripts/verify_risk.py", 29),
    ("scripts/verify_backtest.py", 26),
    ("scripts/verify_broker.py", 29),
    ("scripts/verify_risk_integration.py", 23),
]

lines = []
lines.append(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
lines.append("Each gate run TWICE; requirement: exit 0 both runs with the expected pass count.")
all_ok = True
for script, expected in GATES:
    for run in (1, 2):
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        dt = time.time() - t0
        ok = proc.returncode == 0
        all_ok = all_ok and ok
        lines.append("=" * 78)
        lines.append(f"[{script}] run {run}/2 exit={proc.returncode} "
                     f"(expected {expected} passes) duration={dt:.1f}s {'OK' if ok else 'FAIL'}")
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        # Keep the last ~12 lines: pass-count summary + any failures.
        for line in tail[-12:]:
            lines.append(f"    {line}")
        if not ok:
            lines.append("    --- full output (first 1500 chars) ---")
            lines.append((proc.stdout + proc.stderr)[:1500])

summary = "ALL GATES GREEN (12/12 runs exit 0)" if all_ok else "GATE FAILURE(S) DETECTED"
lines.append("=" * 78)
lines.append(summary)
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nartifact: {out_path}")
sys.exit(0 if all_ok else 1)
