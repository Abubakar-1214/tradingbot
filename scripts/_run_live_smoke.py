"""Run the demo-mode smoke of live/live_trade_mt5.py and record evidence.

Executes ``python live/live_trade_mt5.py --smoke --max-bars 300`` with the
venv python, captures stdout + stderr + exit code to
``artifacts/live_demo_smoke.txt``, and prints the tail.

Smoke isolation (deterministic, disposable): RISK_STATE_DB / LOCAL_STATE_FILE /
LOG_DIR / KILL_SWITCH_PATH are redirected to ``artifacts/_smoke_state`` and
cleaned before each run so a prior smoke can never poison the next one.
ALLOW_SHORT=true and MIN_TRADE_INTERVAL_SEC=0 are set so the oscillating
synthetic feed can demonstrate the FULL cycle (long entry -> close -> short
entry) without the 300s cooldown breaker suppressing re-entries (the breaker
itself is verified separately by scripts/verify_risk.py).

The smoke must show a GENUINE order cycle flowing through the new stack
(RiskSupervisor-gated entry with real ATR-based SL/TP on a MockBroker fill,
then close/SL/TP with real PnL) — no canned/fabricated order results.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
out_path = ART / "live_demo_smoke.txt"
smoke_state = ART / "_smoke_state"

# Fresh isolated state for every smoke run.
if smoke_state.exists():
    shutil.rmtree(smoke_state)
smoke_state.mkdir(parents=True, exist_ok=True)

env = dict(os.environ)
env["TRADING_MODE"] = "demo"
env["ALLOW_SHORT"] = "true"
env["MIN_TRADE_INTERVAL_SEC"] = "0"
env["RISK_STATE_DB"] = str(smoke_state / "risk_state.db")
env["LOCAL_STATE_FILE"] = str(smoke_state / "bot_state.json")
env["LOG_DIR"] = str(smoke_state / "logs")
env["KILL_SWITCH_PATH"] = str(smoke_state / "KILL_SWITCH")
env["SIGNAL_SOURCE"] = "rule"

t0 = time.time()
proc = subprocess.run(
    [sys.executable, str(ROOT / "live" / "live_trade_mt5.py"),
     "--smoke", "--max-bars", "300"],
    cwd=str(ROOT),
    capture_output=True,
    text=True,
    env=env,
)
dt = time.time() - t0
combined = proc.stdout + proc.stderr
out_path.write_text(
    f"command: {sys.executable} live/live_trade_mt5.py --smoke --max-bars 300\n"
    f"exit_code: {proc.returncode}\n"
    f"duration_sec: {dt:.1f}\n"
    f"--- output ---\n{combined}\n",
    encoding="utf-8",
)
print(f"exit_code={proc.returncode} duration={dt:.1f}s artifact={out_path}")
print(combined[-5000:])
sys.exit(proc.returncode)
