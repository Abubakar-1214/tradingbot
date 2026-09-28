"""CRLF-safe append of completion status to plans/plan.md (append-only)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PLAN = ROOT / "plans" / "plan.md"

section = """

## Completion Status (recorded 2026-09-28)

All subtasks below are complete and verified. Evidence artifacts in `artifacts/`.

| # | Subtask | Status | Evidence |
|---|---|---|---|
| 1 | Config layer (core/config.py, .env loading, TRADING_MODE gate) | DONE | `scripts/verify_config.py` 24/24 x2, exit 0; `artifacts/gate_evidence.txt` |
| 2 | Leak-free feature pipeline (causal higher-TF/macro, train-window scaler, feature_contract.json) | DONE | `scripts/verify_features.py` 33/33 x2, exit 0; `artifacts/gate_evidence.txt` |
| 3 | RiskSupervisor fix (explicit size API, SQLite persistence, config correlation guard) | DONE | `scripts/verify_risk.py` 29/29 x2, exit 0; `artifacts/gate_evidence.txt` |
| 4 | PositionSizing fix (risk-based sizing + dynamic_sizing rssm.observe 4-value fix) | DONE | `scripts/_verify_p1_fixes.py` 6/6 dynamic_sizing; `artifacts/pytest_final.txt` 87/87 |
| 5 | Tests (feature causality, contract, risk persistence, sizing, broker, loop smoke) | DONE | `artifacts/pytest_final.txt` — 87 passed, exit 0 |
| 6 | backtesting.py install + introspection | DONE | `backtest/BACKTESTING_API_NOTES.md`, `backtest/README.md`; `backtesting==0.6.2` pinned |
| 7 | NEW backtest package (strategies/costs/engine/walk_forward/baselines/report) | DONE | `backtest/` package; fake engine archived to `archive/backtest_engine_legacy_fake.py` |
| 8 | Real-data verification (deterministic, costs, walk-forward, baselines) | DONE | `backtest/report_backtest.md` (honest: strategies do NOT beat buy-and-hold); `scripts/verify_backtest.py` 26/26 x2 |
| 9 | Broker abstraction (BaseBroker/Mt5Broker/MockBroker) | DONE | `scripts/verify_broker.py` 29/29 x2, exit 0; `artifacts/gate_evidence.txt` |
| 10 | Live loop rewrite (candle-close-aligned, TRADING_MODE gate, SL/TP, retcodes, reconciliation, kill switch) | DONE | `artifacts/live_demo_smoke.txt` — exit 0, fills=10 genuine order cycle |
| 11 | Wiring (RiskSupervisor + PositionSizer before every order, long+short, state persistence) | DONE | `scripts/verify_risk_integration.py` 23/23 x2, exit 0 |
| 12 | Ops artifacts (pinned requirements.txt, .env.example, README, gitignore, smoke test) | DONE | `artifacts/ops_evidence.txt` — pip check clean, 49 env keys documented, git check-ignore PASS |

Deliverables: `FIXES.md` (root, maps all 12 P0 + P1s to fixes + evidence),
`artifacts/final_summary.txt` (final integrity sweep), `artifacts/ops_evidence.txt`,
`artifacts/gate_evidence.txt`, `artifacts/pytest_final.txt`, `artifacts/live_demo_smoke.txt`.

Known limitations (honest): no GPU -> no Dreamer/RL retraining; ML strategy path
requires a trained checkpoint (absent) and refuses to fabricate results; MT5
terminal verification is user-side; strategies underperform buy-and-hold net of
costs (see `backtest/report_backtest.md`) — go-live gates require material
strategy improvement first.
"""

with PLAN.open("a", encoding="utf-8") as f:
    f.write(section)
print(f"[OK] appended completion status to plans/plan.md ({PLAN.stat().st_size} bytes)")
