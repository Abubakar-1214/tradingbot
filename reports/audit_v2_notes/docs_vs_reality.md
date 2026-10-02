# Audit v2 — Docs vs Reality Discrepancy Table

Compare docs claims against verified code + artifact state (as of this audit).

## Headline honesty status

**The NEW documentation set is internally HONEST.** All four docs explicitly state the system is research-stage with NO trained model shipped:
- `README.md` — "STATUS: RESEARCH / NOT PRODUCTION READY"
- `docs/MODELS.md` — "koi trained model ship nahin hota" (no trained model ships)
- `docs/TRAINING_GUIDE.md` — "No trained model is shipped… Passing tests… does not establish future performance"
- `reports/MODELS_REPORT.md` — "Code, wiring, tests aur docs complete hain. Koi model abhi trained nahi hai"

No unsupported "Phase complete / production-ready / trained model available" claim exists in any new doc.

## Discrepancy table

| # | Doc claim | Reality | Severity |
|---|---|---|---|
| D1 | README: ".venv in this repo is Python 3.11" | venv actually runs Python 3.13.12 (pytest platform header, verified this session) | LOW — cosmetic doc error; venv works |
| D2 | README Verified Facts: "139 passed" (full suite) and "87 passed" (`pytest tests env -q`) | This audit re-ran the 7 new/rewritten suites: **78 passed / 23.07s / 0 failures**. The 139/87 totals plausibly cover the wider suite (env/*, risk, broker, position_sizing, backtest) which was NOT independently re-run in this window. | LOW — plausible, not verified-here; recommend full-suite re-run at production time |
| D3 | TRAINING_GUIDE/README imply data CSVs (`data/xauusd_h1.csv`, `data/xauusd_d1.csv`) are user-provided | `data/xauusd_h1.csv` EXISTS on disk; `data/xauusd_d1.csv` does NOT. MODELS_REPORT explicitly says verify_features/verify_backtest could not run for this reason — consistent. | NONE — docs honest |
| D4 | MODELS_REPORT: "139 tests pass" | 78 (7 suites) verified green here; 139 claim consistent with full suite, not re-run. | LOW (same as D2) |
| D5 | docs describe per-model capabilities (ensemble padding, MCTS info keys, first-order MAML, adversarial clean-eval, epistemic KL caveat) | All descriptions MATCH the code read in this audit (see catalog.md) | NONE — docs accurate |
| D6 | docs describe promotion gate semantics (demo warn vs live refuse, evaluation.json passed + contract_hash) | Code verified: `live/live_trade_mt5.py` `model_promotion_failures` + `enforce_model_promotion_gate` match exactly | NONE |
| D7 | docs claim "no trained model shipped" | TRUE — 0 checkpoints repo-wide: no `artifacts/models/` dir, no `train/*.pt|.zip|.pth|.onnx`, no model files outside `.venv` | NONE — accurate |
| D8 | README backtest verdict table: "all strategies LOSE to buy-and-hold net of costs (e.g. SmaCrossAtr 52.40% vs 884.26%)" | Consistent with `reports/backtesting_py_research_report.md` on disk; not re-run in this audit | LOW — not re-run here |

## Unsupported "Phase complete" claims

None found. The closest claims are "interfaces implemented and tested" (README) and "Code, wiring, tests aur docs complete hain" (MODELS_REPORT) — both of which are TRUE per this audit (all code paths exist, all 7 targeted suites green, wiring end-to-end verified).

**Conclusion: docs are honest and match reality. Two cosmetic/low-severity items (D1 .venv Python version claim; D2/D4 full-suite counts unverified in this window).**
