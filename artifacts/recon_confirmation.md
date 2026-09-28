# Phase 0 Recon Confirmation — Cycle `exec_full_plan_dd403fbf`

Source of truth: `plans/plan.md` (76 lines, read in full) and `research/_audit_report_text.txt` (241 lines, all 12 P0 + P1 items enumerated).

## (a) plan.md subtasks still OPEN this cycle

| Subtask | Current state |
|---|---|
| Fix 2 remaining bugs in `tests/test_executor.py` (full rewrite via write_file) | OPEN — baseline pytest: `84 passed, 3 failed` |
| Full pytest suite → exit 0 | OPEN — 3 failures: `test_all_order_paths_call_check_trade` (AttributeError `ex.risk.now_fn`), `test_three_candle_full_loop_smoke` (ATR_UNAVAILABLE, 10 bars < 15), `test_atr_sizing_larger_atr_smaller_position` (test formula wrong) |
| Six verify gates × 2 runs each (exit 0 both) | RUN 1 ALREADY GREEN: config 24/24, features 33/33, risk 29/29, backtest 26/26, broker 29/29, risk_integration 23/23 — run 2 + recorded evidence pending |
| `live/live_trade_mt5.py` production rewrite (new stack) | OPEN — file is still the old 173-line version (VOLUME=0.01 hardcoded, no SL/TP, no RiskSupervisor import, 10s loop) |
| `models/position_sizing.py` P1 fix (`rssm.observe()` wrong args → value_long=value_flat / advantage 0 / win_prob 0.5) | OPEN — `dynamic_sizing()` still broken |
| Dreamer `save()` state (optimizer/RNG/normalizer) | OPEN — `models/dreamer_agent.py` `save()` persists only network weights + training_step |
| ReplayBuffer episode bounds | OPEN — `models/dreamer_agent.py` `ReplayBuffer.sample()` can span episode boundaries (no done-mask filtering) |
| `evaluate_model.py` annualization + ylim | OPEN — uses `252*24*12` (5-min bars hardcoded), `ylim(-0.1, 1.1)` hides shorts |
| Ops artifacts: pinned requirements.txt, .env loading + .env.example, README status, .gitignore | OPEN — requirements.txt unpinned (`>=`), no `.env`, README still has inflated claims, `.gitignore` lacks `state//` + `logs/` |
| `FIXES.md` | OPEN — does not exist |

## (b) P1 items ALREADY fixed in earlier cycles (DO NOT re-do — gate-verified)

| P1 item | Fix evidence |
|---|---|
| `calendar_features.py` O(N×E) loop → vectorized searchsorted | `scripts/verify_features.py` section [4] PASS (`calendar_features: vectorized searchsorted`, `no per-row python loop`) |
| `sentiment_analysis.py` hard-coded 0.0 placeholder | `scripts/verify_features.py` section [4] PASS (`sentiment: no hard-coded 0.0 placeholder`) |
| `timeframe_features.py` requires `volume` but MT5 gives `tick_volume` | `scripts/verify_features.py` section [4] PASS (`timeframe_features: tick_volume support`, `volume OR tick_volume guard`) |
| Feature contract (feature_names + scaler saved, mismatch raises) | `scripts/verify_features.py` section [3] PASS (contract write/reload, mismatch/extra/reorder raises) |
| `multi_timeframe.py` higher-TF resample label/closed right + shift(1) | `scripts/verify_features.py` section [4] PASS (`multi_timeframe: label='right'`, `closed='right'`, `shift(1) after resample`) |
| Macro daily shift(1) | `scripts/verify_features.py` section [4] PASS (`macro: daily shift(1) before ffill`) |
| `make_features.py` full-dataset normalization | `scripts/verify_features.py` section [4] PASS (`make_features: no full-dataset normalisation`) |

## (c) The six verify gates + expected pass counts

| Script | Expected passes | Run 1 (recorded above) |
|---|---|---|
| `scripts/verify_config.py` | 24 | 24 passed, exit 0 |
| `scripts/verify_features.py` | 33 | 33 passed, exit 0 |
| `scripts/verify_risk.py` | 29 | 29 passed, exit 0 |
| `scripts/verify_backtest.py` | 26 | 26 passed, exit 0 |
| `scripts/verify_broker.py` | 29 | 29 passed, exit 0 |
| `scripts/verify_risk_integration.py` | 23 | 23 passed, exit 0 |

Run 2 for every gate plus a machine-readable evidence log (`artifacts/gate_evidence.txt`) is produced in the gate-evidence subtask.
