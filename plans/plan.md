# Trading Bot Production Readiness — Full Engineering Fix

## Goal
Fix every P0/P1 issue from `research/tradingbot_audit_report.html` and turn this repo into a production-grade trading system: a leak-free feature pipeline with a train/live contract, a correct risk supervisor wired into every path, a NEW real backtesting engine built on `kernc/backtesting.py` that produces honest, verifiable results on real XAUUSD data, and a production-ready MT5 live trading system with full safety (demo/live gate, SL/TP, order retcode handling, reconciliation, kill switch).

## Research Summary
(All findings verified via internet research + source reading before this plan.)
- **Audit report** (research/tradingbot_audit_report.html, extracted to research/_audit_report_text.txt) documents **12 P0 blockers**, ~35% production readiness:
  1. Live bot has NO risk control (no SL/TP, no daily loss limit, RiskSupervisor not even imported; `VOLUME=0.01` hardcoded in live/live_trade_mt5.py)
  2. `models/risk_supervisor.py` treats action index (0/1) as position fraction → `abs(action) > 0.10` → **every trade rejected** ("POSITION_TOO_LARGE"); state is memory-only (restart bypasses circuit breakers); correlation rule hardcoded
  3. Train (150+ features) vs live (~20 features) mismatch; normalization: train=full-dataset mean/std, live=last-200-bars → model sees unseen numbers
  4. Look-ahead bias in 3 places: higher-TF ffill leak (H1 close known at 10:00 M5 bar), macro daily close stamped 00:00 then ffill'd, full-dataset normalization
  5. `backtest/backtest_engine.py` is FAKE: `_get_observation()` returns `np.random.randn(100)`; `walk_forward_validation()` has training commented out
  6. Execution loop wrong: 10s loop on H1 strategy, incomplete-candle decisions, no order retcode checking/retry/partial fill/requote handling, no idempotency, no reconciliation, no shorts
  7. Config/deps broken: .env unused, `metaapi_cloud_sdk` missing from requirements, versions unpinned, README references missing files
- **backtesting.py API** (deepwiki.com/kernc/backtesting.py + kernc.github.io docs): `Strategy` subclass with `init()`/`next()`; `self.I()` for indicators; `self.buy(sl=, tp=)`, `self.sell(sl=, tp=)`, `self.position.close()`; `Backtest(data, strategy, cash=, commission=, margin=, trade_on_close=)`; `bt.run()` → Stats; `bt.optimize()`; plotting via `bt.plot()`. **Build-time task: verify installed version's exact signature via `help(Backtest)`/introspection** (custom broker / trade-on-close support varies by version).
- **MT5 Python API** (mql5.com): `mt5.order_send(request_dict)` with keys action/type/symbol/volume/price/deviation/magic/comment/type_time/type_filling/position/sl/tp; result.retcode with `TRADE_RETCODE_DONE=10009`, `REQUOTE=10004`, `REJECT=10006`, `CANCEL=10007`, `PLACED=10008`, `DONE_PARTIAL=10010`, `ERROR=10014`, `INVALID_VOLUME=10016`, `INVALID_PRICE=10018`, `INVALID_STOPS=10020`; use `mt5.order_check()` before send; `mt5.positions_get()`, `mt5.history_deals_get()`, `mt5.symbol_info_tick()`, `mt5.copy_rates_from_pos()`.
- **Walk-forward + purging/embargo** (López de Prado; purgedcv/wfvkit): purge samples whose labels overlap train/test boundary; embargo N bars after train window to kill autocorrelation leakage.
- **Risk best practices** (trading_bot_builder skill): 2% risk per trade, fractional Kelly (≤25%), circuit breakers (max daily loss, max drawdown, consecutive-loss breaker), ATR-based SL/TP, position sizing from stop distance, kill switch + alerts, paper-trade before live.

## Approach
Layered architecture with ONE shared code path for train/backtest/live:
`config → data → features (causal + saved contract) → strategy (rule-based or ML) → risk (RiskSupervisor + PositionSizer) → execution (backtesting.py engine OR MT5 broker adapter)`
- The feature pipeline is rewritten so the SAME function computes features everywhere; a saved `feature_contract.json` (feature_names + scaler stats + params hash) is loaded and validated at runtime; mismatch → refuse to start.
- Live trading locked to `TRADING_MODE=demo` by default; `live` mode refuses to start unless all gates pass.
- **User explicitly required kernc/backtesting.py for the new backtest engine — do NOT substitute another library.**
- MT5 terminal is NOT available in this sandbox → build the live module against the real `MetaTrader5` API surface but test it with a `MockBroker` that simulates fills/retcodes; document that final live verification needs the user's MT5 terminal.

## Subtasks

### Phase 1 — Core safety & correctness (do FIRST)
1. **Config layer**: `core/config.py` + `.env` loading (python-dotenv): SYMBOL, TIMEFRAME, VOLUME_MIN, MAX_RISK_PER_TRADE, MAX_DAILY_LOSS, MAX_DRAWDOWN, MAX_POSITIONS, MAX_CONSECUTIVE_LOSSES, TRADING_MODE (demo|live), SPREAD/SLIPPAGE/COMMISSION cost model, MODEL_PATH, FEATURE_CONTRACT_PATH, STATE_DB_PATH, LOG_DIR, ALLOW_SHORT. All hardcoded values in live scripts replaced by config. Provide validated dataclasses + `TRADING_MODE` gate function.
2. **Leak-free feature pipeline** (`features/` rewrite + `core/feature_pipeline.py`): causal higher-TF features (resample with `label='right', closed='right'` then `shift(1)` before merge so a bar only sees CLOSED higher-TF candles), macro daily `shift(1)`, rolling-window indicators only (no future), normalization fit on TRAIN window only, save `feature_contract.json` (feature_names, mean/std or scaler params, feature-params hash). Provide `compute_features(df, mode='train'|'infer', scaler=None)` returning `(X, feature_names)` and `load_feature_contract()` that raises on schema mismatch.
3. **RiskSupervisor fix** (`models/risk_supervisor.py`): new API `check_trade(signal, requested_size, state, market_data)` where size is a real position fraction computed by the sizer (NOT `abs(action)`); persist daily_pnl/peak_equity/consecutive_losses/halt state to SQLite or JSON (`state/risk_state.json`) and reload on init; configurable correlation guard (from config, not hardcoded); event-window handling; scheduled daily reset; keep circuit breakers (daily loss, drawdown, vol filter, spread filter, max trades, cooldown, consecutive losses).
4. **PositionSizing fix** (`models/position_sizing.py`): risk-based sizing — `size = (equity * risk_pct) / (stop_distance_in_price)` capped by max_position; fractional-Kelly variant; fix the broken `rssm.observe()` call in `dynamic_sizing()` (or remove/replace that method with a correct implementation).
5. **Tests** (pytest): (a) feature causality — assert feature at bar t uses only data ≤ t (construct a spike at t+n and assert features at t are unchanged); (b) feature contract — mismatched schema raises; (c) risk supervisor — normal trade approved with real size, oversized rejected, daily-loss breaker persists across instance restarts, halt cleared after 24h; (d) position sizing — 2% risk sizing math. Run all green.

### Phase 2 — NEW real backtesting engine (kernc/backtesting.py)
6. Install `backtesting.py` (pinned version) + deps into project venv; introspect API (`help(Backtest)`, `Strategy`, broker kwargs) and record the exact version + capabilities in `backtest/README.md`.
7. **New `backtest/` package** (keep old file untouched or archived as `backtest_engine_legacy.py`):
   - `strategies.py`: rule-based strategies (SMA/EMA crossover, RSI mean-reversion, ATR breakout) as `backtesting.Strategy` subclasses; `MLSignalStrategy` that loads a trained model + feature contract, computes causal features bar-by-bar, and trades on model signal (long/short/flat).
   - `costs.py`: realistic cost model — spread applied at fill (adjust prices or custom broker fill), commission, slippage; expose as `Backtest(..., commission=...)` + documented fill-price handling.
   - `engine.py`: `run_backtest(df, strategy, config) -> Stats`; `walk_forward(df, strategy_factory, n_train, n_test, purge, embargo) -> list[Stats] + aggregated OOS metrics`; `crisis_windows(df) -> dict[name, df]` (2020 COVID, 2022 rate-hike, 2024-25 events present in data); `baselines.py`: buy&hold, random (seeded), MA-crossover.
   - `report.py`: compute + persist metrics (total/ann return, Sharpe, Sortino, Calmar, max DD, win rate, profit factor, avg trade duration, #trades, total costs) to JSON + Markdown report; log every trade to CSV.
8. **Verification run (REQUIRED, real data)**: run rule-based + ML-signal backtests on `data/xauusd_h1.csv` (real data file — check its schema first; if `tick_volume` column exists rename to `volume` per backtesting.py expectation). Assert: (a) deterministic — same seed → identical results; (b) non-trivial trade count (> 20 trades over the window); (c) costs reduce net return vs zero-cost run; (d) no NaN/infinite equity; (e) walk-forward produces OOS results with purge/embargo; (f) baseline comparisons reported. Write results to `research/backtest_results/`.

### Phase 3 — Production live trading system (MT5)
9. **Broker abstraction** (`live/broker.py`): `BaseBroker` interface (initialize, shutdown, account_info, symbol_info, tick, get_positions, get_deals, send_order, close_position, modify_sl_tp, reconcile); `Mt5Broker` implementing it with full retcode handling (map retcode → action: retry on REQUOTE, raise on REJECT/ERROR, accept DONE/DONE_PARTIAL/PLACED), `order_check()` before send, SL/TP attached to every order (ATR-based from config), idempotency guard (magic + comment + position reconciliation before open), partial-fill awareness; `MockBroker` simulating fills/retcodes for tests.
10. **Live loop** (`live/live_trade_mt5.py` rewritten): candle-close-aligned scheduling (compute seconds to next closed candle for TIMEFRAME, sleep until then), completed-candle gate (only act on closed bars), startup reconciliation (broker positions vs local persisted state → mismatch = close & alert & halt), kill-switch file check each cycle, structured logging to `logs/`, heartbeat log, TRADING_MODE gate (live mode requires feature contract present, model present, risk state loadable, reconciliation pass; else refuse to start with clear error).
11. **Wiring**: loop order = fetch closed candles → compute causal features via shared pipeline → model predict (or rule signal) → PositionSizer → RiskSupervisor.check_trade → broker.execute; support long AND short (ALLOW_SHORT config); update state (pnl, equity, wins/losses) after each closed trade; persist risk state after each update.
12. **Ops artifacts**: pinned `requirements.txt` (add backtesting.py, python-dotenv, metaapi_cloud_sdk if metaapi adapter kept, pytest), updated `.env.example` documenting every variable, `README.md` status section ("research → demo-ready; NOT production/live until gates pass"), `state/` + `logs/` gitignored, smoke test with MockBroker (full loop 3 candles, assert orders placed/closed correctly, risk rejections logged).

## Deliverables
| Path | Description |
|------|-------------|
| `core/config.py`, `core/feature_pipeline.py`, `core/risk_gate.py` | Config, leak-free features + contract, trading-mode gate |
| `features/*.py` (rewritten causal) | Leak-free feature computation, contract save/load |
| `models/risk_supervisor.py`, `models/position_sizing.py` (fixed) | Correct safety layer + sizing with persistence |
| `backtest/engine.py`, `backtest/strategies.py`, `backtest/costs.py`, `backtest/baselines.py`, `backtest/report.py`, `backtest/README.md` | NEW backtesting.py-based engine + walk-forward + crisis + baselines |
| `research/backtest_results/*.json|.md` | Real backtest outputs on xauusd_h1.csv (deterministic, costs modeled) |
| `live/broker.py`, `live/live_trade_mt5.py` (rewritten), `live/mock_broker.py` | Production live system + testable broker |
| `tests/` | Feature causality, contract, risk, sizing, broker, loop smoke tests (pytest, all green) |
| `requirements.txt`, `.env.example`, `README.md` | Pinned deps, full config docs, honest status |

## Evaluation Criteria
- Every P0 item from the audit has a code fix in place; pytest suite green (feature causality, risk persistence, sizing math, mock broker loop).
- Backtest engine: deterministic, >20 trades, costs reduce returns, walk-forward OOS with purge/embargo, baselines compared — all on REAL xauusd_h1.csv; no random observation anywhere in the engine.
- Live module: `TRADING_MODE=demo` default; live mode refuses to start without gates; MockBroker smoke test passes; retcode handling maps known codes; SL/TP on every order; reconciliation logic tested.
- Feature contract: schema mismatch raises at load; live and backtest use identical feature code path.

## Notes
- Sandbox: Windows, 12 cores, ~15GB RAM (only ~1.7GB free), NO GPU. Do NOT attempt heavy RL/Dreamer training here — memory will not fit. Backtests on H1/D1 data only; if M5 files are huge, chunk-read or skip (H1 is primary).
- `live/live_trade_mt5.py` references `train/ppo_xauusd_latest.zip` — check if the checkpoint exists; if not, ML-signal path must raise a clear "model missing" error and rule-based strategies must work standalone.
- `metaapi_cloud_sdk` live file exists but MT5 is the primary adapter; keep metaapi adapter compiling or clearly stub it.
- Do not break the existing `env/` (RealisticTradingEnv + 13 passing tests) — it is the repo's best component; the new backtest engine should be compatible with strategies trained on it (action semantics: 0=flat, 1=long, 2=short).
- Old `backtest/backtest_engine.py` may be archived (rename) but the new engine must be the one used by eval scripts.


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
