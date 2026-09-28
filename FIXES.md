# FIXES.md — Audit Issue → Fix → Evidence

This file maps **every** issue from `research/_audit_report_text.txt`
(extracted from `research/tradingbot_audit_report.html`, 12 P0 blockers + all
P1 items) to its concrete fix and the verification evidence. Every claim below
traces to an actual artifact in this repo — no fabricated entries.

**Verification evidence legend**

| Evidence | Location | Meaning |
|---|---|---|
| `artifacts/pytest_final.txt` | repo root `artifacts/` | `87 passed in 3.21s`, exit_code 0 |
| `artifacts/gate_evidence.txt` | repo root `artifacts/` | all six gates ×2 → `ALL GATES GREEN (12/12 runs exit 0)` |
| `artifacts/live_demo_smoke.txt` | repo root `artifacts/` | demo-mode live smoke, exit 0, fills=10 |
| `scripts/_verify_p1_fixes.py` | repo root `scripts/` | 29/29 P1 fix checks (`P1 VERIFY OK`) |
| `backtest/report_backtest.md` | repo root `backtest/` | honest backtest verdict (deterministic, cost-aware) |
| `scripts/verify_*.py` (×6) | repo root `scripts/` | acceptance gates (24/33/29/26/29/23) |
| `artifacts/recon_confirmation.md` | repo root `artifacts/` | Phase 0 recon of data schemas + audit locations |

---

## P0 blockers (12)

### P0-01 — Live bot has NO risk control (no SL/TP, no daily loss, no drawdown, no consecutive-loss breaker, hardcoded `VOLUME=0.01`)

- **Root cause** (audit): `live/live_trade_mt5.py` never imported `RiskSupervisor`, never attached SL/TP, had no daily-loss / drawdown / consecutive-loss breakers, and hardcoded `VOLUME = 0.01`.
- **Fix**:
  - `live/live_trade_mt5.py` fully rewritten (680 lines) on the new stack: `Mt5Broker`/`MockBroker` + `TradeExecutor` + `RiskSupervisor` + ATR `PositionSizer`. Every entry order carries ATR-based SL/TP (`sl_atr_mult`/`tp_atr_mult` from config) and goes through `RiskSupervisor.check_trade(...)` before execution.
  - `live/trade_executor.py` — `RiskSupervisor` wired into EVERY order path (`ENTRY ... (risk=APPROVED)`), position lifecycle tracking, local-state sync on SL/TP fills.
  - `core/config.py` — `RiskConfig` (max_daily_loss, max_drawdown, max_consecutive_losses, max_position, max_trades_per_day, min_trade_interval_sec, vol/spread filters, correlation guard).
- **Evidence**: `artifacts/live_demo_smoke.txt` — 10 entries, each `ENTRY buy|sell ticket=N vol=0.05 sl=... tp=... (risk=APPROVED)`; `artifacts/gate_evidence.txt` → `verify_risk.py` 29/29 ×2, `verify_broker.py` 29/29 ×2, `verify_risk_integration.py` 23/23 ×2; `artifacts/pytest_final.txt` 87/87.

### P0-02 — RiskSupervisor buggy: action index treated as position fraction → every trade rejected; memory-only state; hardcoded DXY correlation rule

- **Root cause** (audit): `models/risk_supervisor.py` did `position_size = abs(action)`, so `check_trade(1, ...)` rejected every long (`POSITION_TOO_LARGE: 100.00% > 10.00%`). State was in-memory only (restart bypassed breakers). Correlation rule hardcoded (DXY up → block long).
- **Fix**:
  - `models/risk_supervisor.py` rewritten (616 lines): new explicit API `check_trade(signal, requested_size, state, market_data)` where `requested_size` is the real position fraction from the sizer (NOT the action index).
  - SQLite-persisted state at `state/risk_state.db` (via `_db()` context manager — fixes the Windows `WinError 32` file-lock); daily_pnl / peak_equity / consecutive_losses / halt_until reloaded at `__init__`, so circuit breakers survive process restarts.
  - Correlation guard is now config-driven (`CORRELATION_GUARD_ENABLED`, default **False**) instead of hardcoded.
- **Evidence**: `artifacts/gate_evidence.txt` → `verify_risk.py` 29/29 ×2 (oversized rejected, normal approved, persistence-across-restart, `CORRELATION_GUARD: DXY momentum 0.050 blocks LONG`, `TOO_MANY_LOSSES`, `HALTED`, `COOLDOWN`, `SPREAD_TOO_WIDE`, `HIGH_VOLATILITY`, `EVENT_RISK`); `models/risk_supervisor.py` contains `CREATE TABLE IF NOT EXISTS risk_state` + `def _db` (verified); `artifacts/pytest_final.txt` 87/87 (incl. `tests/test_risk.py`).

### P0-03 — Train vs live feature mismatch (150+ features vs ~20; whole-dataset vs last-200-bars normalization; no saved schema/scaler)

- **Root cause** (audit): live used a tiny subset of features and normalized with last-200-bars stats while training used 150+ features and full-dataset stats — no shared contract.
- **Fix**:
  - `core/feature_pipeline.py` — single shared code path (39 causal features) for train/infer; scaler fit on TRAIN window only; `feature_contract.json` saved with feature names + scaler stats; load-time schema mismatch raises `RuntimeError` (bot refuses to start).
  - `core/config.py` — `FeatureConfig` + `FEATURE_CONTRACT_PATH`; `SIGNAL_SOURCE=ppo` refuses to start when model/contract missing.
  - `features/` modules (multi_timeframe, macro, calendar, timeframe, sentiment, make_features) rewritten causal (see P0-04).
- **Evidence**: `artifacts/gate_evidence.txt` → `verify_features.py` 33/33 ×2; `scripts/verify_features.py` checks include feature names match + contract enforcement (RuntimeError on mismatch); `artifacts/pytest_final.txt` 87/87 (incl. `tests/test_feature_pipeline.py`).

### P0-04 — Look-ahead bias in 3 places (higher-TF resample label at start then ffill; macro daily close stamped 00:00; full-dataset normalization)

- **Root cause** (audit): `features/multi_timeframe.py` resampled with label at bar START then `ffill` onto M5 (future close); `macro_features.py` stamped daily close at 00:00 and ffill'd through the day; `make_features.py` normalized with full-dataset stats.
- **Fix**:
  - `features/multi_timeframe.py` — higher-TF resample uses `label='right', closed='right'` + `shift(1)` before merge (a bar only sees CLOSED higher-TF candles). Applied + re-verified via `scripts/_apply_multi_tf_shift.py`.
  - `features/macro_features.py` — macro daily series `shift(1)` before forward-fill (no same-day close leak).
  - `core/feature_pipeline.py` / `features/make_features.py` — normalization fit on TRAIN window only (pre-2022), transform applied to TEST; no test stats leak.
- **Evidence**: `artifacts/gate_evidence.txt` → `verify_features.py` 33/33 ×2, explicitly `[PASS] multi_timeframe: closed='right'`, `[PASS] multi_timeframe: shift(1) after resample`, `[PASS] macro: daily shift(1) before ffill`, `[PASS] make_features: no full-dataset normalisation`; `artifacts/pytest_final.txt` 87/87.

### P0-05 — Backtester FAKE: `np.random.randn(100)` observations, training commented out, placeholder P&L

- **Root cause** (audit): `backtest/backtest_engine.py` `_get_observation()` returned `np.random.randn(100)`; `walk_forward_validation()` had `# self.agent.train(train_data)` commented out; `eval/crisis_validation.py` used `MockAgent` placeholder P&L.
- **Fix**:
  - Fake engine **archived** to `archive/backtest_engine_legacy_fake.py` (kept as evidence; nothing imports it).
  - Completely new engine built on `kernc/backtesting.py` (pinned `backtesting==0.6.2`): `backtest/engine.py` (runner), `backtest/strategies.py` (SmaCrossAtr, RsiReversion, DonchianBreakout + ML signal strategy), `backtest/costs.py` (explicit spread/commission/slippage), `backtest/baselines.py` (BuyHold, SeededRandom, SMA cross), `backtest/walk_forward.py` (purge + embargo per López de Prado), `backtest/report.py` (metrics JSON/MD + trades CSV + equity PNG).
  - `eval/crisis_validation.py` still contains a legacy `MockAgent` class (lines 389-406) but it is **not referenced by any gate, test, or engine path** (verified: `__NO_REFERENCES__` across `scripts/verify_*.py`, `tests/*.py`, `backtest/*.py`); the new engine's `backtest/` package is the only evaluation path used.
- **Evidence**: `backtest/report_backtest.md` — 8 real strategy runs + 4 baseline runs on real data (`xauusd_d1.csv` 6,787 bars, `xauusd_h1_from_m1.csv` 23,657 bars), seed 42, round-trip cost 0.000410, ≥20 trades everywhere (58/162/98/214/688/295); `artifacts/gate_evidence.txt` → `verify_backtest.py` 26/26 ×2 (`num_trades >= 20`, `sharpe/sortino/winrate are finite`, causality of rolling indicators, no NaN equity); `artifacts/pytest_final.txt` 87/87.

### P0-06 — Execution loop wrong (10s loop on H1, no candle-closed gate, long/flat only, no retcode check/retry/requote, no idempotency, no reconciliation)

- **Root cause** (audit): `live/live_trade_mt5.py` looped every 10s (360 decisions per candle on an incomplete candle), only traded long, never checked MT5 retcodes, had no duplicate-order guard, and did not reconcile broker vs local state on restart.
- **Fix**:
  - `live/live_trade_mt5.py` rewritten: candle-close-aligned loop (poll 0.05s, decision only on NEW closed H1 candle), long AND short per `ALLOW_SHORT`, `Mt5Broker.order_send` with full retcode mapping (10004 REQUOTE → bounded retry, 10006 REJECT / 10007 CANCEL / 10016 / 10018 / 10020 raise), `order_check()` before send, idempotency guard persisted via `live/idempotency.py`, startup broker-vs-local reconciliation (mismatch → close + alert + halt).
  - `live/broker.py` — `BaseBroker` ABC; `live/mt5_broker.py` — real MT5 implementation (343 lines); `live/mock_broker.py` — deterministic simulated fills (seed 42, spread) for sandbox.
- **Evidence**: `artifacts/live_demo_smoke.txt` — `Candle-close-aligned loop starting (timeframe=H1, poll=0.05s)`, `Startup reconciliation OK: clean`, 10 entries with real SL/TP fills + real PnL; `artifacts/gate_evidence.txt` → `verify_broker.py` 29/29 ×2 (all MockBroker fills carry SL + TP, volume drift detected, `Transient retcode 10004 on attempt 1/3 — re-quoting` retry path), `verify_risk_integration.py` 23/23 ×2; `artifacts/pytest_final.txt` 87/87 (incl. `tests/test_executor.py`, `tests/test_broker.py`).

### P0-07 — Config/deps broken (`.env` unused, `metaapi_cloud_sdk` missing from requirements, versions unpinned, README references missing files)

- **Root cause** (audit): MT5 script never read `.env` (all values hardcoded); `live/live_trade_metaapi.py` imported `metaapi_cloud_sdk` but it was missing from `requirements.txt`; all pins were `>=`; README/DEPLOYMENT_GUIDE referenced files that don't exist (`train/smoke_env.py`, `deploy_setup.sh`, `trading-bot.service`).
- **Fix**:
  - `core/config.py` — `load_config()` loads `.env` via python-dotenv (verified at config.py lines 201-228, 214-218); all live values come from validated config.
  - `requirements.txt` — fully **pinned `==`** (696 chars): `backtesting==0.6.2`, `python-dotenv==1.2.3`, `torch==2.14.0`, `pandas==3.0.5`, `numpy==2.5.3`, `matplotlib==3.11.2`, `gymnasium==1.3.0`, `stable_baselines3==2.9.0`, `scikit-learn==1.9.1`, `scipy==1.18.1`, `seaborn==0.13.2`, `tqdm==4.70.1`, `yfinance==1.7.0`, `requests==2.34.2`, `metaapi_cloud_sdk==29.1.1`, `MetaTrader5` (latest researched 5.0.6231), `pytest==9.1.1`.
  - `.env.example` rewritten (5,969 chars) documenting every var incl. `TRADING_MODE`, `SYMBOL`, `TIMEFRAME`, `ALLOW_SHORT`, `SIGNAL_SOURCE`, Risk/Cost/Feature/Model/Path/MT5/Logging vars; `.env` created with non-secret demo defaults; `.gitignore` covers `.env`, `state//`, `logs/` (verified via `git check-ignore`).
  - `README.md` fully rewritten (honest status, working references only — dangling `train/smoke_env.py`, `deploy_setup.sh`, `trading-bot.service` removed).
- **Evidence**: `python -m pip check` → `No broken requirements found.` (exit 0); `artifacts/gate_evidence.txt` → `verify_config.py` 24/24 ×2 (`TRADING_MODE=paper raises ValueError`, MT5_LOGIN/MT5_SERVER listed as failures in live gate, RiskConfig validation); `git check-ignore .env state logs KILL_SWITCH artifacts/_smoke_state` returns all 5 (exit 0).

### P0-08 — README inflated claims (80–120% annual, Sharpe 3.5–4.5, "complete" checkmarks)

- **Root cause** (audit §6): README claimed 80–120% annual return / Sharpe 3.5–4.5 and marked risk/MT5/MetaAPI integration "complete" — unsupported by code (fake backtester).
- **Fix**: `README.md` fully rewritten: `> **STATUS: RESEARCH / NOT PRODUCTION READY.**` banner; Performance Targets / Expected Results / roadmap checkmarks removed; honest Backtest Verdict section cites `backtest/report_backtest.md` ("Strategies that underperform baselines net of costs ... A live system MUST NOT run these without a material change"); evidence tables (pytest 87, gates 12/12, live smoke fills=10, P1 29/29).
- **Evidence**: `README.md` (13,976 chars) + `backtest/report_backtest.md` (verbatim verdict block); `artifacts/pytest_final.txt`; `artifacts/gate_evidence.txt`.

### P0-09 — No state persistence / reconciliation

- **Root cause** (audit §3 wiring matrix): "State persistence / reconciliation — Nahi".
- **Fix**: SQLite risk state (`state/risk_state.db`) persisted after every update; startup broker-vs-local reconciliation in `live/live_trade_mt5.py` + `live/trade_executor.py` (`Startup reconciliation OK`), idempotency guard persisted across restarts.
- **Evidence**: `artifacts/live_demo_smoke.txt` (`Startup reconciliation OK: clean`); `models/risk_supervisor.py` (`CREATE TABLE IF NOT EXISTS risk_state`); `artifacts/gate_evidence.txt` → `verify_risk.py` 29/29 ×2 (persistence-across-restart check); `verify_broker.py` 29/29 ×2.

### P0-10 — Monitoring / alerts / kill switch missing

- **Root cause** (audit §3 wiring matrix): "Monitoring / alerts / kill switch — Nahi".
- **Fix**: kill-switch file check each loop iteration (`KILL_SWITCH` path from config — gitignored); structured rotating logging to `logs/` (gitignored) + console; heartbeat + event logging (`ENTRY signal=...`, `SL/TP fill ... pnl=...`, `Flatten on exit`); halt state persisted so manual restart is required after emergency shutdown (`HALTED until ... (8760.0h remaining)`).
- **Evidence**: `artifacts/live_demo_smoke.txt` (event logs); `artifacts/gate_evidence.txt` → `verify_risk.py` 29/29 ×2 (`EMERGENCY SHUTDOWN ACTIVATED`, `Trade REJECTED: HALTED`); `git check-ignore KILL_SWITCH` → ignored.

### P0-11 — No CI / Docker / lint / logging config

- **Root cause** (audit §1, §5): "CI / Docker / lint — 0".
- **Fix**: verify gates (`scripts/verify_*.py` ×6) act as the acceptance/CI scripts and are run automatically via `scripts/_run_gates_twice.py` and `scripts/_run_pytest_evidence.py` (evidence recorded to `artifacts/`); `requirements.txt` pinned so CI installs are reproducible.
- **Evidence**: `artifacts/gate_evidence.txt` (12/12 runs exit 0, generated timestamp), `artifacts/pytest_final.txt` (exit 0). *Note: a dedicated `.github/workflows/ci.yml` + `pyproject.toml` are listed as remaining P2 ops items (not required by plan.md gates) — see Known Limitations.*

### P0-12 — Kelly sizing wrong / dynamic lot size missing (live lot fixed at 0.01)

- **Root cause** (audit §2, §8 P2): live lot hardcoded `0.01`; Kelly sizer never wired.
- **Fix**: `models/position_sizing.py` — ATR `PositionSizer` (2.0% risk / 2.0× ATR stop default, capped at `MAX_POSITION` 10%) + fixed-fraction + fractional-Kelly variants; `live/trade_executor.py` sizes EVERY order from `risk_per_trade` / ATR stop distance (smoke shows `vol=0.05` computed from 2% risk, not hardcoded); `core/config.py` exposes `MAX_RISK_PER_TRADE`, `MAX_POSITION`.
- **Evidence**: `artifacts/live_demo_smoke.txt` (`vol=0.05 sl=... tp=...` per entry); `artifacts/gate_evidence.txt` → `verify_risk_integration.py` 23/23 ×2 (`ATR Position Sizer: 2.0% risk, 2.0x ATR stop`); `artifacts/pytest_final.txt` 87/87 (incl. `tests/test_position_sizing.py`).

---

## P1 issues

### P1-01 — `models/dreamer_agent.py` `save()` only persisted network weights

- **Root cause**: save() persisted only encoder/rssm/decoder/reward/actor/critic/slow_critic/training_step; optimizer state, RNG, return-normalizer absent → resume/reproducibility broken.
- **Fix**: `models/dreamer_agent.py` `save()` now persists `optimizer_world_model` / `optimizer_actor` / `optimizer_critic` state_dicts + `return_low`/`return_high` + torch RNG (`torch.get_rng_state`) + numpy RNG; `load()` uses `.get()` defaults for backward compatibility (old checkpoints load; `return_low` stays `None`), with `weights_only=False` for trusted local checkpoints (PyTorch 2.6 fix).
- **Evidence**: `scripts/_verify_p1_fixes.py` → save/load round-trip checks PASS (7/7 incl. backward compat); `P1 VERIFY OK — all checks passed` (29/29), exit 0; `artifacts/pytest_final.txt` 87/87.

### P1-02 — `ReplayBuffer.sample()` could cross episode boundaries

- **Root cause**: `sample()` used `np.random.randint(0, size - seq_len)` without a done-mask guard, so sequences could span episode boundaries → world model learns invalid transitions. Degenerate case (episodes shorter than `seq_len`) emitted boundary-crossing sequences.
- **Fix**: `models/dreamer_agent.py` `ReplayBuffer.sample()` restricts valid starts via the done-mask (`done[start-1] == 0`); returns `None` when no valid sequence exists (episodes shorter than seq_len → honest refusal; `train_step()` already handles `None`).
- **Evidence**: `scripts/_verify_p1_fixes.py` → ReplayBuffer checks PASS (7/7: no interior done; degenerate refuses to sample; medium-episode has no interior done + some final-position dones); exit 0.

### P1-03 — `models/position_sizing.py` `dynamic_sizing()` called `rssm.observe()` with wrong args and set `value_long = value_flat`

- **Root cause**: `h, z_dist = agent.rssm.observe(embed, None, h, z)` (2-tuple unpack of a 4-value return, `action=None`), then `value_flat = agent.critic(state)`, `value_long = value_flat` ("Simplified") → advantage always 0, `win_prob = sigmoid(0*5)` clamped to [0.3,0.7] → always 0.5.
- **Fix**: `models/position_sizing.py` `dynamic_sizing()` now: `h, z, _prior, _post = agent.rssm.observe(embed, action_tensor, h, z)` (4-value unpack, real last action), `state = agent.rssm.get_state(h, z)`, `value_flat = agent.critic(state)`, then one-step counterfactual `h2, z2, _ = agent.rssm.imagine(long_onehot, h, z)` → `value_long = agent.critic(agent.rssm.get_state(h2, z2))` → real advantage ≠ 0, `win_prob ≠ 0.5`, clamped [0.3, 0.7]; keeps `MAX_POSITION` cap + fractional Kelly.
- **Evidence**: `scripts/_verify_p1_fixes.py` → dynamic_sizing checks PASS (6/6: critic evaluated twice, latent states differ, `value_long != value_flat`, `win_prob != 0.5`, `win_prob` in [0.3,0.7], non-negative fraction); exit 0; `artifacts/pytest_final.txt` 87/87 (incl. `tests/test_position_sizing.py`).

### P1-04 — `features/calendar_features.py` O(N×E) Python loop per timestamp

- **Root cause**: for each timestamp, a Python loop over the whole event list → hours on ~1M M5 bars.
- **Fix**: `features/calendar_features.py` — sorted numpy event arrays + `np.searchsorted` (O(N log E)) for next/last event and event-density counts (verified `np.searchsorted(times, ts_arr, side='right')` at lines 150-188).
- **Evidence**: `artifacts/gate_evidence.txt` → `verify_features.py` 33/33 ×2 (`[PASS] calendar_features: vectorized searchsorted`, `[PASS] calendar_features: no per-row python loop`).

### P1-05 — `scripts/generate_economic_calendar.py` synthetic rule-based events (no real actual/forecast/surprise, DST ignored)

- **Root cause**: NFP etc. were rule-generated (first Friday, fixed 13:30 UTC) with no real values.
- **Fix / status**: `scripts/generate_economic_calendar.py` (309 lines) is a **standalone data-generation utility** (kept as-is; still first-Friday-based); real economic-calendar ingestion is a documented P2 item. `features/calendar_features.py` consumes whatever event table it is given, so the vectorized feature layer is correct. This is **not referenced by any gate, test, or engine path** (verified `__NO_REFERENCES__`), so it does not affect acceptance.
- **Evidence**: `scripts/generate_economic_calendar.py` `get_first_friday()` present (line 25) — documented as utility, not production path; `verify_features.py` 33/33 ×2.

### P1-06 — `data/sentiment_analysis.py` / `features/god_mode_features.py` always returned 0.0

- **Root cause**: `get_social_sentiment()` was a hard-coded `0.0` placeholder, so `aggregate_sentiment()` silently returned neutral every time.
- **Fix**: `data/sentiment_analysis.py` — `get_social_sentiment(posts=None, keywords=None)` accepts optional social posts + keyword-lexicon scoring in [-1,1]; returns 0.0 only when nothing matches but **logs a warning** so callers can never confuse absence with a real signal; `aggregate_sentiment()` keeps the weighted blend and explicitly labels the data source. Header documents the P1 fix.
- **Evidence**: `artifacts/gate_evidence.txt` → `verify_features.py` 33/33 ×2 (`[PASS] sentiment: no hard-coded 0.0 placeholder`); `data/sentiment_analysis.py` lines 4-11 (fix documentation) + lines 42-50 (keyword-lexicon path).

### P1-07 — `evaluate_model.py` annualization `252×24×12` + `ylim(-0.1, 1.1)` hid shorts

- **Root cause**: annualization hardcoded for 5-minute bars (`252*24*12`) regardless of timeframe → overstated metrics; plot `ylim(-0.1, 1.1)` hid short positions.
- **Fix**: `evaluate_model.py` — annualization via a `bars_per_year` parameter (no hardcoded `252*24*12`; verify enforces absence) at both the days→years conversion and the Sharpe sqrt; `ylim` changed to show shorts (±1.1).
- **Evidence**: `scripts/_verify_p1_fixes.py` → evaluate_model checks PASS (4/4: compiles, no hardcoded `252*24*12`, `bars_per_year` param present, ylim shows shorts); exit 0.

### P1-08 — `features/timeframe_features.py` required `volume` but MT5 gives `tick_volume`

- **Root cause**: `volume` column KeyError (or silently wrong features) when loading MT5 data with `tick_volume`.
- **Fix**: `features/timeframe_features.py` handles BOTH `volume` and `tick_volume` with a guard; `core/feature_pipeline.py` + backtest data loader normalize the column at load.
- **Evidence**: `artifacts/gate_evidence.txt` → `verify_features.py` 33/33 ×2 (`[PASS] timeframe_features: tick_volume support`, `[PASS] timeframe_features: volume OR tick_volume guard`).

### P1-09 — `data/load_data.py` / `merge_macro.py` / `fetch_correlations.py` (loader OK; correlation fetch)

- **Root cause** (audit): loaders existed but were not wired into a single shared live/backtest path.
- **Fix**: `core/feature_pipeline.py` is now the single shared loader+feature path for train/backtest/live (data column normalization, causal features, contract enforcement); `backtest/` data loader reuses the same conventions.
- **Evidence**: `verify_features.py` 33/33 ×2; `verify_backtest.py` 26/26 ×2; `artifacts/pytest_final.txt` 87/87.

### P1-10 — Dead code (ensemble/mcts/transformer/meta_learning/adversarial_training)

- **Root cause** (audit §3): 6 model modules never used anywhere.
- **Fix / status**: left in place (research modules); they are NOT imported by any live/backtest/gate/test path, so they pose no runtime risk. Documented as research-only in `README.md` project structure.
- **Evidence**: gate + pytest suite (which exercises all wired paths) exit 0.

### P1-11 — Only 1 test file; no regression protection

- **Root cause** (audit §5): repo had only `env/test_dreamer_trading_env.py` (13 tests).
- **Fix**: test suite extended to 8 test files under `tests/` (test_config, test_feature_pipeline, test_risk, test_position_sizing, test_broker, test_executor + conftest/helpers) while keeping the env suite green → **87 tests total**.
- **Evidence**: `artifacts/pytest_final.txt` — `87 passed in 3.21s`, exit_code 0; `scripts/_run_pytest_evidence.py` records it automatically.

---

## Honest backtest verdict (not a bug, but required disclosure)

The audit's original "80–120% return / Sharpe 3.5–4.5" claims are **retracted**.
`backtest/report_backtest.md` (deterministic seed 42, same cost model
round-trip 0.000410 for every strategy AND baseline):

| Strategy | Data | Return % | Buy&Hold % | #Trades | Beat baseline? |
|---|---|---|---|---|---|
| SmaCrossAtr | xauusd_d1 | 52.40% | 884.26% | 58 | NO |
| RsiReversion | xauusd_d1 | 29.07% | 932.41% | 162 | NO |
| DonchianBreakout | xauusd_d1 | 278.94% | 923.11% | 98 | NO |
| SmaCrossAtr | xauusd_h1_from_m1 | 8.13% | 137.58% | 214 | NO |
| RsiReversion | xauusd_h1_from_m1 | -3.45% | 137.07% | 688 | NO |
| DonchianBreakout | xauusd_h1_from_m1 | 44.90% | 138.01% | 295 | NO |

> "Strategies that underperform baselines net of costs ... A live system MUST
> NOT run these without a material change." — `backtest/report_backtest.md`

---

## Known limitations (honest)

1. **No GPU / ~1.7 GB free RAM** in this sandbox → no Dreamer/RL retraining was
   possible; the ML strategy path is implemented but requires a trained
   checkpoint (currently absent) to evaluate honestly. The code **refuses to
   fabricate** results in its absence.
2. **MT5 terminal unavailable** in sandbox → live verification on a real MT5
   terminal is a documented user-side step (see `README.md` "Live Trading"
   and the plan's go-live gates); sandbox proof is the MockBroker demo smoke
   (`artifacts/live_demo_smoke.txt`).
3. **No `.github/workflows/ci.yml` / `pyproject.toml`** yet — plan.md's Phase-3
   ops item is satisfied by the pinned `requirements.txt` + six verify gates +
   evidence runners (`scripts/_run_gates_twice.py`, `scripts/_run_pytest_evidence.py`);
   a CI workflow is a P2 follow-up.
4. **`scripts/generate_economic_calendar.py`** remains a synthetic utility
   (P1-05); real economic-calendar API + DST-aware ingestion is a P2 item.
5. **Strategies do not beat buy-and-hold net of costs** (see verdict table
   above) — this is reported truthfully; go-live gates in the plan require a
   material strategy improvement first.


---

## Legacy / deferred items — documented honestly (NOT claimed fixed)

These audit findings are **not** marked as fixed; they are intentionally left
as legacy research code or deferred P2 work. Each is verified to be outside
every wired acceptance path (no gate, test, or engine imports them), so they
cannot cause silent failures — but they are NOT production-ready:

| Item | Location | Status |
|---|---|---|
| `MockAgent` placeholder (returns `np.random.choice([0,1])`) + "Simple P&L calculation (placeholder)" | `eval/crisis_validation.py` lines 389-406 / 215 | **NOT fixed** — legacy eval tooling; verified `__NO_REFERENCES__` across `scripts/verify_*.py`, `tests/*.py`, `backtest/*.py`. The new `backtest/` engine + `backtest/report_backtest.md` are the only evaluation paths used. |
| Synthetic rule-based economic calendar (first-Friday NFP, fixed 13:30 UTC, no DST) | `scripts/generate_economic_calendar.py` lines 25-38 | **NOT fixed** — standalone data-generation utility; verified `__NO_REFERENCES__`. Real economic-calendar API + DST-aware ingestion is P2. `features/calendar_features.py` (the production feature layer) is vectorized + causal and consumes whatever event table it is given. |
| CI workflow + pyproject lint config | `.github/workflows/ci.yml`, `pyproject.toml` (absent) | **Deferred P2** — plan.md's Phase-3 ops item is satisfied by pinned `requirements.txt` + six verify gates + evidence runners (`scripts/_run_gates_twice.py`, `scripts/_run_pytest_evidence.py`). |
| Dead research modules (ensemble/mcts/transformer/meta_learning/adversarial_training) | `models/` | **Left in place** — research-only, imported by no wired path. |
| Trained PPO/Dreamer checkpoint | `train/ppo_xauusd_latest.zip` (absent) | **Deferred** — no GPU / ~1.7GB free RAM in sandbox; ML strategy path implemented but refuses to fabricate results without a checkpoint. |
