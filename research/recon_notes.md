# Recon Notes — Trading Bot Full Engineering

Date: session start. Purpose: record exact feature/mode/action semantics, normalization
fit windows, allow_short handling, and the exact source location of every audit issue so
the implementation phases honor them. All paths relative to repo root.

## Data schema verdicts (verified by reading the first rows + row counts)

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| data/xauusd_h1.csv | 59,865 | time,open,high,low,close,volume | HAS `volume` column. NO `tick_volume`. No rename needed for backtesting.py. Range starts 2015-11-17 00:00 |
| data/xauusd_d1.csv | 6,787 | time,open,high,low,close,volume | 2005-01-02 onwards |
| data/macro_daily.csv | 3,699 | time,dxy_close,spx_close,vix_close,oil_close,btc_close,silver_close,gld_close,gold_futures_close | Daily closes, time is a DATE (00:00 stamped) |
| data/xauusd_h1_macro.csv | exists | merged hourly + dxy/spx/us10y closes | produced by data/merge_macro.py |

## Action-space / environment semantics (env/dreamer_trading_env.py — DO NOT BREAK)

- RealisticTradingEnv: `_decode_action(action_onehot)` → `{0:0, 1:1, 2:-1}` when
  `allow_short=True` (action_space=3), else `{0:0, 1:1}` (action_space=2).
- Causality: at step t the agent sees features up to bar t-1; position chosen at t is
  filled at open of bar t and earns `returns[t]`.
- Costs: spread (fraction of notional, half paid per side), commission per side,
  slippage (mean |slippage| per fill, vol-scaled, mostly adverse), swap (daily rollover,
  triple on Wednesday), stop_loss / take_profit as fraction of trade, max_drawdown ends
  episode. All defaults live in DEFAULT_ENV_KWARGS.
- 13 passing tests in env/test_dreamer_trading_env.py (causality, cost accounting, SL,
  swap, drawdown, random start, episode stats). MUST stay green.

## Normalization semantics (BEFORE fix — the bug)

- features/make_features.py `compute_features()`: computes ~11 features, drops first 120
  bars, then `mu = feats.mean(axis=0)`, `sig = feats.std(axis=0) + 1e-8`,
  `feats = (feats-mu)/sig` on the WHOLE dataset passed in → test stats leak into train
  (P0-4 leak 3).
- Live (live/live_trade_mt5.py) uses only features.make_features.compute_features (~20
  cols) while training uses the 150+ feature pipeline → P0-3 mismatch. Live normalization
  on last-200-bars (implicit; actually none beyond make_features' own whole-frame norm).
- FIX: fit scaler on TRAIN window only; save feature_contract.json (feature_names +
  scaler mean/scale + params hash); enforce schema at load.

## Look-ahead leaks (P0-4)

1. features/multi_timeframe.py `create_multi_timeframe_data()` uses default
   `df_base.resample(rule)` which labels the RESAMPLE BAR AT ITS START TIME, then
   `MultiTimeframeFeatures.create_features` computes H1/H4/D1 features and
   ultimate_150_features / timeframe_features reindex+ffill onto M5 → an M5 bar at 10:00
   sees an H1 close that closes at 11:00. FIX: resample with `label='right',
   closed='right'` and `shift(1)` before merging so a bar only sees CLOSED higher-TF
   candles.
2. features/macro_features.py / data/merge_macro.py: daily macro close stamped at 00:00
   then `reindex(..., method='ffill')` through the day → today's close known at day
   start. FIX: shift(1) the daily macro series before ffill.
3. Normalization on full dataset (see above).

## Audit issue → exact source location map

| Issue | Where | What's wrong |
|-------|-------|--------------|
| P0-1 no risk control live | live/live_trade_mt5.py | No SL/TP, no daily loss, no drawdown, no consecutive-loss breaker, RiskSupervisor never imported, VOLUME=0.01 hardcoded, DEVIATION=20 |
| P0-2 RiskSupervisor buggy | models/risk_supervisor.py | `position_size = abs(action)` treats action index as fraction → every trade rejected ('POSITION_TOO_LARGE'); state memory-only; DXY correlation hardcoded (`if action == 1: if dxy_momentum > 0.01: reject`) |
| P0-3 train/live feature mismatch | live/live_trade_mt5.py vs features/ultimate_150_features.py + features/make_features.py | 150+ vs ~20 features; whole-dataset vs last-200 normalization; no saved feature contract/scaler |
| P0-4 look-ahead | features/multi_timeframe.py resample; features/macro_features.py + data/merge_macro.py ffill; features/make_features.py normalization | 3 leaks (see above) |
| P0-5 fake backtester | backtest/backtest_engine.py | `_get_observation()` returns `np.random.randn(100)`; `walk_forward_validation()` has `# self.agent.train(train_data)` commented out; MockAgent random actions; eval/crisis_validation.py placeholder P&L |
| P0-6 execution loop wrong | live/live_trade_mt5.py | 10s loop on H1 strategy; no candle-closed gate; long/flat only; no MT5 retcode checking/retry/requote; no idempotency; no reconciliation on restart |
| P0-7 config/deps broken | .env unused; requirements.txt unpinned (all >=); README references missing files (train/smoke_env.py, deploy_setup.sh, trading-bot.service); metaapi_cloud_sdk missing | — |
| P1 dreamer save() | models/dreamer_agent.py `save()` | Only network weights saved; no optimizer state, RNG, return-normalizer, feature schema |
| P1 ReplayBuffer.sample() | models/dreamer_agent.py | contiguous sequence sampling can cross episode boundaries |
| P1 dynamic_sizing rssm.observe args | models/position_sizing.py `dynamic_sizing()` | calls `agent.rssm.observe(embed, None, h, z)` with wrong args/returns (observe returns 4 values, needs action tensor); advantage always 0 → win_prob always 0.5 |
| P1 calendar O(N×E) | features/calendar_features.py | per-timestamp Python loop over whole event list; use searchsorted/vectorized |
| P1 sentiment always 0.0 | data/sentiment_analysis.py | placeholder social sentiment returns 0.0; aggregate defaults to 0.0 |
| P1 evaluate_model annualization | eval/ scripts (annualization 252*24*12, ylim hides shorts) | overstates metrics, hides shorts |
| P1 timeframe_features requires 'volume' | features/timeframe_features.py | `df['volume'].rolling(20)` KeyError if only 'tick_volume' (MT5 copy_rates) present |
| P1 no CI/Docker/lint/logging/alerting | repo | single test file, no CI, no Dockerfile, no lint config, no structured logging, no alerting |

## Good components to NOT break (verified)

- env/dreamer_trading_env.py + env/test_dreamer_trading_env.py (13 passing tests)
- models/dreamer_components.py + models/dreamer_agent.py (RSSM, symlog, two-hot critic)
- SECURITY.md, .gitignore

## Environment facts

- venv at .venv (python 3.11.9). Key installed: pandas 3.0.5, numpy 2.5.3, torch 2.14.0,
  stable_baselines3 2.9.0, pytest 9.1.1, python-dotenv 1.2.3, matplotlib 3.11.2,
  scikit-learn 1.9.1, backtesting 0.6.2 (just installed).
- ~1.7 GB RAM free, NO GPU → NO Dreamer/RL training in sandbox; backtests on H1/D1 only.
- MT5 terminal NOT available → Mt5Broker built against real API surface, tested via
  MockBroker + retcode table unit tests; document live verification on user's MT5.
- Windows cmd.exe: inline multiline python -c fails → write .py helper scripts.
