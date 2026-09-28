# 🤖 DRL Trading Bot — XAUUSD

> **STATUS: RESEARCH / NOT PRODUCTION READY.**
> This repository is an engineering reference. The legacy `backtest_engine.py`
> produced **fake** results (`np.random.randn(100)` observations, commented-out
> training, placeholder P&L). It has been **archived** to
> `archive/backtest_engine_legacy_fake.py` and replaced by a completely new
> engine built on [`kernc/backtesting.py`](https://github.com/kernc/backtesting.py).
> **Do NOT run this bot with real money** until the honest caveats below are
> resolved. Every audit finding and its fix is mapped in **[FIXES.md](FIXES.md)**.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 📋 Table of Contents

- [Honest Status](#-honest-status)
- [Model Pipeline Status](#model-pipeline-status)
- [Model Catalog](docs/MODELS.md)
- [Training and Deployment Guide](docs/TRAINING_GUIDE.md)
- [Verified Facts](#-verified-facts)
- [How It Works](#-how-it-works)
- [Installation](#%EF%B8%8F-installation)
- [Configuration](#-configuration)
- [Quick Start Guide](#-quick-start-guide)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [Backtest Verdict (Honest)](#-backtest-verdict-honest)
- [Live Trading](#-live-trading)
- [Disclaimer](#%EF%B8%8F-disclaimer)

---

## Model pipeline status

The shared observation, policy, artifact, training, backtest, and live model
interfaces are implemented and covered by tests. **No trained model artifacts
are shipped.** Users must train a model on their own data, evaluate it on a
separate period, and pass the promotion gate before selecting it for live
inference.

The code is production-structured; no model is validated for real money until
it passes the gate on your data and a demo period. See the
[model catalog](docs/MODELS.md) and [training and deployment guide](docs/TRAINING_GUIDE.md)
for supported model types, artifact contents, commands, limitations, and
operational checks.

---

## 🩺 Honest Status

This project started as an aspirational "fully autonomous AI trading bot"
README. The audit (`research/_audit_report_text.txt`, extracted from
`research/tradingbot_audit_report.html`) found **12 P0 blockers** and multiple
P1 issues, the most serious being:

| Issue | Was | Now |
|---|---|---|
| P0-2 | `RiskSupervisor` treated the action *index* as a position *fraction* — every trade rejected; state was memory-only | explicit `check_trade(signal, requested_size, state, market_data)`; SQLite-persisted circuit breakers |
| P0-5 | backtester fabricated `np.random.randn(100)` observations, training commented out, placeholder P&L | archived; new engine on `backtesting.py` with real data, real costs, deterministic seeds |
| P0-4 | look-ahead bias in higher-TF resample, macro daily stamping, full-dataset normalization | `closed='right'` + `shift(1)`; macro `shift(1)`; scaler fit on TRAIN window only |
| P0-1 / P0-6 | live bot: no SL/TP, `VOLUME=0.01` hardcoded, 10s loop, no retcodes, no reconciliation | `live/live_trade_mt5.py` on the new stack (see [Live Trading](#-live-trading)) |

**The backtests are honest — and the strategies do NOT beat buy-and-hold net
of costs.** See the [Backtest Verdict](#-backtest-verdict-honest) section.

---

## ✅ Verified Facts

Current implementation verification is recorded below. The synthetic data,
training artifacts, and command log used for this phase are under `/tmp` and
are not shipped with the repository.

| Check | Result | Evidence |
|---|---|---|
| Full pytest suite | **139 passed** | `.venv/bin/python -m pytest tests env -q` |
| Required verification gates | `verify_config.py` (24), `verify_risk.py` (29), `verify_broker.py` (29), `verify_risk_integration.py` (23) | `scripts/verify_*.py` |
| Synthetic end-to-end workflow | Training, artifact assembly/adaptation, manifest backtest, and bounded model-backed MockBroker run | `/tmp/phase4_e2e.log` |
| MT5 terminal export | Not exercised; MetaTrader5 terminal integration requires Windows | `scripts/export_mt5_history.py --help` and mocked offset test |

---

## 🔍 How It Works

```
OHLCV CSV + optional daily macro CSV
   ↓
train/data.py — chronological split; fit feature scaler on TRAIN only;
   shared causal features and feature_contract.json
   ↓
train/ — PPO, Transformer-PPO, Dreamer, MCTS manifest, ensembles,
   optional MAML adaptation and adversarial fine-tuning
   ↓
manifest + feature_contract + model checkpoint
   ↓
train/evaluate.py — evaluation.json + promotion gate
   ├── backtest/ — rule strategies, baselines, model signals, costs,
   │               purge/embargo walk-forward
   └── live/ — closed-bar features → policy → DecisionEngine
              → TradeExecutor / RiskSupervisor → MockBroker or MT5
```

Every trained policy uses the shared observation width
`window * n_features + 5`; the live executor and risk circuit breakers remain
in place regardless of the selected model. See the detailed
[closed-bar live flow](docs/MODELS.md#live-processing-order).

---

## 🛠️ Installation

### Prerequisites
- Python 3.12+ (the `.venv` in this repo is Python 3.11)
- MetaTrader 5 terminal (live mode only — demo mode needs no MT5)

### Step-by-Step Setup

```bash
# 1. Create venv (Windows)
python -m venv .venv

# 2. Activate
.venv\Scripts\activate.bat

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Copy the environment template and start in DEMO mode (default)
copy .env.example .env
```

All versions in `requirements.txt` are **pinned `==`** to what is verified in
this repo's `.venv` (including `backtesting==0.6.2`, `python-dotenv`,
`metaapi_cloud_sdk`, `MetaTrader5`).

---

## ⚙️ Configuration

Copy `.env.example` → `.env` and edit. `core/config.py` loads `.env` via
python-dotenv and **validates every value**; invalid values raise at startup.
Every supported environment variable is documented in `.env.example`:

- `TRADING_MODE` — `demo` (default) or `live`. Live **refuses to start** unless
  every gate passes (risk state loadable, promoted model when model signals are
  selected, and MT5 credentials configured).
- `SIGNAL_SOURCE` — `rule` (default: causal SMA fast/slow crossover, no model
  needed) or `model` (policy manifest); `ppo` remains an alias for `model`.
- `MODEL_MANIFEST`, `LIVE_HISTORY_BARS`, and optional `MACRO_CSV` configure live
  model inference. `REQUIRE_PROMOTED_MODEL` controls demo-mode promotion gates.
- Decision and trade-management controls include confidence/agreement filters,
  UTC session hours, loss cooldown, Kelly scaling, breakeven/trailing stops,
  partial closes, and a bar-based time stop.
- Risk circuit breakers — `MAX_DAILY_LOSS`, `MAX_DRAWDOWN`,
  `MAX_CONSECUTIVE_LOSSES`, `MAX_RISK_PER_TRADE`, `MAX_POSITION`,
  `MAX_TRADES_PER_DAY`, `MIN_TRADE_INTERVAL_SEC`, `VOL_THRESHOLD`,
  `MAX_SPREAD`, `MARKET_HOURS_ONLY`, `CORRELATION_GUARD_ENABLED`.
- Cost model — `COST_SPREAD`, `COST_COMMISSION`, `COST_SLIPPAGE`, swap rates.
- MT5 — `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_MAGIC`,
  `MT5_DEVIATION`, `BROKER_UTC_OFFSET_HOURS`, `SL_ATR_MULT`,
  `TP_ATR_MULT`, `MAX_REQUOTE_RETRIES`.

> ⚠️ The default production manifest does **not exist** in this repo. The `rule`
> signal source runs standalone; `model`/`ppo` requires a valid model artifact
> and evaluation record and never silently falls back to flat.

---

## 🚀 Quick Start Guide

### 1. Run the verify gates (acceptance scripts)

```bash
python scripts/verify_config.py          # 24/24
python scripts/verify_features.py        # 33/33
python scripts/verify_risk.py            # 29/29
python scripts/verify_backtest.py        # 26/26
python scripts/verify_broker.py          # 29/29
python scripts/verify_risk_integration.py # 23/23
```

### 2. Run the full pytest suite

```bash
python -m pytest tests env -q
# 87 passed, exit 0
```

### 3. Run the new backtest engine on real data

```bash
python backtest/run_backtest.py --data data/xauusd_d1.csv --strategy SmaCrossAtr --seed 42
```

Results (metrics, trades, equity curve, report) are written to
`research/backtest_results/`. See `backtest/report_backtest.md` and
`backtest/README.md` for the honest methodology (causal features, next-bar
fills, costs, deterministic seeds, walk-forward with purge + embargo).

### 4. Run the live loop in demo mode (safe, no real orders)

```bash
python live/live_trade_mt5.py --smoke --max-bars 300
```

This uses a deterministic synthetic bar feed against the **MockBroker** —
entries are gated by the RiskSupervisor, every order carries ATR-based SL/TP,
and fills/PnL are real broker-interface results (see
`artifacts/live_demo_smoke.txt`). No real orders are ever sent in demo mode.

---

## 📁 Project Structure

```
tradingbot/
├── archive/                   # Archived legacy code (fake backtester kept as evidence)
│   └── backtest_engine_legacy_fake.py
├── artifacts/                 # Verification evidence (pytest, gates, smoke, recon, ops)
├── backtest/                  # NEW engine on kernc/backtesting.py
│   ├── engine.py              # Backtest runner
│   ├── strategies.py          # SmaCrossAtr / DonchianBreakout / RsiReversion / ML signal
│   ├── costs.py               # spread + commission + slippage
│   ├── baselines.py           # BuyHold / SeededRandom / SMA cross
│   ├── walk_forward.py        # purge + embargo
│   ├── report_backtest.md     # HONEST results
│   └── README.md / BACKTESTING_API_NOTES.md
├── core/
│   ├── config.py              # validated dataclasses + TRADING_MODE gate
│   └── feature_pipeline.py    # causal features + contract enforcement
├── data/                      # XAUUSD CSVs (xauusd_h1.csv, xauusd_d1.csv, …)
├── env/                       # RL gym environment (+ 13 passing tests)
├── features/                  # feature modules (multi_timeframe, macro, calendar, …)
├── live/
│   ├── broker.py              # BaseBroker ABC
│   ├── mt5_broker.py          # real MT5 (retcodes, REQUOTE retry, idempotency)
│   ├── mock_broker.py         # deterministic simulated fills (demo)
│   ├── trade_executor.py      # RiskSupervisor wired into EVERY order path
│   ├── idempotency.py         # persisted order-idempotency guard
│   ├── live_trade_mt5.py      # candle-close-aligned live loop (demo|live)
│   └── live_trade_metaapi.py  # MetaAPI adapter (secondary)
├── models/
│   ├── risk_supervisor.py     # SQLite circuit breakers
│   ├── position_sizing.py     # Kelly / fixed-fraction / ATR sizing
│   ├── dreamer_agent.py       # DreamerV3 (save/load full training state)
│   └── dreamer_components.py  # RSSM, symlog, two-hot critic
├── scripts/                   # verify gates + evidence runners
├── tests/                     # pytest suite (executor, sizing, risk, features, …)
├── research/                  # audit report, backtest results
├── evaluate_model.py          # honest annualization (--bars-per-year)
├── requirements.txt           # pinned == versions
└── FIXES.md                   # audit issue → fix → evidence mapping
```

---

## 🧪 Testing

| Command | Purpose |
|---|---|
| `python -m pytest tests env -q` | Full suite (87 passed, exit 0) |
| `python scripts/_run_pytest_evidence.py` | Re-run pytest, record evidence to `artifacts/pytest_final.txt` |
| `python scripts/_run_gates_twice.py` | Re-run all six gates ×2, record evidence to `artifacts/gate_evidence.txt` |
| `python scripts/_verify_p1_fixes.py` | Verify P1 code fixes (29/29) |
| `python scripts/_ops_env_crosscheck.py` | Ops cross-check (env keys vs .env.example, pins, README honesty) → `artifacts/ops_evidence.txt` |

The DreamerV3 environment suite (`env/test_dreamer_trading_env.py`, 13 tests)
covers realistic costs, spread, adverse slippage, swap, stop-loss, max
drawdown, event mask, and causal observation — it stays green.

---

## 📉 Backtest Verdict (Honest)

> Source: `backtest/report_backtest.md` — deterministic seed 42, same data,
> same cost model (round-trip 0.000410) for every strategy AND baseline.

| Run | Data | Return % | Buy&Hold % | Sharpe | MaxDD % | #Trades | Beat baseline? |
|---|---|---|---|---|---|---|---|
| SmaCrossAtr | xauusd_d1 | 52.40% | 884.26% | 0.39 | -12.46% | 58 | **NO** |
| RsiReversion | xauusd_d1 | 29.07% | 932.41% | 0.16 | -19.84% | 162 | **NO** |
| DonchianBreakout | xauusd_d1 | 278.94% | 923.11% | 0.56 | -28.69% | 98 | **NO** |
| SmaCrossAtr | xauusd_h1_from_m1 | 8.13% | 137.58% | 0.40 | -7.00% | 214 | **NO** |
| RsiReversion | xauusd_h1_from_m1 | -3.45% | 137.07% | -0.13 | -13.67% | 688 | **NO** |
| DonchianBreakout | xauusd_h1_from_m1 | 44.90% | 138.01% | 0.86 | -17.65% | 295 | **NO** |

**None of the rule-based strategies beat buy-and-hold net of costs.** This is
reported plainly — the audit's original inflated performance targets
(multi-hundred-percent annual returns, Sharpe ratios above 3) were never
backed by a real engine and are **retracted**. A live system MUST NOT run
these strategies as-is.

The ML (PPO) strategy path is implemented (`backtest/strategies.py` signal
strategy) but there is **no trained model checkpoint in this repo** — it
cannot be honestly evaluated without one, and the code refuses to fabricate
results in its absence.

---

## 🔌 Live Trading

`live/live_trade_mt5.py` is the production entry point:

- `TRADING_MODE=demo` (default) — trades only against MockBroker, never sends a
  real order. Smoke: `python live/live_trade_mt5.py --smoke --max-bars 300`.
- `TRADING_MODE=live` — **refuses to start** unless every gate passes:
  a promoted model artifact (for `SIGNAL_SOURCE=model` or `ppo`) + risk state
  loadable + MT5 credentials configured.
- Candle-close-aligned loop (one decision per NEW closed candle — no 10s spam).
- SL/TP (ATR-based) attached to every order; MT5 retcode handling with bounded
  REQUOTE retry; idempotency guard persisted across restarts; startup
  broker-vs-local reconciliation (halt on drift); kill-switch file check.
- Broker bar times are converted to UTC using `BROKER_UTC_OFFSET_HOURS`.
- Export H1 history on a Windows MT5 host with
  `python scripts/export_mt5_history.py --from 2018-01-01 --to 2024-01-01`.
- Structured rotating logging to `logs/`.

> ⚠️ Final live verification requires the user's own MT5 terminal. Demo-mode
> smoke is the sandbox verification path (see `artifacts/live_demo_smoke.txt`).

---

## ⚠️ Disclaimer

**IMPORTANT — PLEASE READ**

This software is provided for **educational and research purposes only**.

- Trading financial instruments involves **substantial risk of loss**.
- Past performance does **NOT** guarantee future results.
- **The strategies in this repo do not beat buy-and-hold net of costs.**
- Automated trading can fail due to bugs, connectivity, or market conditions.
- The authors are **NOT responsible** for any financial losses incurred
  through use of this software. Use at your own risk.

---

## 📄 License

See the [LICENSE](LICENSE) file. Third-party components (notably
`backtesting.py`) have their own licenses (AGPL-3.0 for `backtesting.py` —
review before any commercial use).

---

## 🙏 Acknowledgments

- **[kernc/backtesting.py](https://github.com/kernc/backtesting.py)** — the
  backtest engine this project now builds on
- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** — RL algorithms
- **[Dreamer V3](https://danijar.com/project/dreamerv3/)** — world-model RL
- **MetaTrader 5** — trading platform and data provider
