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

Everything below was re-run in the current cycle and the evidence is recorded
in `artifacts/`:

| Check | Result | Evidence |
|---|---|---|
| Full pytest suite | **87 passed, 0 failed, exit 0** | `artifacts/pytest_final.txt` |
| Six verify gates × 2 runs | **12/12 runs exit 0** (config 24/24, features 33/33, risk 29/29, backtest 26/26, broker 29/29, risk_integration 23/23) | `artifacts/gate_evidence.txt` |
| P1 code fixes | **29/29 checks** (position sizing, Dreamer save/load, ReplayBuffer, evaluate_model) | `scripts/_verify_p1_fixes.py` |
| Live loop demo smoke (demo mode) | exit 0, **10 real entries (long + short) with real ATR-based SL/TP, 10 real SL/TP fills with real PnL** through MockBroker | `artifacts/live_demo_smoke.txt` |
| Feature pipeline | 39 causal features, train-window scaler only, contract enforced at load | `scripts/verify_features.py` |
| Ops artifacts | pinned `requirements.txt` (20 `==` pins incl. `backtesting==0.6.2`), `.env` created + gitignored, every env key documented in `.env.example`, `pip check` clean | `artifacts/ops_evidence.txt` |

---

## 🔍 How It Works

```
data/ (XAUUSD H1/D1 CSVs)
   ↓
core/feature_pipeline.py — causal features (no future leakage),
   scaler fit on TRAIN window only, feature_contract.json enforced at load
   ↓
backtest/ — NEW engine on kernc/backtesting.py:
   strategies.py (SmaCrossAtr, DonchianBreakout, RsiReversion, ML signal)
   costs.py (spread + commission + slippage)
   walk_forward.py (purge + embargo per López de Prado)
   baselines.py (BuyHold, SeededRandom, SMA cross)
   engine.py (Backtest runner + deterministic seeds)
   ↓
live/ — production stack:
   broker.py (ABC) → mt5_broker.py (real MT5, retcode handling, idempotency)
                   → mock_broker.py (deterministic simulated fills)
   trade_executor.py (RiskSupervisor wired into EVERY order path)
   risk_supervisor.py (SQLite circuit breakers)
   live_trade_mt5.py (candle-close-aligned loop, TRADING_MODE=demo|live)
```

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
Every environment variable (49 documented keys) is documented in
`.env.example`:

- `TRADING_MODE` — `demo` (default) or `live`. Live **refuses to start** unless
  every gate passes (feature contract present, model present, risk state
  loadable, reconciliation passed, MT5 credentials configured).
- `SIGNAL_SOURCE` — `rule` (default: causal SMA fast/slow crossover, no model
  needed) or `ppo` (requires `MODEL_PATH` + `FEATURE_CONTRACT_PATH`).
- Risk circuit breakers — `MAX_DAILY_LOSS`, `MAX_DRAWDOWN`,
  `MAX_CONSECUTIVE_LOSSES`, `MAX_RISK_PER_TRADE`, `MAX_POSITION`,
  `MAX_TRADES_PER_DAY`, `MIN_TRADE_INTERVAL_SEC`, `VOL_THRESHOLD`,
  `MAX_SPREAD`, `MARKET_HOURS_ONLY`, `CORRELATION_GUARD_ENABLED`.
- Cost model — `COST_SPREAD`, `COST_COMMISSION`, `COST_SLIPPAGE`, swap rates.
- MT5 — `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_MAGIC`,
  `MT5_DEVIATION`, `SL_ATR_MULT`, `TP_ATR_MULT`, `MAX_REQUOTE_RETRIES`.

> ⚠️ The default `MODEL_PATH=train/ppo_xauusd_latest.zip` and
> `FEATURE_CONTRACT_PATH=train/feature_contract.json` do **not exist** in this
> repo. The `rule` signal source runs standalone; the `ppo` source raises a
> clear error when the model is missing (never silently flat).

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
  feature contract present + model present (for `SIGNAL_SOURCE=ppo`) + risk
  state loadable + MT5 credentials configured.
- Candle-close-aligned loop (one decision per NEW closed candle — no 10s spam).
- SL/TP (ATR-based) attached to every order; MT5 retcode handling with bounded
  REQUOTE retry; idempotency guard persisted across restarts; startup
  broker-vs-local reconciliation (halt on drift); kill-switch file check.
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
