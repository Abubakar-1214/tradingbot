# backtest/ — the NEW backtest engine

This package replaces the archived legacy fake engine
(`archive/backtest_engine_legacy_fake.py`) with a real, honest backtesting
pipeline built on **[kernc/backtesting.py](https://github.com/kernc/backtesting.py) 0.6.2**
(installed in `.venv`; API verified by introspection in
`BACKTESTING_API_NOTES.md`).

## Why a new engine

The old `backtest/backtest_engine.py` generated its "observations" with
`np.random.randn(100)` — every backtest was a coin flip wearing a lab coat.
It is archived at `archive/backtest_engine_legacy_fake.py` and **must not be
imported**. This package is a ground-up replacement.

## Modules

| Module | Purpose |
|---|---|
| `costs.py` | Explicit cost model from `core.config.CostConfig` → the exact `spread` / `commission` args backtesting.py consumes. |
| `strategies.py` | Causal rule-based strategies: `SmaCrossAtr`, `RsiReversion`, `DonchianBreakout`, plus `MlSignalStrategy` (consumes a pre-shifted causal signal column). |
| `baselines.py` | Honest baselines through the SAME engine and cost model: `BuyHold`, `SeededRandom` (deterministic via `np.random.default_rng(seed)`). |
| `engine.py` | `prepare_ohlc`, `run_backtest`, `walk_forward` (with purge/embargo gap). |
| `report.py` | Writes metrics JSON, trades CSV, equity PNG and `report_backtest.md` with HONEST conclusions. |

## Honesty rules (enforced)

1. **Real data only** — engine runs on `data/xauusd_*.csv`, never synthetic fills.
2. **Deterministic** — same seed + data + strategy ⇒ identical equity curve.
3. **Real costs** — `CostModel` maps validated config values; the report states
   the exact mapping (backtesting.py has no slippage kwarg: one-way
   `spread = round_trip_spread/2 + slippage`).
4. **Causal only** — indicators are rolling/lag; market fills happen on the
   next bar's open; `MlSignalStrategy` refuses to trade without a pre-shifted
   signal column.
5. **No manufactured numbers** — if a strategy does not beat its baseline net
   of costs, `report_backtest.md` says so explicitly.

## Usage

```python
from core.config import load_config, CostConfig
from backtest.costs import CostModel
from backtest.engine import run_backtest
from backtest.strategies import SmaCrossAtr

cfg = load_config()
cost = CostModel.from_config(cfg.cost)
result = run_backtest(ohlc_df, SmaCrossAtr, cost=cost, data_name="xauusd_d1")
print(result.metrics)
```

## Verification

```bat
.venv\Scripts\python.exe scripts\verify_backtest.py
```

The gate checks: real trades, SL/TP on every trade, costs reduce returns,
same-seed determinism, full metrics dict, and causality of rolling indicators.
