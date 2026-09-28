"""backtest — the NEW backtest engine built on kernc/backtesting.py (0.6.2).

Modules:
    costs.py      — explicit cost model (spread / commission / slippage)
    strategies.py — causal rule-based strategies (SMA, RSI, Donchian breakout)
    baselines.py  — honest baselines (buy & hold, seeded random)
    engine.py     — backtesting.py runner + walk-forward with purge/embargo
    report.py     — metrics JSON/MD, trades CSV, equity-curve PNG
"""

from backtest.costs import CostModel
from backtest.engine import BacktestResult, prepare_ohlc, run_backtest, walk_forward

__all__ = [
    "CostModel",
    "BacktestResult",
    "prepare_ohlc",
    "run_backtest",
    "walk_forward",
]
