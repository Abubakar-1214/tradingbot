"""
backtest/engine.py — the NEW backtest runner on kernc/backtesting.py 0.6.2.

Responsibilities:
    * prepare_ohlc()   — adapt a CSV OHLC frame to the exact format
      backtesting.py needs (capitalized Open/High/Low/Close/Volume columns,
      pandas DatetimeIndex, no lookahead columns).
    * run_backtest()   — deterministic single-window backtest with the explicit
      CostModel, returning a typed BacktestResult with trades + metrics.
    * walk_forward()   — honest out-of-sample walk-forward with a purge/embargo
      gap between train and test windows (Lopez de Prado), so no test bar's
      statistics can leak into the in-sample decision process.

Determinism: backtesting.py is deterministic for a fixed strategy + data
(verified by scripts/introspect_backtesting.py — two identical run() calls
produce identical trades and identical final equity).  The only stochastic
baseline (SeededRandom) is seeded via np.random.default_rng(seed), so every
run is reproducible.

No lookahead: strategies consume only rolling windows / pre-shifted signal
columns; market orders fill on the next bar's open (backtesting.py default
when trade_on_close=False).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd

from backtesting import Backtest

from backtest.costs import CostModel
from backtest.strategies import SmaCrossAtr

# Column names required by backtesting.py
OHLCV = ("Open", "High", "Low", "Close", "Volume")


@dataclass
class BacktestResult:
    """Typed result of one backtest run."""

    strategy_name: str
    data_name: str
    start: pd.Timestamp
    end: pd.Timestamp
    stats: pd.Series
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    cost_model: CostModel
    params: Dict[str, object] = field(default_factory=dict)

    # --- convenience metrics ------------------------------------------------- #
    @property
    def metrics(self) -> Dict[str, float]:
        s = self.stats
        n_trades = int(s["# Trades"])
        return {
            "total_return_pct": float(s["Return [%]"]),
            "buy_hold_return_pct": float(s["Buy & Hold Return [%]"]),
            "annual_return_pct": float(s["Return (Ann.) [%]"]),
            "sharpe": float(s["Sharpe Ratio"]),
            "sortino": float(s["Sortino Ratio"]),
            "calmar": float(s["Calmar Ratio"]),
            "max_drawdown_pct": float(s["Max. Drawdown [%]"]),
            "win_rate_pct": float(s["Win Rate [%]"]),
            "profit_factor": float(s["Profit Factor"]),
            "expectancy_pct": float(s["Expectancy [%]"]),
            "sqn": float(s["SQN"]),
            "num_trades": n_trades,
            "exposure_time_pct": float(s["Exposure Time [%]"]),
            "commissions_usd": float(s.get("Commissions [$]", 0.0)),
            "equity_final": float(s["Equity Final [$]"]),
            "equity_peak": float(s["Equity Peak [$]"]),
            "avg_trade_pct": float(s["Avg. Trade [%]"]),
            "best_trade_pct": float(s["Best Trade [%]"]),
            "worst_trade_pct": float(s["Worst Trade [%]"]),
        }

    def trades_csv_path(self, out_dir: Path) -> Path:
        p = out_dir / f"trades_{self.strategy_name}_{self.data_name}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        self.trades.to_csv(p, index=False)
        return p


def prepare_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a lower-case OHLC CSV frame into backtesting.py's input format.

    Accepts columns: time/open/high/low/close[/volume|tick_volume] or already
    capitalized.  Returns a DataFrame with a DatetimeIndex and the exact
    OHLCV capitalized columns backtesting.py 0.6.2 requires.
    """
    out = df.copy()
    lower_to_upper = {"open": "Open", "high": "High", "low": "Low",
                      "close": "Close", "volume": "Volume",
                      "tick_volume": "Volume"}
    rename = {}
    for lk, uk in lower_to_upper.items():
        if lk in out.columns and uk not in out.columns:
            rename[lk] = uk
    out = out.rename(columns=rename)

    missing = [c for c in ("Open", "High", "Low", "Close") if c not in out.columns]
    if missing:
        raise ValueError(f"prepare_ohlc missing required columns: {missing}; got {list(out.columns)}")

    for c in ("Open", "High", "Low", "Close"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "Volume" not in out.columns:
        out["Volume"] = 1.0
    else:
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0.0)

    if isinstance(out.index, pd.DatetimeIndex):
        pass
    elif "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        out = out.set_index("time")
    else:
        raise ValueError("prepare_ohlc requires a 'time' column or DatetimeIndex")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if not (out["High"] >= out[["Open", "Close", "Low"]].max(axis=1)).all():
        raise ValueError("Invalid OHLC: high < max(open, close, low)")
    if not (out["Low"] <= out[["Open", "Close", "High"]].min(axis=1)).all():
        raise ValueError("Invalid OHLC: low > min(open, close, high)")
    return out[[c for c in OHLCV if c in out.columns]]


def run_backtest(
    data: pd.DataFrame,
    strategy: Type,
    cost: Union[CostModel, None] = None,
    cash: float = 10_000.0,
    trade_on_close: bool = False,
    strategy_params: Optional[Dict[str, object]] = None,
    data_name: str = "data",
    strategy_name: Optional[str] = None,
) -> BacktestResult:
    """Run one deterministic backtest and return typed results."""
    ohlc = prepare_ohlc(data)
    cost = cost or CostModel()
    params = dict(strategy_params or {})

    bt = Backtest(
        ohlc,
        strategy,
        cash=cash,
        spread=cost.spread,
        commission=cost.commission,
        trade_on_close=trade_on_close,
        exclusive_orders=True,
    )
    stats = bt.run(**params)

    trades = stats["_trades"].copy()
    equity_curve = stats["_equity_curve"].copy()

    if strategy_name is None:
        strategy_name = strategy.__name__
    return BacktestResult(
        strategy_name=strategy_name,
        data_name=data_name,
        start=stats["Start"],
        end=stats["End"],
        stats=stats,
        trades=trades,
        equity_curve=equity_curve,
        cost_model=cost,
        params=params,
    )


def walk_forward(
    data: pd.DataFrame,
    strategy: Type,
    cost: CostModel,
    train_bars: int = 800,
    test_bars: int = 300,
    embargo_bars: int = 25,
    cash: float = 10_000.0,
    trade_on_close: bool = False,
    data_name: str = "data",
) -> List[BacktestResult]:
    """Honest walk-forward backtest with purge/embargo between windows.

    The train window is used ONLY to fit/select the strategy parameters (the
    caller passes already-fitted parameters via the strategy class defaults —
    here we simply leave them fixed to avoid in-sample optimisation), and the
    test window is evaluated strictly out-of-sample.  ``embargo_bars`` is
    dropped between train end and test start so the features' rolling windows
    cannot contain test bars at train time.
    """
    ohlc = prepare_ohlc(data)
    n = len(ohlc)
    results: List[BacktestResult] = []
    start = 0
    window = 0
    while start + train_bars + embargo_bars + test_bars <= n:
        train_end = start + train_bars
        test_start = train_end + embargo_bars
        test_end = test_start + test_bars

        train = ohlc.iloc[start:train_end]
        test = ohlc.iloc[test_start:test_end]

        # (No hyper-parameter search in this release — parameters come from
        # config/defaults; this is the honest fixed-rule walk-forward.)
        bt = Backtest(test, strategy, cash=cash, spread=cost.spread,
                      commission=cost.commission, trade_on_close=trade_on_close,
                      exclusive_orders=True)
        stats = bt.run()
        results.append(
            BacktestResult(
                strategy_name=strategy.__name__,
                data_name=f"{data_name}_wf{window}",
                start=stats["Start"],
                end=stats["End"],
                stats=stats,
                trades=stats["_trades"].copy(),
                equity_curve=stats["_equity_curve"].copy(),
                cost_model=cost,
                params={"train_bars": train_bars, "embargo_bars": embargo_bars,
                        "test_bars": test_bars, "window": window},
            )
        )
        window += 1
        start += test_bars
    return results


def summarize_walk_forward(results: Sequence[BacktestResult]) -> Dict[str, float]:
    """Aggregate per-window metrics into one honest summary (equal-weight)."""
    if not results:
        return {}
    keys = [
        "total_return_pct", "annual_return_pct", "sharpe", "sortino",
        "max_drawdown_pct", "win_rate_pct", "profit_factor", "num_trades",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r.metrics[k]) for r in results]
        out[k] = float(np.mean(vals))
    out["n_windows"] = float(len(results))
    out["total_trades"] = float(sum(int(r.metrics["num_trades"]) for r in results))
    return out
