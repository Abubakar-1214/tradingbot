"""
backtest/strategies.py — causal rule-based strategies as backtesting.Strategy.

Every indicator is computed with rolling windows only (no future bars), and
market orders are filled on the NEXT bar's open (backtesting.py default), so
these strategies are look-ahead free by construction.

Strategies:
    * SmaCrossAtr      — fast/slow SMA crossover with ATR-based SL/TP.
    * RsiReversion     — RSI oversold/overbought mean reversion with SL/TP.
    * DonchianBreakout — N-bar high/low breakout (turtle-style) with SL/TP.
    * MlSignalStrategy — consumes a PRE-COMPUTED causal signal column
      (e.g. produced offline by a fitted model + feature contract).  It is
      a real bridge for ML signals, not a fake: if ``signal_col`` is not
      present in the data the strategy refuses to trade.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from backtesting import Strategy
from backtesting.lib import crossover


# --------------------------------------------------------------------------- #
# Indicator helpers (pure pandas on arrays) — causal by construction
# --------------------------------------------------------------------------- #
def sma(series: Sequence[float], n: int) -> np.ndarray:
    return pd.Series(np.asarray(series, dtype=float)).rolling(int(n)).mean().to_numpy()


def atr(high: Sequence[float], low: Sequence[float], close: Sequence[float],
        period: int = 14) -> np.ndarray:
    h = pd.Series(np.asarray(high, dtype=float))
    l = pd.Series(np.asarray(low, dtype=float))
    c = pd.Series(np.asarray(close, dtype=float))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(int(period)).mean().to_numpy()


def rsi(close: Sequence[float], period: int = 14) -> np.ndarray:
    c = pd.Series(np.asarray(close, dtype=float))
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(int(period)).mean()
    loss = (-delta.clip(upper=0)).rolling(int(period)).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0).to_numpy()


def rolling_max(series: Sequence[float], n: int) -> np.ndarray:
    return pd.Series(np.asarray(series, dtype=float)).rolling(int(n)).max().to_numpy()


def rolling_min(series: Sequence[float], n: int) -> np.ndarray:
    return pd.Series(np.asarray(series, dtype=float)).rolling(int(n)).min().to_numpy()


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
class SmaCrossAtr(Strategy):
    """Fast/slow SMA crossover with ATR-based stop-loss and take-profit."""

    fast: int = 20
    slow: int = 60
    atr_period: int = 14
    sl_atr: float = 2.0
    tp_atr: float = 3.0
    allow_short: bool = False

    def init(self) -> None:
        self.fast_ma = self.I(sma, self.data.Close, int(self.fast), name=f"SMA{int(self.fast)}")
        self.slow_ma = self.I(sma, self.data.Close, int(self.slow), name=f"SMA{int(self.slow)}")
        self.atr_s = self.I(atr, self.data.High, self.data.Low, self.data.Close,
                            int(self.atr_period), name="ATR")

    def next(self) -> None:
        if not self.position:
            if crossover(self.fast_ma, self.slow_ma):
                entry = float(self.data.Close[-1])
                self.buy(sl=entry - float(self.atr_s[-1]) * self.sl_atr,
                         tp=entry + float(self.atr_s[-1]) * self.tp_atr,
                         tag="SMA_LONG")
            elif self.allow_short and crossover(self.slow_ma, self.fast_ma):
                entry = float(self.data.Close[-1])
                self.sell(sl=entry + float(self.atr_s[-1]) * self.sl_atr,
                          tp=entry - float(self.atr_s[-1]) * self.tp_atr,
                          tag="SMA_SHORT")
        else:
            if self.position.is_long and crossover(self.slow_ma, self.fast_ma):
                self.position.close()
            elif self.position.is_short and crossover(self.fast_ma, self.slow_ma):
                self.position.close()


class RsiReversion(Strategy):
    """RSI mean reversion: buy oversold, sell overbought, exit at neutral 50."""

    rsi_period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    sl_atr: float = 2.0
    tp_atr: float = 3.0
    allow_short: bool = False

    def init(self) -> None:
        self.rsi_s = self.I(rsi, self.data.Close, int(self.rsi_period), name="RSI")
        self.atr_s = self.I(atr, self.data.High, self.data.Low, self.data.Close, 14, name="ATR")

    def next(self) -> None:
        if not self.position:
            if float(self.rsi_s[-1]) < self.oversold:
                entry = float(self.data.Close[-1])
                self.buy(sl=entry - float(self.atr_s[-1]) * self.sl_atr,
                         tp=entry + float(self.atr_s[-1]) * self.tp_atr,
                         tag="RSI_OVERSOLD")
            elif self.allow_short and float(self.rsi_s[-1]) > self.overbought:
                entry = float(self.data.Close[-1])
                self.sell(sl=entry + float(self.atr_s[-1]) * self.sl_atr,
                          tp=entry - float(self.atr_s[-1]) * self.tp_atr,
                          tag="RSI_OVERBOUGHT")
        else:
            if self.position.is_long and float(self.rsi_s[-1]) > 50.0:
                self.position.close()
            elif self.position.is_short and float(self.rsi_s[-1]) < 50.0:
                self.position.close()


class DonchianBreakout(Strategy):
    """N-bar high/low breakout (turtle-style) with ATR stop-loss/take-profit.

    A long is taken when today's close exceeds the highest high of the PREVIOUS
    ``entry_n`` bars (``upper[-2]``) — the classic causal formulation; the fill
    occurs on the next bar's open.
    """

    entry_n: int = 40
    exit_n: int = 20
    sl_atr: float = 3.0
    tp_atr: float = 6.0
    allow_short: bool = False

    def init(self) -> None:
        self.upper = self.I(rolling_max, self.data.High, int(self.entry_n), name="DC_HIGH")
        self.lower = self.I(rolling_min, self.data.Low, int(self.entry_n), name="DC_LOW")
        self.exit_low = self.I(rolling_min, self.data.Low, int(self.exit_n), name="EXIT_LOW")
        self.exit_high = self.I(rolling_max, self.data.High, int(self.exit_n), name="EXIT_HIGH")
        self.atr_s = self.I(atr, self.data.High, self.data.Low, self.data.Close, 14, name="ATR")

    def next(self) -> None:
        if not self.position:
            if float(self.data.Close[-1]) > float(self.upper[-2]):
                entry = float(self.data.Close[-1])
                self.buy(sl=entry - float(self.atr_s[-1]) * self.sl_atr,
                         tp=entry + float(self.atr_s[-1]) * self.tp_atr,
                         tag="DC_LONG")
            elif self.allow_short and float(self.data.Close[-1]) < float(self.lower[-2]):
                entry = float(self.data.Close[-1])
                self.sell(sl=entry + float(self.atr_s[-1]) * self.sl_atr,
                          tp=entry - float(self.atr_s[-1]) * self.tp_atr,
                          tag="DC_SHORT")
        else:
            if self.position.is_long and float(self.data.Close[-1]) < float(self.exit_low[-1]):
                self.position.close()
            elif self.position.is_short and float(self.data.Close[-1]) > float(self.exit_high[-1]):
                self.position.close()


class MlSignalStrategy(Strategy):
    """Consume a PRE-COMPUTED causal signal column produced offline.

    The ``signal_col`` column must already be causally shifted by the producer
    (e.g. a fitted model + the leak-free ``core.feature_pipeline`` contract).
    Signal semantics: >0 → long, <0 → short, ==0 → flat.  SL/TP are ATR based.
    If the column is absent the strategy never trades (explicit fail-safe).
    """

    signal_col: str = "signal"
    atr_period: int = 14
    sl_atr: float = 2.0
    tp_atr: float = 3.0
    allow_short: bool = False

    def init(self) -> None:
        self.signals = self.I(self._read_signal, name=self.signal_col)
        self.atr_s = self.I(atr, self.data.High, self.data.Low, self.data.Close,
                            int(self.atr_period), name="ATR")

    def _read_signal(self, *_args) -> np.ndarray:
        df = self.data.df
        if self.signal_col not in df.columns:
            return np.zeros(len(df))
        return df[self.signal_col].to_numpy(dtype=float)

    def next(self) -> None:
        sig = float(self.signals[-1])
        if not self.position:
            if sig > 0:
                entry = float(self.data.Close[-1])
                self.buy(sl=entry - float(self.atr_s[-1]) * self.sl_atr,
                         tp=entry + float(self.atr_s[-1]) * self.tp_atr,
                         tag="ML_LONG")
            elif self.allow_short and sig < 0:
                entry = float(self.data.Close[-1])
                self.sell(sl=entry + float(self.atr_s[-1]) * self.sl_atr,
                          tp=entry - float(self.atr_s[-1]) * self.tp_atr,
                          tag="ML_SHORT")
        else:
            if sig == 0:
                self.position.close()
            elif self.position.is_long and sig < 0:
                self.position.close()
            elif self.position.is_short and sig > 0:
                self.position.close()
