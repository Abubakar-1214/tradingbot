"""
backtest/baselines.py — honest comparison baselines.

The mandate: "if the model does not beat baselines, SAY SO".  These baselines
run through the SAME backtesting.py engine with the SAME cost model, so any
strategy outperformance claim is against a like-for-like benchmark.

Baselines:
    * BuyHold         — buy at the first bar and hold to the end.
    * SeededRandom    — deterministic (seeded) random entries with ATR SL/TP
                        and a bounded holding period.  Seed fixes the RNG, so
                        the baseline is reproducible across runs.
"""
from __future__ import annotations

import numpy as np

from backtesting import Strategy

from backtest.strategies import atr


class BuyHold(Strategy):
    """Buy-and-hold benchmark: enter a long on the first bar, never exit."""

    def init(self) -> None:
        self._entered = False

    def next(self) -> None:
        if not self._entered:
            self.buy(size=0.9999, tag="BUY_HOLD")
            self._entered = True


class SeededRandom(Strategy):
    """Deterministic random-entropy baseline with the same cost model.

    ``seed`` makes the entry stream reproducible.  Entries occur with
    probability ``entry_prob`` per bar (long only unless ``allow_short``);
    every trade carries an ATR-based SL/TP and a maximum holding period, so
    the baseline pays the same cost structure as the rule strategies.
    """

    seed: int = 42
    entry_prob: float = 0.03
    max_bars: int = 50
    sl_atr: float = 3.0
    tp_atr: float = 6.0
    allow_short: bool = False

    def init(self) -> None:
        self.rng = np.random.default_rng(int(self.seed))
        self.atr_s = self.I(atr, self.data.High, self.data.Low, self.data.Close, 14, name="ATR")
        self._entry_bar: int = -1

    def next(self) -> None:
        if self.position:
            if len(self.data) - self._entry_bar >= int(self.max_bars):
                self.position.close()
            return
        if float(self.rng.random()) < self.entry_prob:
            self._entry_bar = len(self.data)
            entry = float(self.data.Close[-1])
            if self.allow_short and float(self.rng.random()) < 0.5:
                self.sell(sl=entry + float(self.atr_s[-1]) * self.sl_atr,
                          tp=entry - float(self.atr_s[-1]) * self.tp_atr,
                          tag="RANDOM_SHORT")
            else:
                self.buy(sl=entry - float(self.atr_s[-1]) * self.sl_atr,
                         tp=entry + float(self.atr_s[-1]) * self.tp_atr,
                         tag="RANDOM_LONG")
