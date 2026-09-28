"""
scripts/probe_null_sl.py — root-cause the null SL trade produced by the new
engine on the D1 subset.  Prints the offending trade row with entry/exit bars
and times so the cause (end-of-data force close, indicator warm-up, or stop
fill) can be identified precisely before fixing the strategy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.config import CostConfig
from backtest.costs import CostModel
from backtest.engine import prepare_ohlc, run_backtest
from backtest.strategies import SmaCrossAtr, RsiReversion, DonchianBreakout


def load_d1_subset(n: int = 900) -> pd.DataFrame:
    return pd.read_csv(REPO / "data" / "xauusd_d1.csv").head(n)


def main() -> None:
    df = load_d1_subset(900)
    ohlc = prepare_ohlc(df)
    cost = CostModel.from_config(CostConfig())

    for strat in (SmaCrossAtr, RsiReversion, DonchianBreakout):
        r = run_backtest(ohlc, strat, cost=cost, data_name="d1_subset")
        t = r.trades
        print(f"\n=== {strat.__name__}: {len(t)} trades ===")
        null_sl = t[t["SL"].isna()]
        null_tp = t[t["TP"].isna()]
        print(f"null SL: {len(null_sl)}  null TP: {len(null_tp)}")
        if len(null_sl):
            print("null-SL rows:")
            print(null_sl.to_string())
        if len(t):
            print("all trades head:")
            print(t.head(5).to_string())


if __name__ == "__main__":
    main()
