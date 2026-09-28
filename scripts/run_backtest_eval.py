"""
scripts/run_backtest_eval.py — honest full backtest evaluation (Subtask 4).

Runs every rule strategy and both baselines on:
    * data/xauusd_d1.csv          (D1, 2005-)
    * data/xauusd_h1_from_m1.csv  (H1, 2022-)
with the SAME validated CostModel and a fixed seed, then:

    * saves trades CSV + metrics.json + equity PNG for every run under
      research/backtest_results/full_eval_<ts>/
    * writes backtest/report_backtest.md (tables + HONEST conclusions)

The honesty mandate is enforced by backtest.report.write_report(): every
strategy's net return is compared against the best baseline on the SAME data,
and underperformance is stated explicitly.  No fabricated numbers.

Determinism: backtesting.py is deterministic for fixed strategy+data; the only
stochastic element (SeededRandom) uses np.random.default_rng(SEED).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd

from backtest.baselines import BuyHold, SeededRandom
from backtest.costs import CostModel
from backtest.engine import prepare_ohlc, run_backtest, walk_forward, summarize_walk_forward
from backtest.report import save_results, write_report
from backtest.strategies import SmaCrossAtr, RsiReversion, DonchianBreakout

SEED = 42

COST = CostModel.from_config(
    __import__("core.config", fromlist=["CostConfig"]).CostConfig()
)

DATA_FILES = {
    "xauusd_d1": REPO / "data" / "xauusd_d1.csv",
    "xauusd_h1_from_m1": REPO / "data" / "xauusd_h1_from_m1.csv",
}

STRATEGIES = [
    SmaCrossAtr,
    RsiReversion,
    DonchianBreakout,
]


def load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return prepare_ohlc(df)


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "research" / "backtest_results" / f"full_eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []      # strategy runs
    all_baselines = [] # baseline runs

    for data_name, path in DATA_FILES.items():
        print(f"\n=== {data_name}: loading {path.name} ===")
        ohlc = load_ohlc(path)
        print(f"    {len(ohlc)} bars, {ohlc.index.min()} -> {ohlc.index.max()}")

        for strat in STRATEGIES:
            print(f"  running {strat.__name__} ...")
            res = run_backtest(ohlc, strat, cost=COST, data_name=data_name)
            save_results(res, out_dir, seed=SEED)
            all_runs.append(res)
            m = res.metrics
            print(f"    trades={m['num_trades']} return={m['total_return_pct']:.2f}% "
                  f"sharpe={m['sharpe']:.2f} maxDD={m['max_drawdown_pct']:.2f}%")

        print("  running BuyHold ...")
        bh = run_backtest(ohlc, BuyHold, cost=COST, data_name=data_name,
                          strategy_name="BuyHold")
        save_results(bh, out_dir, seed=SEED)
        all_baselines.append(bh)
        print(f"    return={bh.metrics['total_return_pct']:.2f}%")

        print(f"  running SeededRandom(seed={SEED}) ...")
        sr = run_backtest(ohlc, SeededRandom, cost=COST, data_name=data_name,
                          strategy_name="SeededRandom",
                          strategy_params={"seed": SEED})
        save_results(sr, out_dir, seed=SEED)
        all_baselines.append(sr)
        print(f"    trades={sr.metrics['num_trades']} "
              f"return={sr.metrics['total_return_pct']:.2f}% "
              f"sharpe={sr.metrics['sharpe']:.2f}")

        # Zero-cost comparison for ONE strategy per dataset (cost honesty)
        print(f"  zero-cost comparison (SmaCrossAtr) ...")
        zc = CostModel(spread=0.0, commission=0.0, slippage=0.0)
        res_zc = run_backtest(ohlc, SmaCrossAtr, cost=zc, data_name=data_name,
                              strategy_name="SmaCrossAtr_zero_cost")
        save_results(res_zc, out_dir, seed=SEED)
        all_runs.append(res_zc)

    # --- Out-of-sample walk-forward on D1 (purge/embargo) ------------------- #
    print("\n=== walk-forward OOS (D1, purge/embargo) ===")
    ohlc_d1 = load_ohlc(DATA_FILES["xauusd_d1"])
    for strat in (SmaCrossAtr, RsiReversion, DonchianBreakout):
        wf = walk_forward(ohlc_d1, strat, COST, train_bars=1200, test_bars=400,
                          embargo_bars=25, data_name="xauusd_d1")
        summ = summarize_walk_forward(wf)
        print(f"  {strat.__name__}: windows={summ.get('n_windows')} "
              f"trades={summ.get('total_trades')} "
              f"meanReturn={summ.get('total_return_pct', 0.0):.2f}% "
              f"meanSharpe={summ.get('sharpe', 0.0):.2f}")
        for i, w in enumerate(wf):
            save_results(w, out_dir / "walk_forward", seed=SEED)

    # --- Report ------------------------------------------------------------- #
    print("\n=== writing backtest/report_backtest.md ===")
    notes = (
        "## Data\n\n"
        "| File | TF | Bars | Range |\n|---|---|---|---|\n"
        "| `data/xauusd_d1.csv` | D1 | 6,787 | 2005-01-02 → 2023-08 |\n"
        "| `data/xauusd_h1_from_m1.csv` | H1 | 23,657 | 2022-01-02 23:00 → 2024-12 |\n\n"
        "All strategy and baseline runs use the **same** validated cost model "
        f"(`CostModel.from_config(CostConfig())` → `{COST.describe()}`) and seed `{SEED}`.\n\n"
        "## Method\n\n"
        "* Market orders fill on the next bar's open (backtesting.py default; "
        "`trade_on_close=False`), so there is no look-ahead in fills.\n"
        "* Every strategy order carries an ATR-based SL and TP; strategies close "
        "positions on their exit rule.  BuyHold is intentionally stopless "
        "(buy-and-hold by definition).\n"
        "* Walk-forward on D1 uses train/test/embargo of 1200/400/25 bars "
        "(purge + embargo per López de Prado) — test windows are strictly "
        "out-of-sample with no warm-up leakage.\n"
        "* Zero-cost comparison is included so the true cost drag is visible.\n"
    )
    write_report(
        REPO / "backtest" / "report_backtest.md",
        "Backtest evaluation report — NEW backtesting.py engine (honest)",
        runs=all_runs,
        baselines=all_baselines,
        seed=SEED,
        notes=notes,
    )

    # --- save a small manifest ---------------------------------------------- #
    manifest = {
        "created": ts,
        "cost_model": COST.describe(),
        "seed": SEED,
        "runs": [f"{r.strategy_name}|{r.data_name}" for r in all_runs],
        "baselines": [f"{b.strategy_name}|{b.data_name}" for b in all_baselines],
        "walk_forward": {
            "method": "purge+embargo (train 1200 / test 400 / embargo 25)",
            "datasets": ["xauusd_d1"],
        },
    }
    (out_dir / "manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"results in: {out_dir}")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
