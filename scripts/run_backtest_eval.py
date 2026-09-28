"""
scripts/run_backtest_eval.py — honest full backtest evaluation (Subtask 4).

Runs every rule strategy and both baselines on:
    * data/xauusd_d1.csv          (D1, 2005-), or --d1-data
    * data/xauusd_h1_from_m1.csv  (H1, 2022-), or --h1-data
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

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from backtest.baselines import BuyHold, SeededRandom
from backtest.costs import CostModel
from backtest.engine import (
    prepare_ohlc,
    run_backtest,
    summarize_walk_forward,
    walk_forward,
)
from backtest.model_signals import model_signal_frame
from backtest.report import save_results, write_report
from backtest.strategies import (
    DonchianBreakout,
    MlSignalStrategy,
    RsiReversion,
    SmaCrossAtr,
)
from core.model_artifacts import load_manifest
from train.data import load_bars

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the honest full backtest evaluation")
    parser.add_argument("--manifest", help="optional production model manifest")
    parser.add_argument(
        "--d1-data", type=Path, default=DATA_FILES["xauusd_d1"],
        help="D1 OHLC CSV (default: data/xauusd_d1.csv)",
    )
    parser.add_argument(
        "--h1-data", type=Path, default=DATA_FILES["xauusd_h1_from_m1"],
        help="H1 OHLC CSV (default: data/xauusd_h1_from_m1.csv)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="base directory for timestamped backtest results",
    )
    parser.add_argument(
        "--report-path", type=Path,
        help="backtest report path (default: backtest/report_backtest.md)",
    )
    args = parser.parse_args(argv)
    data_files = {
        "xauusd_d1": args.d1_data,
        "xauusd_h1_from_m1": args.h1_data,
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir or REPO / "research" / "backtest_results"
    if not output_root.is_absolute():
        output_root = REPO / output_root
    out_dir = output_root / f"full_eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []      # strategy runs
    all_baselines = [] # baseline runs
    dataset_summaries = []

    for data_name, path in data_files.items():
        print(f"\n=== {data_name}: loading {path.name} ===")
        ohlc = load_ohlc(path)
        print(f"    {len(ohlc)} bars, {ohlc.index.min()} -> {ohlc.index.max()}")
        timeframe = "D1" if data_name == "xauusd_d1" else "H1"
        dataset_summaries.append(
            (path, timeframe, len(ohlc), ohlc.index.min(), ohlc.index.max())
        )

        for strat in STRATEGIES:
            print(f"  running {strat.__name__} ...")
            res = run_backtest(ohlc, strat, cost=COST, data_name=data_name)
            save_results(res, out_dir, seed=SEED)
            all_runs.append(res)
            m = res.metrics
            print(f"    trades={m['num_trades']} return={m['total_return_pct']:.2f}% "
                  f"sharpe={m['sharpe']:.2f} maxDD={m['max_drawdown_pct']:.2f}%")

        manifest = load_manifest(args.manifest) if args.manifest else None
        if manifest and manifest.timeframe.upper() == timeframe:
            model_ohlc = load_bars(path)
            signal = model_signal_frame(
                args.manifest,
                model_ohlc,
                model_ohlc["time"].iloc[0],
                model_ohlc["time"].iloc[-1],
            )
            model_ohlc["signal"] = signal.to_numpy(dtype=np.int8)
            print("  running supplied model ...")
            model_result = run_backtest(
                model_ohlc,
                MlSignalStrategy,
                cost=COST,
                data_name=data_name,
                strategy_params={
                    "signal_col": "signal",
                    "allow_short": manifest.allow_short,
                },
            )
            save_results(model_result, out_dir, seed=SEED)
            all_runs.append(model_result)

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
        print("  zero-cost comparison (SmaCrossAtr) ...")
        zc = CostModel(spread=0.0, commission=0.0, slippage=0.0)
        res_zc = run_backtest(ohlc, SmaCrossAtr, cost=zc, data_name=data_name,
                              strategy_name="SmaCrossAtr_zero_cost")
        save_results(res_zc, out_dir, seed=SEED)
        all_runs.append(res_zc)

    # --- Out-of-sample walk-forward on D1 (purge/embargo) ------------------- #
    print("\n=== walk-forward OOS (D1, purge/embargo) ===")
    ohlc_d1 = load_ohlc(data_files["xauusd_d1"])
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
    data_rows = "\n".join(
        f"| `{path}` | {timeframe} | {bars} | {start} → {end} |"
        for path, timeframe, bars, start, end in dataset_summaries
    )
    notes = (
        "## Data\n\n"
        "| File | TF | Bars | Range |\n|---|---|---|---|\n"
        f"{data_rows}\n\n"
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
    report_path = args.report_path or REPO / "backtest" / "report_backtest.md"
    if not report_path.is_absolute():
        report_path = REPO / report_path
    write_report(
        report_path,
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
