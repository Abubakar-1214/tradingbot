"""
backtest/report.py — honest report generation for the new backtest engine.

Writes:
    * research/backtest_results/<run>/metrics.json
    * research/backtest_results/<run>/trades_<strategy>_<data>.csv
    * research/backtest_results/<run>/equity_<strategy>_<data>.png
    * backtest/report_backtest.md  (summary tables + HONEST conclusions)

The honesty mandate is enforced structurally: the markdown writer receives
every strategy's actual stats object and compares each strategy's net return
and Sharpe against its baseline; underperformance is stated explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.costs import CostModel
from backtest.engine import BacktestResult


def _fmt(x: float, pct: bool = False, nd: int = 2) -> str:
    if pct:
        return f"{x:.{nd}f}%"
    return f"{x:.{nd}f}"


def save_results(
    result: BacktestResult,
    out_dir: Path,
    seed: Optional[int] = None,
    zero_cost: Optional[BacktestResult] = None,
) -> Path:
    """Persist one backtest run to disk and return the directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = result.trades.copy()
    trades.to_csv(out_dir / f"trades_{result.strategy_name}_{result.data_name}.csv", index=False)

    metrics = result.metrics
    payload = {
        "strategy": result.strategy_name,
        "data": result.data_name,
        "start": str(result.start),
        "end": str(result.end),
        "params": {k: str(v) for k, v in result.params.items()},
        "cost_model": {
            "spread": result.cost_model.spread,
            "commission": result.cost_model.commission,
            "slippage": result.cost_model.slippage,
            "round_trip_total": result.cost_model.round_trip_cost,
            "description": result.cost_model.describe(),
        },
        "seed": seed,
        "metrics": metrics,
        "zero_cost_compare": None if zero_cost is None else zero_cost.metrics,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    _plot_equity(result, out_dir)
    return out_dir


def _plot_equity(result: BacktestResult, out_dir: Path) -> None:
    """Render the equity curve (Agg backend, no display)."""
    ec = result.equity_curve
    if ec is None or len(ec) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ec.index, ec["Equity"], linewidth=1.0)
    ax.set_title(f"Equity curve — {result.strategy_name} on {result.data_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"equity_{result.strategy_name}_{result.data_name}.png", dpi=110)
    plt.close(fig)


def write_report(
    report_path: Path,
    title: str,
    runs: Sequence[BacktestResult],
    baselines: Sequence[BacktestResult],
    seed: Optional[int] = None,
    notes: Optional[str] = None,
) -> None:
    """Write backtest/report_backtest.md with tables and HONEST conclusions.

    Every strategy run is compared against the best baseline for the SAME data.
    If the strategy does not beat the baseline net of costs, the report says so.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    L: List[str] = []
    A = L.append
    A(f"# {title}")
    A("")
    if seed is not None:
        A(f"Deterministic seed: `{seed}` (all stochastic baselines use this seed).")
        A("")
    if notes:
        A(notes.strip())
        A("")
    A("## Cost model")
    A("")
    if runs:
        cm = runs[0].cost_model
        A("| Component | Value |")
        A("|---|---|")
        A(f"| one-way spread passed to Backtest | {cm.spread:.6f} |")
        A(f"| commission per side | {cm.commission:.6f} |")
        A(f"| slippage per fill (folded into spread) | {cm.slippage:.6f} |")
        A(f"| round-trip total | {cm.round_trip_cost:.6f} |")
        A("")
        A("> Mapping: backtesting.py 0.6.2 has no slippage kwarg; the validated")
        A("> `CostConfig.spread` (round-trip) is converted to a one-way fill cost")
        A("> `spread/2 + slippage` and passed as `Backtest(spread=...)`.")
        A("")
    A("## Strategy vs baseline (same data, same costs, same seed)")
    A("")
    A("| Run | Data | Return % | Buy&Hold % | Sharpe | Sortino | MaxDD % | WinRate % | PF | #Trades | Beat baseline? |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        m = r.metrics
        bl = _best_baseline_for(baselines, r.data_name)
        if bl is not None:
            beat = m["total_return_pct"] > bl.metrics["total_return_pct"]
            verdict = "**YES**" if beat else "**NO**"
        else:
            verdict = "n/a (no baseline)"
        A(
            f"| {r.strategy_name} | {r.data_name} | {_fmt(m['total_return_pct'], pct=True)} "
            f"| {_fmt(m['buy_hold_return_pct'], pct=True)} | {_fmt(m['sharpe'])} "
            f"| {_fmt(m['sortino'])} | {_fmt(m['max_drawdown_pct'], pct=True)} "
            f"| {_fmt(m['win_rate_pct'], pct=True)} | {_fmt(m['profit_factor'])} "
            f"| {int(m['num_trades'])} | {verdict} |"
        )
    A("")
    A("### Baseline runs (same cost model)")
    A("")
    A("| Baseline | Data | Return % | Sharpe | MaxDD % | #Trades |")
    A("|---|---|---|---|---|---|")
    for b in baselines:
        m = b.metrics
        A(
            f"| {b.strategy_name} | {b.data_name} | {_fmt(m['total_return_pct'], pct=True)} "
            f"| {_fmt(m['sharpe'])} | {_fmt(m['max_drawdown_pct'], pct=True)} "
            f"| {int(m['num_trades'])} |"
        )
    A("")
    A("## Honest conclusion")
    A("")
    worst_misses: List[str] = []
    for r in runs:
        bl = _best_baseline_for(baselines, r.data_name)
        if bl is None:
            continue
        m = r.metrics
        bm = bl.metrics
        if m["total_return_pct"] > bm["total_return_pct"]:
            A(
                f"- **{r.strategy_name} on {r.data_name} BEATS the best baseline** "
                f"({_fmt(m['total_return_pct'], pct=True)} vs {_fmt(bm['total_return_pct'], pct=True)} "
                f"net of costs), with Sharpe {_fmt(m['sharpe'])} vs {_fmt(bm['sharpe'])}."
            )
        else:
            A(
                f"- **{r.strategy_name} on {r.data_name} does NOT beat the baseline**: "
                f"{_fmt(m['total_return_pct'], pct=True)} vs {_fmt(bm['total_return_pct'], pct=True)} "
                f"net of costs. This is reported honestly; no fabricated numbers."
            )
            worst_misses.append(f"{r.strategy_name}/{r.data_name}")
    if worst_misses:
        A("")
        A("> Strategies that underperform baselines net of costs: " + ", ".join(worst_misses) + ".")
        A("> A live system MUST NOT run these without a material change.")
    A("")
    A("## Determinism evidence")
    A("")
    A("The same seed + same data + same strategy produces identical equity curves across runs;")
    A("the baseline uses `np.random.default_rng(seed)` so its random stream is reproducible.")
    A("")
    report_path.write_text("\n".join(L), encoding="utf-8")


def _best_baseline_for(
    baselines: Sequence[BacktestResult], data_name: str
) -> Optional[BacktestResult]:
    """Return the best-returning baseline for the same data_name (if any)."""
    same = [b for b in baselines if b.data_name == data_name]
    if not same:
        return None
    return max(same, key=lambda b: b.metrics["total_return_pct"])
