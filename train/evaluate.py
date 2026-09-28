import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import FeatureConfig
from core.feature_pipeline import (
    load_feature_contract,
    load_macro_daily,
    transform_feature_pipeline,
)
from env.dreamer_trading_env import (
    DEFAULT_ENV_KWARGS,
    EVAL_ENV_OVERRIDES,
    RealisticTradingEnv,
)
from models.registry import load_policy
from train.data import load_bars

DEFAULT_THRESHOLDS = {
    "sharpe": 0.5,
    "max_dd_pct": 20.0,
    "trades": 20,
    "total_return_pct": 0.0,
}


def evaluate_policy(
    policy,
    X_test,
    r_test,
    ts_test,
    env_kwargs=None,
    bars_per_year=252 * 24,
):
    if len(X_test) != len(r_test) or len(X_test) != len(ts_test):
        raise ValueError("test features, returns, and timestamps must have equal lengths")
    if len(X_test) == 0:
        raise ValueError("test period is empty")
    kwargs = {**DEFAULT_ENV_KWARGS, **EVAL_ENV_OVERRIDES, **(env_kwargs or {})}
    kwargs.setdefault("window", 64)
    if env_kwargs is None or "allow_short" not in env_kwargs:
        kwargs["allow_short"] = policy.action_dim == 3
    env = RealisticTradingEnv(X_test, r_test, timestamps=ts_test, **kwargs)
    policy.reset()
    obs = env._get_obs()
    equities = [env.equity]
    positions = []
    while True:
        output = policy.act(obs)
        action = int(output.action)
        if not 0 <= action < env.action_space:
            raise ValueError(f"policy returned invalid action {action}")
        one_hot = np.zeros(env.action_space, dtype=np.float32)
        one_hot[action] = 1.0
        obs, _, done, info = env.step(one_hot)
        policy.observe_executed(action)
        equities.append(float(info["equity"]))
        positions.append(int(info["position"]))
        if done:
            break

    equity = np.asarray(equities, dtype=np.float64)
    log_returns = np.diff(np.log(np.maximum(equity, 1e-12)))
    sharpe = 0.0
    if len(log_returns) > 1 and log_returns.std(ddof=1) > 0:
        sharpe = float(log_returns.mean() / log_returns.std(ddof=1) * math.sqrt(bars_per_year))
    stats = env.episode_stats()
    total_return_pct = (float(equity[-1]) - 1.0) * 100.0
    years = len(log_returns) / bars_per_year if bars_per_year else 0.0
    cagr = (float(equity[-1]) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
    return {
        "total_return_pct": total_return_pct,
        "cagr_pct": float(cagr),
        "sharpe": sharpe,
        "max_dd_pct": stats["max_drawdown_pct"],
        "trades": stats["trades"],
        "win_rate": stats["win_rate"],
        "exposure_pct": float(np.mean(np.asarray(positions) != 0) * 100.0) if positions else 0.0,
        "costs": stats["costs_paid"],
        "buy_hold_return_pct": float(np.expm1(np.asarray(r_test, dtype=np.float64).sum()) * 100.0),
        "final_equity": float(equity[-1]),
        "bars": len(log_returns),
        "bars_per_year": bars_per_year,
    }


def promotion_gate(metrics, thresholds=None):
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    return bool(
        metrics.get("sharpe", float("-inf")) >= limits["sharpe"]
        and metrics.get("max_dd_pct", float("inf")) <= limits["max_dd_pct"]
        and metrics.get("trades", 0) >= limits["trades"]
        and metrics.get("total_return_pct", float("-inf")) > limits["total_return_pct"]
    )


def write_evaluation(
    directory,
    metrics,
    test_start,
    test_end,
    contract_hash,
    thresholds=None,
):
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    payload = {
        "metrics": metrics,
        "thresholds": limits,
        "passed": promotion_gate(metrics, limits),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "test_start": str(test_start),
        "test_end": str(test_end),
        "contract_hash": contract_hash,
    }
    path = Path(directory) / "evaluation.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _evaluation_frame(data_path, contract, window, start, end, macro_csv=None):
    bars = load_bars(data_path)
    macro = load_macro_daily(macro_csv) if macro_csv else None
    cfg = FeatureConfig(window=window)
    X, _ = transform_feature_pipeline(bars, contract, macro, cfg)
    times = pd.DatetimeIndex(bars["time"]).drop_duplicates().sort_values()
    if cfg.drop_warmup and len(times) > cfg.drop_warmup:
        times = times[cfg.drop_warmup:]
    if len(times) != len(X):
        raise ValueError("evaluation feature timestamps are misaligned")
    returns = pd.Series(
        np.log(bars["close"].to_numpy(dtype=np.float64)),
        index=pd.DatetimeIndex(bars["time"]),
    ).diff().reindex(times).fillna(0.0).to_numpy(dtype=np.float32)
    mask = times >= pd.Timestamp(start)
    if end:
        mask &= times <= pd.Timestamp(end)
    return X[mask], returns[mask], times[mask].to_numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a production model artifact")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end")
    parser.add_argument("--macro")
    parser.add_argument("--bars-per-year", type=float, default=252 * 24)
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    policy, manifest = load_policy(manifest_path)
    contract = load_feature_contract(manifest_path.parent / manifest.contract_file)
    X, r, timestamps = _evaluation_frame(
        args.data, contract, manifest.window, args.start, args.end, args.macro
    )
    metrics = evaluate_policy(
        policy,
        X,
        r,
        timestamps,
        {"window": manifest.window, "allow_short": manifest.allow_short},
        args.bars_per_year,
    )
    path = write_evaluation(
        manifest_path.parent,
        metrics,
        timestamps[0],
        timestamps[-1],
        manifest.contract_hash,
    )
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
