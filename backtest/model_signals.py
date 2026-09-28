from pathlib import Path

import numpy as np
import pandas as pd

from core.config import FeatureConfig
from core.feature_pipeline import load_feature_contract, transform_feature_pipeline
from core.model_artifacts import load_manifest
from env.dreamer_trading_env import RealisticTradingEnv
from models.registry import load_policy


def model_signal_frame(manifest_path, ohlc_df, start, end) -> pd.Series:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    policy, _ = load_policy(manifest_path)
    contract = load_feature_contract(manifest_path.parent / manifest.contract_file)
    bars = ohlc_df.copy()
    if "time" not in bars.columns:
        if not isinstance(bars.index, pd.DatetimeIndex):
            raise ValueError("ohlc_df must have a time column or DatetimeIndex")
        bars = bars.reset_index().rename(columns={bars.index.name or "index": "time"})
    bars["time"] = pd.to_datetime(bars["time"])
    bars = bars.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    cfg = FeatureConfig(window=manifest.window)
    macro_columns = [name for name in bars.columns if name.endswith("_close") and name != "close"]
    macro = (
        bars.set_index("time")[macro_columns]
        if macro_columns
        else None
    )
    if any(name.startswith("macro_") for name in contract["feature_names"]) and macro is None:
        raise ValueError("model contract requires macro close columns in ohlc_df")
    X, _ = transform_feature_pipeline(bars, contract, macro, cfg)
    times = pd.DatetimeIndex(bars["time"])
    if cfg.drop_warmup and len(times) > cfg.drop_warmup:
        times = times[cfg.drop_warmup:]
    if len(times) != len(X):
        raise ValueError("feature timestamps are misaligned")
    returns = pd.Series(
        np.log(bars["close"].to_numpy(dtype=np.float64)),
        index=pd.DatetimeIndex(bars["time"]),
    ).diff().reindex(times).fillna(0.0).to_numpy(dtype=np.float32)
    begin = pd.Timestamp(start)
    finish = pd.Timestamp(end)
    requested = (times >= begin) & (times <= finish)
    if not requested.any():
        return pd.Series(0, index=pd.DatetimeIndex(bars["time"]), dtype=np.int8, name="signal")

    first = int(np.flatnonzero(requested)[0])
    left = max(0, first - manifest.window)
    X_sim = X[left:]
    r_sim = returns[left:]
    ts_sim = times[left:].to_numpy()
    env = RealisticTradingEnv(
        X_sim,
        r_sim,
        timestamps=ts_sim,
        window=manifest.window,
        allow_short=manifest.allow_short,
        leverage=1.0,
        spread=0.0,
        commission=0.0,
        slippage=0.0,
        swap_long=0.0,
        swap_short=0.0,
        stop_loss=None,
        max_drawdown=None,
        max_episode_steps=None,
        random_start=False,
        seed=0,
    )
    policy.reset()
    signals = pd.Series(0, index=pd.DatetimeIndex(bars["time"]), dtype=np.int8, name="signal")
    obs = env._get_obs()
    while True:
        output = policy.act(obs)
        action = int(output.action)
        one_hot = np.zeros(env.action_space, dtype=np.float32)
        one_hot[action] = 1.0
        signal_time = pd.Timestamp(ts_sim[env.t - 1])
        execution_time = pd.Timestamp(ts_sim[env.t])
        if begin <= execution_time <= finish:
            side = 0 if action == 0 else (1 if action == 1 else -1)
            signals.loc[signal_time] = side
        obs, _, done, info = env.step(one_hot)
        executed = int(info["position"])
        policy.observe_executed(0 if executed == 0 else (1 if executed > 0 else 2))
        if done:
            break
    return signals
