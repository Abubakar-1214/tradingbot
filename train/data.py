from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
train_dir = str(Path(__file__).resolve().parent)
while train_dir in sys.path:
    sys.path.remove(train_dir)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.config import FeatureConfig
from core.feature_pipeline import (
    fit_feature_pipeline,
    load_macro_daily,
    transform_feature_pipeline,
)
from data.load_data import load_ohlc_csv


@dataclass(frozen=True)
class PreparedData:
    X_train: np.ndarray
    r_train: np.ndarray
    ts_train: np.ndarray
    X_test: np.ndarray
    r_test: np.ndarray
    ts_test: np.ndarray
    feature_names: list[str]
    contract: dict
    window: int


def load_bars(csv_path: str | Path) -> pd.DataFrame:
    bars = load_ohlc_csv(csv_path)
    if "tick_volume" in bars and "volume" not in bars:
        bars = bars.rename(columns={"tick_volume": "volume"})
    return bars


def prepare_data(
    csv_path: str | Path,
    train_end: str,
    window: int = 64,
    macro_csv: str | Path | None = None,
    test_end: str | None = None,
    cfg: FeatureConfig | None = None,
) -> PreparedData:
    bars = load_bars(csv_path)
    bars["time"] = pd.to_datetime(bars["time"])
    bars = bars.sort_values("time").reset_index(drop=True)
    split_at = pd.Timestamp(train_end)
    train_bars = bars.loc[bars["time"] < split_at]
    full_bars = bars if test_end is None else bars.loc[bars["time"] < pd.Timestamp(test_end)]
    if train_bars.empty:
        raise ValueError(f"no training bars precede train_end={train_end}")

    feature_cfg = cfg or FeatureConfig(window=window)
    window = feature_cfg.window
    macro = load_macro_daily(macro_csv) if macro_csv is not None else None
    _, feature_names, contract = fit_feature_pipeline(train_bars, macro, feature_cfg)
    X_all, transformed_names = transform_feature_pipeline(full_bars, contract, macro, feature_cfg)
    if feature_names != transformed_names:
        raise ValueError("feature names changed between fit and transform")

    feature_times = pd.DatetimeIndex(full_bars["time"])
    if feature_cfg.drop_warmup and len(feature_times) > feature_cfg.drop_warmup:
        feature_times = feature_times[feature_cfg.drop_warmup:]
    raw_returns = pd.Series(
        np.log(full_bars["close"].to_numpy(dtype=np.float64)),
        index=pd.DatetimeIndex(full_bars["time"]),
    ).diff()
    returns = raw_returns.reindex(feature_times).to_numpy(dtype=np.float32)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    if len(feature_times) != len(X_all):
        raise ValueError("feature timestamps are not aligned with transformed features")

    train_mask = feature_times < split_at
    test_mask = feature_times >= split_at
    X_train, r_train = X_all[train_mask], returns[train_mask]
    ts_train = feature_times[train_mask].to_numpy()
    X_test, r_test = X_all[test_mask], returns[test_mask]
    ts_test = feature_times[test_mask].to_numpy()
    return PreparedData(
        X_train=X_train,
        r_train=r_train,
        ts_train=ts_train,
        X_test=X_test,
        r_test=r_test,
        ts_test=ts_test,
        feature_names=feature_names,
        contract=contract,
        window=window,
    )
