"""
features/make_features.py — LEGACY-compatible feature builder (leak-free).

P0-4 FIX: the previous implementation normalised over the FULL dataset
(train+test mean/std => test statistics leaked into training).  This module is
now a thin compatibility wrapper over the SINGLE shared causal pipeline in
``core/feature_pipeline.py``:

  * every indicator is causal (rolling / lag only),
  * normalisation is delegated to the contract-based fit/transform API which
    fits the scaler on the TRAIN window only,
  * ``compute_features`` returns RAW (unnormalised) features so downstream
    callers can never silently re-introduce the full-dataset leak.

New callers should use ``core.feature_pipeline`` directly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.config import FeatureConfig
from core.feature_pipeline import (
    build_feature_frame,
    fit_feature_pipeline,
    transform_feature_pipeline,
)
from data.load_data import load_ohlc_csv


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI kept for backwards compatibility (causal)."""
    from core.feature_pipeline import compute_rsi as _rsi

    return _rsi(series, period)


def compute_features(
    df: pd.DataFrame,
    macro_close: Optional[pd.DataFrame] = None,
    cfg: Optional[FeatureConfig] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compute RAW (unnormalised) causal features and returns.

    IMPORTANT: no normalisation happens here — normalisation must go through
    the contract-based ``fit_feature_pipeline`` / ``transform_feature_pipeline``
    so scaler statistics never leak across train/test.

    Returns:
        (feature_frame, feats float32 [n, n_features], rets float32 [n])
    """
    cfg = cfg or FeatureConfig()
    if "time" in df.columns:
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()

    frame = build_feature_frame(df, macro_close, cfg)
    feats = np.nan_to_num(
        frame.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    rets = frame["ret"].to_numpy(dtype=np.float32) if "ret" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    return frame, feats, rets


def fit_features(
    df: pd.DataFrame,
    macro_close: Optional[pd.DataFrame] = None,
    cfg: Optional[FeatureConfig] = None,
):
    """Fit the train-window scaler and return (X, names, contract)."""
    return fit_feature_pipeline(df, macro_close, cfg)


def transform_features(
    df: pd.DataFrame,
    contract: dict,
    macro_close: Optional[pd.DataFrame] = None,
    cfg: Optional[FeatureConfig] = None,
):
    """Apply a previously fitted contract (raises on schema mismatch)."""
    return transform_feature_pipeline(df, contract, macro_close, cfg)


def make_features(csv_path: str, window: int = 64):
    """
    Load OHLCV from a CSV and return raw causal features (no normalisation).
    """
    cfg = FeatureConfig(window=window)
    df = load_ohlc_csv(csv_path)
    return compute_features(df, cfg=cfg)
