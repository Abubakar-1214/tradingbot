"""
core/feature_pipeline.py — leak-free causal feature pipeline (SINGLE shared path).

This module is the ONE code path used by training, backtesting and the live
loop.  It fixes the three P0-4 look-ahead leaks in the legacy ``features/``
modules:

1. Higher-timeframe resample leaks: bars were labelled at their START time and
   forward-filled onto the base timeframe (a 10:00 M5 bar saw the H1 candle
   close that happens at 11:00).  Here we resample with
   ``label='right', closed='right'`` (candle labelled at its END) and then
   ``shift(1)`` so a base bar only ever sees higher-TF candles that are already
   CLOSED.  (Verified against pandas 3.0.5.)

2. Macro daily-close leak: daily closes stamped at 00:00 were forward-filled
   through the whole day (the day's close was "known" at day start).  Here
   every daily-derived feature is ``shift(1)`` before forward-fill.

3. Full-dataset normalisation leak: legacy code computed mean/std over the
   whole dataset (test stats leaked into training).  Here the z-score scaler is
   fit ONLY on the train window and its parameters are persisted in a
   ``feature_contract.json`` that the model is trained against and that live /
   backtest MUST load and match.  A schema mismatch raises
   :class:`FeatureContractError` (a RuntimeError) at load time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from core.config import FeatureConfig

EPS = 1e-8
CONTRACT_VERSION = 1

# Higher-timeframe context sources (resampled from the base bars).
HIGHER_TF_RULES: Dict[str, str] = {"H4": "4h", "D1": "1D"}
# Rolling windows for base features (bar units of the base timeframe).
_BASE_WINDOWS = {"vol": 24, "mom": 24, "ma_fast": 24, "ma_slow": 120, "rsi": 14,
                 "bb": 20, "atr": 14, "vma": 20}
_MACRO_CORR_WINDOW = 120  # in DAILY bars
_MACRO_MOM_WINDOW = 5     # in DAILY bars


class FeatureContractError(RuntimeError):
    """Raised when a feature contract is missing, malformed or mismatched."""


# --------------------------------------------------------------------------- #
# Small indicator helpers (causal — rolling/lag only)
# --------------------------------------------------------------------------- #
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + EPS)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean().fillna(0.0)


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    else:
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _causal_align(series: pd.Series, target_index: pd.DatetimeIndex, shift: int = 1) -> pd.Series:
    """
    Make a higher-frequency series available to a base index with NO future leak.

    ``series`` is indexed at the END (right) of its candles.  ``shift(1)`` moves
    every value one candle later so it is only usable once that candle has
    closed, then forward-fill onto the base index.
    """
    return series.shift(shift).reindex(target_index, method="ffill").fillna(0.0)


# --------------------------------------------------------------------------- #
# Feature builders (all causal)
# --------------------------------------------------------------------------- #
def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal rolling/lag features on the base timeframe."""
    close = df["close"]
    ret = np.log(close).diff().fillna(0.0)
    out = pd.DataFrame(index=df.index)
    out["ret"] = ret
    out["vol"] = ret.rolling(_BASE_WINDOWS["vol"]).std().fillna(0.0)
    out["mom"] = close.pct_change(_BASE_WINDOWS["mom"]).fillna(0.0)

    ma_fast = close.rolling(_BASE_WINDOWS["ma_fast"]).mean()
    ma_slow = close.rolling(_BASE_WINDOWS["ma_slow"]).mean()
    out["ma_diff"] = ((ma_fast - ma_slow) / close).fillna(0.0)

    out["rsi"] = (compute_rsi(close, _BASE_WINDOWS["rsi"]) / 100.0).fillna(0.5)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_diff"] = ((macd - signal) / close).fillna(0.0)

    bb_mid = close.rolling(_BASE_WINDOWS["bb"]).mean()
    bb_std = close.rolling(_BASE_WINDOWS["bb"]).std()
    bb_up, bb_lo = bb_mid + 2 * bb_std, bb_mid - 2 * bb_std
    out["bb_position"] = ((close - bb_lo) / (bb_up - bb_lo).replace(0.0, np.nan)).clip(0, 1).fillna(0.5)

    atr = compute_atr(df, _BASE_WINDOWS["atr"])
    out["atr_pct"] = (atr / close).fillna(0.0)

    vol_col = "volume" if "volume" in df.columns else ("tick_volume" if "tick_volume" in df.columns else None)
    if vol_col is not None:
        v = df[vol_col].astype(float)
        vma = v.rolling(_BASE_WINDOWS["vma"]).mean().replace(0.0, np.nan)
        out["volume_ratio"] = (v / vma).fillna(1.0)
    else:
        out["volume_ratio"] = 1.0
    return out


def compute_higher_tf_features(df: pd.DataFrame, rules: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Causal higher-timeframe context features resampled from the base bars.

    For each higher TF (e.g. 4h, 1D): resample with ``label='right',
    closed='right'`` so the candle is labelled at its END, compute features on
    that series, then ``shift(1)`` + forward-fill — a base bar at time t only
    sees higher-TF candles that closed strictly before t.
    """
    rules = rules or HIGHER_TF_RULES
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    for tf, rule in rules.items():
        rs = close.resample(rule, label="right", closed="right").last().dropna()
        r_ret = rs.pct_change().fillna(0.0)
        r_mom = rs.pct_change(_MACRO_MOM_WINDOW).fillna(0.0)
        fast = rs.rolling(_BASE_WINDOWS["ma_fast"]).mean()
        slow = rs.rolling(_BASE_WINDOWS["ma_slow"]).mean()
        r_ma = ((fast - slow) / rs).fillna(0.0)
        for suffix, s in (("ret", r_ret), ("mom", r_mom), ("ma_diff", r_ma)):
            out[f"{tf.lower()}_{suffix}"] = _causal_align(s, df.index, shift=1)
    return out


def load_macro_daily(path: Union[str, Path] = "data/macro_daily.csv") -> pd.DataFrame:
    """Load the daily macro CSV (time + ``*_close`` columns) indexed by day."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    cols = [c for c in df.columns if c.endswith("_close")]
    return df[cols]


def compute_macro_features_causal(df: pd.DataFrame, macro_close: pd.DataFrame,
                                  shift: int = 1) -> pd.DataFrame:
    """
    Causal daily macro features (fixes the 00:00-stamp + ffill leak).

    Gold is resampled to daily (right-closed), every macro feature is computed
    on the DAILY series and then shifted by one day + forward-filled onto the
    base index — the base bar at time t only sees macro values whose daily
    candle has already closed.
    """
    out = pd.DataFrame(index=df.index)
    if macro_close is None or macro_close.empty:
        return out
    gold_daily = df["close"].resample("1D", label="right", closed="right").last().dropna()
    gold_daily_ret = gold_daily.pct_change()
    for col in macro_close.columns:
        s = macro_close[col].dropna().sort_index()
        if len(s) < _MACRO_MOM_WINDOW + 2:
            continue
        s_ret = s.pct_change().fillna(0.0)
        s_mom = s.pct_change(_MACRO_MOM_WINDOW).fillna(0.0)
        joined = pd.concat([gold_daily_ret, s_ret], axis=1, sort=True).dropna()
        corr = joined.iloc[:, 0].rolling(_MACRO_CORR_WINDOW).corr(joined.iloc[:, 1]).fillna(0.0)
        name = col.replace("_close", "").lower()
        out[f"macro_{name}_ret"] = _causal_align(s_ret, df.index, shift=shift)
        out[f"macro_{name}_mom"] = _causal_align(s_mom, df.index, shift=shift)
        out[f"macro_{name}_corr"] = _causal_align(corr, df.index, shift=shift)
    return out


def build_feature_frame(df: pd.DataFrame,
                        macro_close: Optional[pd.DataFrame] = None,
                        cfg: Optional[FeatureConfig] = None) -> pd.DataFrame:
    """Assemble the full causal feature frame indexed by base-bar time."""
    cfg = cfg or FeatureConfig()
    df = _to_datetime_index(df)
    parts = [compute_base_features(df), compute_higher_tf_features(df)]
    if macro_close is not None and not macro_close.empty:
        macro_part = compute_macro_features_causal(df, macro_close, shift=cfg.macro_shift)
        if not macro_part.empty:
            parts.append(macro_part)
    feats = pd.concat(parts, axis=1)
    if cfg.drop_warmup and len(feats) > cfg.drop_warmup:
        feats = feats.iloc[cfg.drop_warmup:]
    return feats


# --------------------------------------------------------------------------- #
# Contract save / load / validate
# --------------------------------------------------------------------------- #
def _contract_hash(feature_names: Sequence[str], params: dict) -> str:
    payload = json.dumps(
        {"names": list(feature_names), "params": params}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_contract(feature_names: Sequence[str], mean: np.ndarray, std: np.ndarray,
                   cfg: FeatureConfig) -> dict:
    params = {
        "window": cfg.window,
        "higher_timeframes": list(cfg.higher_timeframes),
        "macro_shift": cfg.macro_shift,
        "drop_warmup": cfg.drop_warmup,
        "normalize": cfg.normalize,
    }
    return {
        "version": CONTRACT_VERSION,
        "feature_names": list(feature_names),
        "scaler": {
            "mean": [float(x) for x in np.asarray(mean, dtype=np.float64).ravel()],
            "std": [float(x) for x in np.asarray(std, dtype=np.float64).ravel()],
        },
        "pipeline_params": params,
        "hash": _contract_hash(feature_names, params),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_feature_contract(contract: dict, path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)
    return p


def _validate_contract_shape(contract: dict) -> None:
    """Raise FeatureContractError when a contract dict is malformed/unsupported."""
    required = {"version", "feature_names", "scaler", "pipeline_params"}
    missing = required - set(contract.keys())
    if missing:
        raise FeatureContractError(
            f"feature contract is malformed; missing keys: {sorted(missing)}"
        )
    if contract.get("version") != CONTRACT_VERSION:
        raise FeatureContractError(
            f"feature contract version {contract.get('version')} != expected {CONTRACT_VERSION}"
        )


def load_feature_contract(path: Union[str, Path]) -> dict:
    p = Path(path)
    if not p.exists():
        raise FeatureContractError(
            f"feature contract not found at {p}. Train the model first "
            "(fit_feature_pipeline) so the contract is saved next to the weights."
        )
    with open(p, "r", encoding="utf-8") as f:
        contract = json.load(f)
    _validate_contract_shape(contract)
    return contract


def validate_contract_features(feature_names: Sequence[str], contract: dict) -> None:
    """Raise FeatureContractError when the computed feature schema differs."""
    expected = list(contract["feature_names"])
    actual = list(feature_names)
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise FeatureContractError(
            "feature schema mismatch: contract has %d features, pipeline produced %d "
            "(missing=%s extra=%s). The model / live bot MUST be retrained or the "
            "pipeline restored to the contract." % (len(expected), len(actual), missing, extra)
        )
    n_scaler = len(contract["scaler"]["mean"])
    if n_scaler != len(expected):
        raise FeatureContractError(
            f"scaler width {n_scaler} != feature count {len(expected)} in contract"
        )


# --------------------------------------------------------------------------- #
# Public API — the single shared code path
# --------------------------------------------------------------------------- #
def fit_feature_pipeline(df: pd.DataFrame,
                         macro_close: Optional[pd.DataFrame] = None,
                         cfg: Optional[FeatureConfig] = None) -> Tuple[np.ndarray, List[str], dict]:
    """
    Compute causal features and FIT the z-score scaler on the given data.

    IMPORTANT: pass ONLY the train window here.  The returned contract carries
    the train-window scaler parameters; every later transform (test / backtest /
    live) must use :func:`transform_feature_pipeline` with that contract so no
    test statistics ever leak into training.

    Returns:
        (X float32 [n_bars, n_features], feature_names, contract dict)
    """
    cfg = cfg or FeatureConfig()
    feats = build_feature_frame(df, macro_close, cfg)
    feature_names = list(feats.columns)
    X = np.nan_to_num(feats.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if cfg.normalize:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + cfg.scaler_eps
        X = ((X - mean) / std).astype(np.float32)
    else:
        mean = np.zeros((1, X.shape[1]))
        std = np.ones((1, X.shape[1]))

    contract = build_contract(feature_names, mean, std, cfg)
    return X, feature_names, contract


def transform_feature_pipeline(df: pd.DataFrame,
                               contract: dict,
                               macro_close: Optional[pd.DataFrame] = None,
                               cfg: Optional[FeatureConfig] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Compute causal features and APPLY a previously fitted contract's scaler.

    Raises FeatureContractError if the contract is missing/mismatched — the
    live bot and backtests refuse to run against a wrong feature schema.

    Returns:
        (X float32 [n_bars, n_features], feature_names)
    """
    if contract is None:
        raise FeatureContractError(
            "transform_feature_pipeline requires a feature contract "
            "(fit_feature_pipeline first, then save/load it)."
        )
    _validate_contract_shape(contract)
    cfg = cfg or FeatureConfig()
    feats = build_feature_frame(df, macro_close, cfg)
    feature_names = list(feats.columns)
    validate_contract_features(feature_names, contract)

    X = np.nan_to_num(feats.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.asarray(contract["scaler"]["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(contract["scaler"]["std"], dtype=np.float32).reshape(1, -1)
    X = ((X - mean) / std).astype(np.float32)
    return X, feature_names


def compute_features(df: pd.DataFrame,
                     macro_close: Optional[pd.DataFrame] = None,
                     cfg: Optional[FeatureConfig] = None,
                     mode: str = "train",
                     contract: Optional[dict] = None) -> Tuple[np.ndarray, List[str], Optional[dict]]:
    """
    Convenience wrapper around fit/transform.

    mode='train' -> fit scaler, return (X, names, contract)
    mode='infer' -> apply contract scaler, return (X, names, None)
    """
    if mode == "train":
        X, names, c = fit_feature_pipeline(df, macro_close, cfg)
        return X, names, c
    X, names = transform_feature_pipeline(df, contract, macro_close, cfg)
    return X, names, None


# --------------------------------------------------------------------------- #
# CLI smoke check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover - manual verification
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / "data"
    h1 = pd.read_csv(data_dir / "xauusd_h1.csv")
    macro = load_macro_daily(data_dir / "macro_daily.csv")

    split_ts = pd.Timestamp("2022-01-01")
    h1["time"] = pd.to_datetime(h1["time"])
    train = h1[h1["time"] < split_ts]
    test = h1[h1["time"] >= split_ts]

    Xtr, names, contract = fit_feature_pipeline(train, macro)
    print("TRAIN:", Xtr.shape, "features:", len(names))
    print("  first 6:", names[:6])

    full = pd.concat([train, test]).reset_index(drop=True)
    Xall, names2 = transform_feature_pipeline(full, contract, macro)
    n_train = Xtr.shape[0]
    Xtest = Xall[n_train:]
    print("TEST :", Xtest.shape, "names match:", names == names2)
    print("NaN/Inf in test:", int(np.isnan(Xtest).sum()), int(np.isinf(Xtest).sum()))
    print("OK")
