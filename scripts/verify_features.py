"""
scripts/verify_features.py — acceptance gate for the leak-free feature pipeline.

Proves:
  1. Shared causal pipeline builds features end-to-end from data/xauusd_h1.csv
     (+ data/macro_daily.csv) with a train-window-only scaler fit.
  2. CAUSALITY: a spike at bar t+n does NOT change any feature at bar t;
     a spike at bar t DOES change features at t (the pipeline is sensitive to
     current-bar data but never to future data).
  3. feature_contract.json is WRITTEN (feature_names + scaler params), can be
     RELOADED, and a deliberately reordered/renamed feature list RAISES
     RuntimeError (FeatureContractError) at load/transform time.
  4. The leak sources are gone from the fixed modules:
       - multi_timeframe.py: resample label="right", closed="right" + shift(1)
       - macro_features.py: daily shift(1) before ffill
       - timeframe_features.py: handles BOTH volume and tick_volume
       - calendar_features.py: vectorized (searchsorted) - no O(N*E) loop
       - sentiment_analysis.py: no hard-coded 0.0 social sentiment
       - make_features.py: no full-dataset mean/std normalisation
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.config import FeatureConfig
from core.feature_pipeline import (
    FeatureContractError,
    build_feature_frame,
    fit_feature_pipeline,
    load_feature_contract,
    save_feature_contract,
    transform_feature_pipeline,
)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def _load() -> tuple:
    h1 = pd.read_csv(REPO / "data" / "xauusd_h1.csv")
    h1["time"] = pd.to_datetime(h1["time"])
    macro = pd.read_csv(REPO / "data" / "macro_daily.csv")
    macro["time"] = pd.to_datetime(macro["time"])
    macro = macro.set_index("time").sort_index()
    macro_close = macro[[c for c in macro.columns if c.endswith("_close")]]
    return h1, macro_close


def main() -> int:
    print("=" * 66)
    print("verify_features.py - leak-free causal feature pipeline")
    print("=" * 66)

    h1, macro_close = _load()
    cfg = FeatureConfig(window=64, drop_warmup=200)

    # 1. End-to-end train fit + transform on the shared path
    print("\n[1] shared pipeline: train fit + full transform")
    split = pd.Timestamp("2022-01-01")
    train = h1[h1["time"] < split].copy()
    test = h1[h1["time"] >= split].copy()

    Xtr, names, contract = fit_feature_pipeline(train, macro_close, cfg)
    check("train X shape (n>0, f>0)", Xtr.ndim == 2 and Xtr.shape[0] > 0 and Xtr.shape[1] > 0,
          f"got {Xtr.shape}")
    check("feature count >= 20", Xtr.shape[1] >= 20, f"got {Xtr.shape[1]}")
    check("no NaN/Inf in train X", int(np.isnan(Xtr).sum()) == 0 and int(np.isinf(Xtr).sum()) == 0)
    check("train-window z-score (mean~0)", float(np.abs(Xtr.mean(axis=0)).max()) < 0.2)
    check("train-window z-score (std~1)", float(np.abs(Xtr.std(axis=0) - 1).max()) < 0.1)

    full = pd.concat([train, test]).reset_index(drop=True)
    Xall, names2 = transform_feature_pipeline(full, contract, macro_close, cfg)
    check("transform uses same feature names", names == names2)
    check("no NaN/Inf in full transform", int(np.isnan(Xall).sum()) == 0 and int(np.isinf(Xall).sum()) == 0)
    check("full transform length = train+test", Xall.shape[0] == Xtr.shape[0] + len(test))

    # 2. CAUSALITY - spike at t+n does not change feature row at t
    print("\n[2] causality: future spike cannot change the present")
    head = h1.head(5000).copy()
    feats_base = build_feature_frame(head, macro_close, cfg)
    check("base feature frame non-empty", len(feats_base) > 0)
    ts_t = feats_base.index[400]
    ts_n = feats_base.index[405]

    row_t_base = feats_base.loc[ts_t].to_numpy(dtype=float)

    head_future = head.copy()
    mask_n = head_future["time"] == ts_n
    head_future.loc[mask_n, "close"] = head_future.loc[mask_n, "close"] * 1.5
    feats_future = build_feature_frame(head_future, macro_close, cfg)
    row_t_future = feats_future.loc[ts_t].to_numpy(dtype=float)
    check("spike at t+n leaves features at t unchanged",
          np.allclose(row_t_base, row_t_future, atol=1e-12),
          f"max diff {float(np.max(np.abs(row_t_base - row_t_future))):.3e}")

    head_self = head.copy()
    mask_t = head_self["time"] == ts_t
    head_self.loc[mask_t, "close"] = head_self.loc[mask_t, "close"] * 1.5
    feats_self = build_feature_frame(head_self, macro_close, cfg)
    row_t_self = feats_self.loc[ts_t].to_numpy(dtype=float)
    check("spike at t DOES change features at t (sensitive, not frozen)",
          not np.allclose(row_t_base, row_t_self, atol=1e-12))

    # 3. feature_contract.json - write / reload / mismatch raises
    print("\n[3] feature contract: write, reload, mismatch raises")
    with tempfile.TemporaryDirectory() as td:
        cpath = Path(td) / "feature_contract.json"
        save_feature_contract(contract, cpath)
        check("contract file written", cpath.exists())
        check("contract has feature_names", "feature_names" in contract and len(contract["feature_names"]) == len(names))
        check("contract has scaler mean/std",
              "scaler" in contract and len(contract["scaler"]["mean"]) == len(names)
              and len(contract["scaler"]["std"]) == len(names))

        reloaded = load_feature_contract(cpath)
        check("contract reloads", reloaded["feature_names"] == names)
        check("reloaded scaler matches", np.allclose(reloaded["scaler"]["mean"], contract["scaler"]["mean"]))

        bad_contract = dict(contract)
        bad_contract["feature_names"] = list(reversed(names))
        raised = False
        try:
            transform_feature_pipeline(test, bad_contract, macro_close, cfg)
        except RuntimeError as exc:
            raised = True
            msg = str(exc)
            check("mismatch message mentions schema mismatch", "feature schema mismatch" in msg.lower())
            check("mismatch message lists missing", "missing=" in msg)
            check("mismatch message lists extra", "extra=" in msg)
        check("reordered feature list raises RuntimeError", raised)

        raised2 = False
        try:
            transform_feature_pipeline(test, None, macro_close, cfg)
        except RuntimeError:
            raised2 = True
        check("transform without contract raises RuntimeError", raised2)

        raised3 = False
        try:
            load_feature_contract(Path(td) / "does_not_exist.json")
        except RuntimeError:
            raised3 = True
        check("load missing contract raises RuntimeError", raised3)

        check("FeatureContractError subclasses RuntimeError",
              issubclass(FeatureContractError, RuntimeError))

    # 4. leak fixes present in the fixed modules
    print("\n[4] leak sources removed from fixed modules")
    mtf = (REPO / "features" / "multi_timeframe.py").read_text(encoding="utf-8")
    check("multi_timeframe: label='right'", 'label="right"' in mtf or "label='right'" in mtf)
    check("multi_timeframe: closed='right'", 'closed="right"' in mtf or "closed='right'" in mtf)
    check("multi_timeframe: shift(1) after resample", "resampled = resampled.shift" in mtf)

    mf = (REPO / "features" / "macro_features.py").read_text(encoding="utf-8")
    check("macro: daily shift(1) before ffill", "macro_features_daily.shift(1)" in mf)

    tf = (REPO / "features" / "timeframe_features.py").read_text(encoding="utf-8")
    check("timeframe_features: tick_volume support", "tick_volume" in tf)
    check("timeframe_features: volume OR tick_volume guard", "'volume' if 'volume' in df.columns" in tf)

    cf = (REPO / "features" / "calendar_features.py").read_text(encoding="utf-8")
    check("calendar_features: vectorized searchsorted", "searchsorted" in cf)
    check("calendar_features: no per-row python loop", "for i, ts in enumerate(df_timestamps)" not in cf)

    sa = (REPO / "data" / "sentiment_analysis.py").read_text(encoding="utf-8")
    check("sentiment: no hard-coded 0.0 placeholder", "no social posts supplied" in sa)

    mk = (REPO / "features" / "make_features.py").read_text(encoding="utf-8")
    check("make_features: no full-dataset normalisation", "feats.mean(axis=0" not in mk)

    print("\n" + "=" * 66)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
