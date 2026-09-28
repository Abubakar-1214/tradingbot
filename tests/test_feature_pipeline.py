"""Feature-pipeline pytest: causality (no look-ahead) + feature contract enforcement."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import FeatureConfig
from core.feature_pipeline import (
    FeatureContractError,
    build_feature_frame,
    compute_base_features,
    compute_higher_tf_features,
    fit_feature_pipeline,
    load_feature_contract,
    save_feature_contract,
    transform_feature_pipeline,
    validate_contract_features,
)


def _df(n: int = 300, seed: int = 3, start_price: float = 2000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = start_price + rng.normal(0, 2.0, n).cumsum() * 0.05
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "time": times, "open": closes, "high": closes + 1.5,
        "low": closes - 1.5, "close": closes, "volume": 100.0,
    })


def _macro() -> pd.DataFrame:
    """Processed macro frame exactly like load_macro_daily: datetime index,
    only ``*_close`` columns (the pipeline iterates macro_close.columns, so a
    stray 'time' column would be treated as a macro series and break)."""
    idx = pd.date_range("2023-12-01", periods=400, freq="D")
    df = pd.DataFrame({
        "dxy_close": 100.0 + np.sin(np.arange(len(idx)) / 7.0),
        "vix_close": 15.0 + np.cos(np.arange(len(idx)) / 5.0),
    }, index=idx)
    df.index.name = "time"
    return df


def test_causality_spike_at_t_plus_n_does_not_change_feature_at_t() -> None:
    """P0-4 regression: a spike at bar t+n must leave features at bars < t+n unchanged."""
    df = _df(n=300)
    base = build_feature_frame(df.copy())

    # Inject a huge spike at the LAST bar only.
    spiked = df.copy()
    spiked.loc[spiked.index[-1], "close"] *= 10.0
    spiked.loc[spiked.index[-1], "high"] *= 10.0
    spiked.loc[spiked.index[-1], "low"] *= 10.0
    spiked_frame = build_feature_frame(spiked)

    # Compare all rows EXCEPT the final row: features may only change at the
    # bar where the new price becomes visible.
    assert base.shape == spiked_frame.shape
    base_before = base.iloc[:-1].to_numpy()
    spike_before = spiked_frame.iloc[:-1].to_numpy()
    np.testing.assert_allclose(base_before, spike_before, atol=1e-9,
                               err_msg="feature at t changed when data at t+n spiked")


def test_causality_spike_affects_only_current_and_later_rows() -> None:
    df = _df(n=200)
    base = build_feature_frame(df.copy())
    spiked = df.copy()
    spike_idx = 100
    spiked.loc[spiked.index[spike_idx], "close"] *= 100.0
    spiked.loc[spiked.index[spike_idx], "high"] *= 100.0
    spiked_frame = build_feature_frame(spiked)

    # Bars before the spike must be identical.
    np.testing.assert_allclose(base.iloc[:spike_idx].to_numpy(),
                               spiked_frame.iloc[:spike_idx].to_numpy(),
                               atol=1e-9)
    # At least one feature must react at/after the spike bar.
    assert not np.allclose(base.iloc[spike_idx:].to_numpy(),
                           spiked_frame.iloc[spike_idx:].to_numpy(),
                           atol=1e-9)


def test_higher_tf_alignment_is_shifted() -> None:
    """H4 context must be shift(1)-aligned: bar t uses only H4 candles closed < t."""
    df = _df(n=500)
    # compute_higher_tf_features expects a datetime-indexed frame (the shared
    # path _to_datetime_index does this inside build_feature_frame).
    dfi = df.set_index("time")
    htf = compute_higher_tf_features(dfi)
    assert "h4_ret" in htf.columns
    closes = dfi["close"]
    rs = closes.resample("4h", label="right", closed="right").last().dropna()
    r_ret = rs.pct_change().fillna(0.0)
    s = r_ret.shift(1).reindex(dfi.index, method="ffill").fillna(0.0)
    np.testing.assert_allclose(htf["h4_ret"].to_numpy(), s.to_numpy(), atol=1e-12)


def test_macro_features_shifted_one_day() -> None:
    """Macro daily features are shifted by one day — day's close not known at open."""
    df = _df(n=400)
    macro = _macro()
    feats = build_feature_frame(df, macro, FeatureConfig(macro_shift=1))
    assert "macro_dxy_ret" in feats.columns
    daily = macro["dxy_close"]
    d_ret = daily.pct_change().fillna(0.0)
    expected = d_ret.shift(1).reindex(feats.index, method="ffill").fillna(0.0)
    np.testing.assert_allclose(feats["macro_dxy_ret"].to_numpy(),
                               expected.to_numpy(), atol=1e-12)


def test_contract_roundtrip_and_apply() -> None:
    df = _df(n=300)
    Xtr, names, contract = fit_feature_pipeline(df, cfg=FeatureConfig(drop_warmup=50))
    assert Xtr.shape[1] == len(names)
    assert contract["feature_names"] == names
    assert len(contract["scaler"]["mean"]) == len(names)
    assert len(contract["scaler"]["std"]) == len(names)

    Xall, names2 = transform_feature_pipeline(df, contract,
                                              cfg=FeatureConfig(drop_warmup=50))
    assert names == names2
    # transform with the same data + contract reproduces the training transform
    # EXACTLY (fit stores std already including scaler_eps; transform must not
    # add eps a second time — a real bug fixed in this cycle).
    np.testing.assert_array_equal(Xtr, Xall)


def test_contract_mismatch_raises() -> None:
    df = _df(n=300)
    Xtr, names, contract = fit_feature_pipeline(df, cfg=FeatureConfig(drop_warmup=50))
    del contract["feature_names"]
    with pytest.raises(FeatureContractError):
        transform_feature_pipeline(df, contract, cfg=FeatureConfig(drop_warmup=50))


def test_contract_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FeatureContractError, match="not found"):
        load_feature_contract(tmp_path / "nope.json")


def test_contract_schema_mismatch_raises() -> None:
    df = _df(n=300)
    _, names, contract = fit_feature_pipeline(df, cfg=FeatureConfig(drop_warmup=50))
    wrong = list(names)
    wrong[0] = "SOME_OTHER_FEATURE"
    with pytest.raises(FeatureContractError, match="feature schema mismatch"):
        validate_contract_features(wrong, contract)


def test_contract_save_load_roundtrip(tmp_path) -> None:
    df = _df(n=300)
    _, names, contract = fit_feature_pipeline(df, cfg=FeatureConfig(drop_warmup=50))
    p = save_feature_contract(contract, tmp_path / "feature_contract.json")
    assert p.exists()
    loaded = load_feature_contract(p)
    assert loaded["feature_names"] == names
    assert loaded["hash"] == contract["hash"]


def test_scaler_width_mismatch_raises() -> None:
    df = _df(n=300)
    _, names, contract = fit_feature_pipeline(df, cfg=FeatureConfig(drop_warmup=50))
    contract["scaler"]["mean"] = contract["scaler"]["mean"][:-1]  # one short
    with pytest.raises(FeatureContractError, match="scaler width"):
        validate_contract_features(names, contract)


def test_malformed_contract_raises_contract_error_not_keyerror() -> None:
    """A contract missing required keys must raise FeatureContractError
    (the documented contract error), not a raw KeyError — real bug fixed."""
    df = _df(n=300)
    _, _, contract = fit_feature_pipeline(df, cfg=FeatureConfig(drop_warmup=50))
    bad = {k: v for k, v in contract.items() if k != "pipeline_params"}
    with pytest.raises(FeatureContractError, match="malformed"):
        transform_feature_pipeline(df, bad, cfg=FeatureConfig(drop_warmup=50))
    # Wrong version also raises the documented error.
    bad2 = dict(contract)
    bad2["version"] = 999
    with pytest.raises(FeatureContractError, match="version"):
        transform_feature_pipeline(df, bad2, cfg=FeatureConfig(drop_warmup=50))


def test_base_features_no_nan_or_inf() -> None:
    df = _df(n=300)
    feats = compute_base_features(df)
    assert not np.isnan(feats.to_numpy()).any()
    assert not np.isinf(feats.to_numpy()).any()
    assert {"ret", "vol", "mom", "ma_diff", "rsi", "atr_pct", "volume_ratio"} <= set(feats.columns)
