"""Temp sanity check: real dreamer artifact feature pipeline on a synthetic frame."""
import json
import sys
from pathlib import Path

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from core.feature_pipeline import build_feature_frame, transform_feature_pipeline
from core.config import FeatureConfig
from tests.helpers import sample_df
from dataclasses import replace

manifest_path = (
    REPO / "artifacts" / "models" / "dreamer_20261003T230304213562Z" / "manifest.json"
)
contract_path = manifest_path.parent / "feature_contract.json"
contract = json.loads(contract_path.read_text(encoding="utf-8"))
params = contract.get("pipeline_params", {})
print("pipeline_params:", params)

cfg = replace(
    FeatureConfig(),
    window=int(params.get("window", 64)),
    higher_timeframes=tuple(params.get("higher_timeframes", ())),
    macro_shift=int(params.get("macro_shift", 1)),
    drop_warmup=int(params.get("drop_warmup", 200)),
    normalize=bool(params.get("normalize", True)),
)
print("feature_cfg:", cfg)

df = sample_df(n=320)
print("frame:", df.shape, df["time"].dtype)

feats = build_feature_frame(df, None, cfg)
print("built features:", feats.shape, list(feats.columns)[:5], "...")
print("NaN count:", int(np.isnan(feats.to_numpy()).sum()))

X, names = transform_feature_pipeline(df, contract, None, cfg)
print("transformed X:", X.shape, "names match contract:", names == list(contract["feature_names"]))
print("nan/inf in X:", int(np.isnan(X).sum()), int(np.isinf(X).sum()))
print("usable rows (after drop_warmup):", X.shape[0], "window:", cfg.window)
print("OK" if X.shape[0] >= cfg.window else "TOO_FEW_ROWS")
