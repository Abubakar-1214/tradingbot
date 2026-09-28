from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from core.feature_pipeline import load_macro_daily, transform_feature_pipeline
from core.model_artifacts import load_manifest
from core.observation import AccountState, build_observation
from live.decision_engine import Decision
from models.registry import load_policy

logger = logging.getLogger(__name__)


class ModelSignalSource:
    def __init__(self, cfg, burn_in: int | None = None) -> None:
        self.cfg = cfg
        self.policy, policy_manifest = load_policy(cfg.model.manifest_path)
        self.manifest = load_manifest(cfg.model.manifest_path)
        if self.manifest.contract_hash != policy_manifest.contract_hash:
            raise ValueError("model policy and manifest contract hashes do not match")
        if self.manifest.action_dim not in (2, 3):
            raise ValueError(
                f"live model action_dim must be 2 or 3, got {self.manifest.action_dim}"
            )
        if self.manifest.action_dim == 2 and cfg.broker.allow_short:
            raise ValueError(
                "ALLOW_SHORT=true requires a three-action model; this artifact has action_dim=2"
            )
        contract_path = Path(cfg.model.manifest_path).parent / self.manifest.contract_file
        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.feature_names = list(self.contract["feature_names"])
        self.macro = None
        if any(name.startswith("macro_") for name in self.feature_names):
            if cfg.model.macro_csv is None:
                raise ValueError(
                    "model contract requires macro features but MACRO_CSV is not configured"
                )
            self.macro = load_macro_daily(cfg.model.macro_csv)
            if self.macro.empty:
                raise ValueError(
                    f"model contract requires macro features, but MACRO_CSV has no usable data: "
                    f"{cfg.model.macro_csv}"
                )
        params = self.contract.get("pipeline_params", {})
        self.feature_cfg = replace(
            cfg.feature,
            window=self.manifest.window,
            higher_timeframes=tuple(
                params.get("higher_timeframes", cfg.feature.higher_timeframes)
            ),
            macro_shift=int(params.get("macro_shift", cfg.feature.macro_shift)),
            drop_warmup=int(params.get("drop_warmup", cfg.feature.drop_warmup)),
            normalize=bool(params.get("normalize", cfg.feature.normalize)),
        )
        self.burn_in = self.manifest.window if burn_in is None else max(0, int(burn_in))
        self._needs_burn_in = True
        self._warned_short = False
        if self.manifest.action_dim == 3 and not cfg.broker.allow_short:
            logger.warning("Model supports short actions; ALLOW_SHORT=false maps short to flat")
            self._warned_short = True

    def decide(self, df_closed_bars: pd.DataFrame, account: AccountState) -> Decision:
        if df_closed_bars is None or df_closed_bars.empty:
            return Decision(0, 0.0, 0.0, "INSUFFICIENT_HISTORY")
        window = self.manifest.window
        if len(df_closed_bars) < self.feature_cfg.drop_warmup + window:
            return Decision(0, 0.0, 0.0, "INSUFFICIENT_HISTORY")
        feature_bars = df_closed_bars.copy()
        feature_bars["time"] = (
            pd.to_datetime(feature_bars["time"], utc=True).dt.tz_localize(None)
        )
        features, _ = transform_feature_pipeline(
            feature_bars, self.contract, self.macro, self.feature_cfg
        )
        usable = features
        if len(usable) < window:
            return Decision(0, 0.0, 0.0, "INSUFFICIENT_HISTORY")
        if self._needs_burn_in:
            self.policy.reset()
            self._burn_in(usable, window)
            self._needs_burn_in = False
        observation = build_observation(usable[-window:], account)
        if observation.size != self.manifest.obs_dim:
            raise ValueError(
                f"observation width {observation.size} != manifest obs_dim "
                f"{self.manifest.obs_dim}"
            )
        output = self.policy.act(observation)
        action = int(output.action)
        info = dict(output.info)
        if action == 2 and self.manifest.action_dim == 2:
            raise ValueError("two-action model produced unsupported short action 2")
        if action < 0 or action >= self.manifest.action_dim:
            raise ValueError(
                f"model action {action} is outside action_dim "
                f"{self.manifest.action_dim}"
            )
        if (
            action == 2
            and self.manifest.action_dim == 3
            and not self.cfg.broker.allow_short
        ):
            action = 0
            info["short_mapped_to_flat"] = True
        return Decision(
            action,
            float(output.confidence),
            1.0,
            "MODEL_SIGNAL",
            info,
        )

    def _burn_in(self, features: np.ndarray, window: int) -> None:
        flat = AccountState(0, 0.0, 0, 0.0, 1.0)
        first_end = max(window - 1, len(features) - self.burn_in - 1)
        for end in range(first_end, len(features) - 1):
            observation = build_observation(
                features[end - window + 1 : end + 1], flat
            )
            self.policy.act(observation)
            self.policy.observe_executed(0)

    def executed(self, action: int) -> None:
        self.policy.observe_executed(int(action))
