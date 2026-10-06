"""Subtask 5 verification: SLTP_MODE + fraction bounds + startup guard.

- test_sltp_defaults: load_config(_env={}) -> sl_tp_mode='rules', bounds
  0.005/0.05.
- test_sltp_env_parsing: SLTP_MODE=model + bounds override parse into
  TradingBehaviorConfig.
- test_sltp_invalid_mode_raises: SLTP_MODE=bogus -> ValueError.
- test_sltp_invalid_bounds_raise: bad min/max combos -> ValueError
  (validation: 0 < min <= max < 1).
- test_guard_model_manifest_in_rules_mode_raises: SLTP_MODE=rules +
  sl_tp_mode='model' artifact -> ValueError (never silently discard).
- test_guard_rules_manifest_in_model_mode_raises: SLTP_MODE=model +
  sl_tp_mode='rules' artifact -> ValueError.
- test_guard_match_passes: rules/rules and model/model -> no raise.
- test_guard_model_mode_requires_model_signal: SLTP_MODE=model with
  SIGNAL_SOURCE=rule -> ValueError.
- test_guard_no_manifest_ok_in_rules_mode: SLTP_MODE=rules with signal_source
  not 'model' -> no raise (no manifest load required).
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.config import (  # noqa: E402
    TradingBehaviorConfig,
    enforce_sltp_compatibility,
    load_config,
    sltp_mode_failures,
)
from core.model_artifacts import (  # noqa: E402
    SLTP_ACTION_DIM,
    ModelManifest,
    save_manifest,
)


def _write_stub_artifact(tmp_path, sl_tp_mode, action_dim):
    """Create a minimal valid artifact directory (model.pt + contract)."""
    d = tmp_path / "artifact"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pt").write_bytes(b"\x00" * 8)
    contract_hash = "c0ffee00" * 4
    (d / "feature_contract.json").write_text(
        json.dumps({"hash": contract_hash, "features": ["f1"]}), encoding="utf-8"
    )
    manifest = ModelManifest(
        model_type="dreamer",
        model_file="model.pt",
        contract_file="feature_contract.json",
        contract_hash=contract_hash,
        window=64,
        n_features=15,
        obs_dim=64 * 15 + 5,
        action_dim=action_dim,
        allow_short=True,
        symbol="XAUUSD",
        timeframe="H1",
        train_start="2020-01-01",
        train_end="2022-01-01",
        test_end=None,
        created_at="2026-01-01T00:00:00+00:00",
        git_commit=None,
        hyperparams={},
        sl_tp_mode=sl_tp_mode,
    )
    save_manifest(manifest, d)
    return d / "manifest.json"


def test_sltp_defaults():
    cfg = load_config(_env={})
    assert cfg.behavior.sl_tp_mode == "rules"
    assert cfg.behavior.sl_tp_min_frac == 0.005
    assert cfg.behavior.sl_tp_max_frac == 0.05
    assert cfg.behavior.tp_min_frac == 0.005
    assert cfg.behavior.tp_max_frac == 0.05


def test_sltp_env_parsing():
    cfg = load_config(_env={
        "SLTP_MODE": "model",
        "SL_TP_MIN_FRAC": "0.01",
        "SL_TP_MAX_FRAC": "0.03",
        "TP_MIN_FRAC": "0.005",
        "TP_MAX_FRAC": "0.02",
    })
    assert cfg.behavior.sl_tp_mode == "model"
    assert cfg.behavior.sl_tp_min_frac == 0.01
    assert cfg.behavior.sl_tp_max_frac == 0.03
    assert cfg.behavior.tp_min_frac == 0.005
    assert cfg.behavior.tp_max_frac == 0.02


def test_sltp_invalid_mode_raises():
    with pytest.raises(ValueError, match="SLTP_MODE must be 'rules' or 'model'"):
        load_config(_env={"SLTP_MODE": "auto"})


@pytest.mark.parametrize(
    "env,msg",
    [
        # min <= 0
        ({"SL_TP_MIN_FRAC": "0"}, "SL_TP_MIN_FRAC"),
        ({"SL_TP_MIN_FRAC": "-0.01"}, "SL_TP_MIN_FRAC"),
        # max <= 0
        ({"SL_TP_MAX_FRAC": "0"}, "SL_TP_MIN_FRAC"),
        ({"SL_TP_MAX_FRAC": "-0.01"}, "SL_TP_MIN_FRAC"),
        # min > max
        ({"SL_TP_MIN_FRAC": "0.05", "SL_TP_MAX_FRAC": "0.01"}, "SL_TP_MIN_FRAC"),
        ({"TP_MIN_FRAC": "0"}, "TP_MIN_FRAC"),
        ({"TP_MIN_FRAC": "-0.01"}, "TP_MIN_FRAC"),
        ({"TP_MAX_FRAC": "0"}, "TP_MIN_FRAC"),
        ({"TP_MAX_FRAC": "-0.01"}, "TP_MIN_FRAC"),
        ({"TP_MIN_FRAC": "0.05", "TP_MAX_FRAC": "0.01"}, "TP_MIN_FRAC"),
    ],
)
def test_sltp_invalid_bounds_raise(env, msg):
    with pytest.raises(ValueError, match=msg):
        load_config(_env=env)


def test_sltp_wide_max_is_valid():
    # 0 < min <= max < 1 is the validation contract; a user may widen the max
    # beyond the training-bucket extreme (0.05) for live use.
    cfg = load_config(_env={"SL_TP_MAX_FRAC": "0.15", "TP_MAX_FRAC": "0.15"})
    assert cfg.behavior.sl_tp_max_frac == 0.15
    assert cfg.behavior.tp_max_frac == 0.15


def test_guard_model_manifest_in_rules_mode_raises(tmp_path):
    manifest_path = _write_stub_artifact(tmp_path, sl_tp_mode="model", action_dim=SLTP_ACTION_DIM)
    cfg = load_config(_env={
        "SLTP_MODE": "rules",
        "SIGNAL_SOURCE": "model",
        "MODEL_MANIFEST": str(manifest_path),
        "REQUIRE_PROMOTED_MODEL": "false",
    })
    with pytest.raises(ValueError, match="SILENTLY DISCARD"):
        enforce_sltp_compatibility(cfg)


def test_guard_rules_manifest_in_model_mode_raises(tmp_path):
    manifest_path = _write_stub_artifact(tmp_path, sl_tp_mode="rules", action_dim=3)
    cfg = load_config(_env={
        "SLTP_MODE": "model",
        "SIGNAL_SOURCE": "model",
        "MODEL_MANIFEST": str(manifest_path),
        "REQUIRE_PROMOTED_MODEL": "false",
    })
    with pytest.raises(ValueError, match="cannot supply.*SL/TP"):
        enforce_sltp_compatibility(cfg)


def test_guard_match_passes(tmp_path):
    rules_path = _write_stub_artifact(tmp_path / "r1", sl_tp_mode="rules", action_dim=3)
    cfg_rules = load_config(_env={
        "SLTP_MODE": "rules",
        "SIGNAL_SOURCE": "model",
        "MODEL_MANIFEST": str(rules_path),
        "REQUIRE_PROMOTED_MODEL": "false",
    })
    enforce_sltp_compatibility(cfg_rules)  # no raise

    model_path = _write_stub_artifact(tmp_path / "m1", sl_tp_mode="model", action_dim=SLTP_ACTION_DIM)
    cfg_model = load_config(_env={
        "SLTP_MODE": "model",
        "SIGNAL_SOURCE": "model",
        "MODEL_MANIFEST": str(model_path),
        "REQUIRE_PROMOTED_MODEL": "false",
    })
    enforce_sltp_compatibility(cfg_model)  # no raise


def test_guard_model_mode_requires_model_signal():
    cfg = load_config(_env={"SLTP_MODE": "model"})  # SIGNAL_SOURCE stays 'rule'
    with pytest.raises(ValueError, match="requires SIGNAL_SOURCE=model"):
        enforce_sltp_compatibility(cfg)


def test_guard_no_manifest_ok_in_rules_mode():
    cfg = load_config(_env={})  # signal_source='rule', sl_tp_mode='rules'
    enforce_sltp_compatibility(cfg)  # no raise, no manifest load


def test_sltp_failures_pure_function():
    # Pure function works without loading any manifest.
    cfg = load_config(_env={"SLTP_MODE": "rules"})
    assert sltp_mode_failures(cfg, manifest=None) == []


def test_behavior_config_construct_defaults():
    b = TradingBehaviorConfig()
    assert b.sl_tp_mode == "rules"
    assert b.sl_tp_min_frac == 0.005
    assert b.sl_tp_max_frac == 0.05
    assert b.tp_min_frac == 0.005
    assert b.tp_max_frac == 0.05


if __name__ == "__main__":
    import pytest as pt

    sys.exit(pt.main([__file__, "-v"]))
