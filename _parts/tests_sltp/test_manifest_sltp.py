"""Subtask 4 verification: ModelManifest sl_tp_mode field + validation.

- test_existing_manifests_default_rules: ppo_gold_v1 + dreamer_* artifacts
  (no sl_tp_mode key) load with sl_tp_mode='rules'.
- test_model_mode_manifest_roundtrip: a model-mode manifest (sl_tp_mode=model,
  action_dim=75) writes and round-trips via save_manifest/load_manifest.
- test_model_mode_requires_75: sl_tp_mode=model + action_dim=3 raises
  ModelArtifactError.
- test_invalid_sl_tp_mode_rejected: unknown sl_tp_mode raises.
- test_write_model_artifact_param: train/common.write_model_artifact forwards
  sl_tp_mode to the manifest.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.model_artifacts import (  # noqa: E402
    SLTP_ACTION_DIM,
    ModelArtifactError,
    ModelManifest,
    load_manifest,
    save_manifest,
)


def _existing_manifests():
    candidates = [
        REPO / "artifacts" / "models" / "ppo_gold_v1" / "manifest.json",
        REPO / "artifacts" / "models" / "dreamer_20261003T230304213562Z" / "manifest.json",
        REPO / "artifacts" / "models" / "dreamer_20261003T230535411560Z" / "manifest.json",
    ]
    return [path for path in candidates if path.exists()]


def test_existing_manifests_default_rules():
    manifests = _existing_manifests()
    assert manifests, "no existing artifact manifests found"
    for path in manifests:
        manifest = load_manifest(path)
        assert manifest.sl_tp_mode == "rules", (
            f"{path.name}: expected sl_tp_mode='rules' (dataclass default), "
            f"got {manifest.sl_tp_mode!r}"
        )
        assert manifest.action_dim in (2, 3), (
            f"{path.name}: expected legacy action_dim 2/3, got {manifest.action_dim}"
        )


def _make_artifact_dir(tmp_path, action_dim=75, sl_tp_mode="model"):
    """Create a minimal valid artifact directory (model_file + contract)."""
    d = tmp_path / "artifact"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pt").write_bytes(b"\x00" * 8)
    contract_hash = "deadbeef" * 4
    (d / "feature_contract.json").write_text(
        json.dumps({"hash": contract_hash, "features": ["f1"]}), encoding="utf-8"
    )
    return d, contract_hash


def _make_manifest(directory, contract_hash, action_dim, sl_tp_mode, model_file="model.pt"):
    return ModelManifest(
        model_type="dreamer",
        model_file=model_file,
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


def test_model_mode_manifest_roundtrip(tmp_path):
    d, contract_hash = _make_artifact_dir(tmp_path, action_dim=SLTP_ACTION_DIM, sl_tp_mode="model")
    manifest = _make_manifest(d, contract_hash, SLTP_ACTION_DIM, "model")
    saved = save_manifest(manifest, d)
    loaded = load_manifest(saved)
    assert loaded.sl_tp_mode == "model"
    assert loaded.action_dim == SLTP_ACTION_DIM
    assert loaded == manifest  # full round-trip equality


def test_model_mode_requires_75(tmp_path):
    d, contract_hash = _make_artifact_dir(tmp_path, action_dim=3, sl_tp_mode="model")
    manifest = _make_manifest(d, contract_hash, 3, "model")
    with pytest.raises(ModelArtifactError, match="action_dim=75"):
        save_manifest(manifest, d)


def test_invalid_sl_tp_mode_rejected(tmp_path):
    d, contract_hash = _make_artifact_dir(tmp_path, action_dim=3, sl_tp_mode="auto")
    manifest = _make_manifest(d, contract_hash, 3, "auto")
    with pytest.raises(ModelArtifactError, match="invalid sl_tp_mode"):
        save_manifest(manifest, d)


def test_write_model_artifact_param(tmp_path):
    from train.common import write_model_artifact

    d = tmp_path / "artifact"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pt").write_bytes(b"\x00" * 8)
    contract_hash = "1234abcd" * 4
    contract = {"hash": contract_hash, "features": ["f1"]}
    save_manifest(
        _make_manifest(d, contract_hash, 3, "rules"), d
    ) if False else None  # (placeholder guard; real write below)

    saved = write_model_artifact(
        d,
        model_type="dreamer",
        model_file="model.pt",
        contract=contract,
        window=64,
        n_features=15,
        obs_dim=64 * 15 + 5,
        action_dim=75,
        allow_short=True,
        train_start="2020-01-01",
        train_end="2022-01-01",
        test_end=None,
        hyperparams={"steps": 10},
        sl_tp_mode="model",
    )
    loaded = load_manifest(saved)
    assert loaded.sl_tp_mode == "model"
    assert loaded.action_dim == 75


if __name__ == "__main__":
    import pytest as pt

    sys.exit(pt.main([__file__, "-v"]))
