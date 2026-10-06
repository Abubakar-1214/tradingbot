import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from core.model_artifacts import ModelManifest, save_manifest

ROOT = Path(__file__).resolve().parent.parent


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_model_artifact(
    directory,
    *,
    model_type,
    model_file,
    contract,
    window,
    n_features,
    obs_dim,
    action_dim,
    allow_short,
    train_start,
    train_end,
    test_end,
    hyperparams,
    symbol="XAUUSD",
    timeframe="H1",
    members=None,
    extra=None,
    sl_tp_mode="rules",
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    contract_file = "feature_contract.json"
    (directory / contract_file).write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    manifest = ModelManifest(
        model_type=model_type,
        model_file=model_file,
        contract_file=contract_file,
        contract_hash=contract["hash"],
        window=int(window),
        n_features=int(n_features),
        obs_dim=int(obs_dim),
        action_dim=int(action_dim),
        allow_short=bool(allow_short),
        symbol=symbol,
        timeframe=timeframe,
        train_start=str(train_start),
        train_end=str(train_end),
        test_end=test_end,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit(),
        hyperparams=hyperparams,
        members=list(members or []),
        extra=dict(extra or {}),
        sl_tp_mode=sl_tp_mode,
    )
    return save_manifest(manifest, directory)
