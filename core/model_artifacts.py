import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_VERSION = 1


class ModelArtifactError(RuntimeError):
    pass


@dataclass
class ModelManifest:
    model_type: str
    model_file: str
    contract_file: str
    contract_hash: str
    window: int
    n_features: int
    obs_dim: int
    action_dim: int
    allow_short: bool
    symbol: str
    timeframe: str
    train_start: str
    train_end: str
    test_end: str | None
    created_at: str
    git_commit: str | None
    hyperparams: dict
    members: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def artifact_dir(root: Path, model_type: str, run_name: str | None) -> Path:
    name = run_name or f"{model_type}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_and_validate(path: Path) -> ModelManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.pop("manifest_version")
        if version != MANIFEST_VERSION:
            raise ModelArtifactError(
                f"unsupported manifest version {version}; expected {MANIFEST_VERSION}"
            )
        manifest = ModelManifest(**payload)
    except ModelArtifactError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ModelArtifactError(f"invalid model manifest at {path}: {exc}") from exc

    root = path.parent
    if not manifest.model_file or not (root / manifest.model_file).is_file():
        raise ModelArtifactError(f"model file not found: {root / manifest.model_file}")
    contract_path = root / manifest.contract_file
    if not contract_path.is_file():
        raise ModelArtifactError(f"feature contract not found: {contract_path}")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelArtifactError(f"invalid feature contract at {contract_path}: {exc}") from exc
    if contract.get("hash") != manifest.contract_hash:
        raise ModelArtifactError(
            f"feature contract hash mismatch: manifest={manifest.contract_hash!r}, "
            f"contract={contract.get('hash')!r}"
        )
    if manifest.obs_dim != manifest.window * manifest.n_features + 5:
        raise ModelArtifactError(
            f"obs_dim mismatch: {manifest.obs_dim} != "
            f"{manifest.window}*{manifest.n_features}+5"
        )
    return manifest


def save_manifest(manifest: ModelManifest, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    payload = {"manifest_version": MANIFEST_VERSION, **asdict(manifest)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _read_and_validate(path)
    return path


def load_manifest(path: Path) -> ModelManifest:
    path = Path(path)
    if not path.is_file():
        raise ModelArtifactError(f"model manifest not found: {path}")
    return _read_and_validate(path)
