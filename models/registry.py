from pathlib import Path

from core.model_artifacts import ModelArtifactError, load_manifest
from models.ensemble import EnsemblePolicy
from models.policy import (
    DreamerMCTSPolicy,
    DreamerPolicy,
    PpoPolicy,
    TransformerPolicy,
)


def load_policy(manifest_path, device="cpu"):
    return _load_policy(Path(manifest_path), device, set())


def _member_manifest_path(directory, member):
    path = Path(member)
    if not path.is_absolute():
        path = directory / path
    if path.is_dir() or path.suffix != ".json":
        path = path / "manifest.json"
    return path


def _load_policy(manifest_path, device, seen):
    manifest_path = Path(manifest_path)
    resolved_path = manifest_path.resolve()
    if resolved_path in seen:
        raise ModelArtifactError(f"recursive ensemble member reference: {resolved_path}")
    seen = {*seen, resolved_path}
    manifest = load_manifest(manifest_path)
    model_path = manifest_path.parent / manifest.model_file
    if manifest.model_type == "ppo":
        policy = PpoPolicy(model_path)
    elif manifest.model_type == "dreamer":
        from models.dreamer_agent import DreamerV3Agent

        policy = DreamerPolicy(DreamerV3Agent.from_checkpoint(model_path, device=device))
    elif manifest.model_type == "transformer":
        from models.transformer_policy import TransformerAgentWrapper

        policy = TransformerPolicy(
            TransformerAgentWrapper.from_checkpoint(model_path, map_location=device)
        )
    elif manifest.model_type == "dreamer_mcts":
        from models.dreamer_agent import DreamerV3Agent

        policy = DreamerMCTSPolicy(
            DreamerV3Agent.from_checkpoint(model_path, device=device),
            num_simulations=int(manifest.extra.get("mcts_simulations", 32)),
            c_puct=float(manifest.extra.get("c_puct", 1.0)),
        )
    elif manifest.model_type == "ensemble":
        if not manifest.members:
            raise ModelArtifactError("ensemble manifest must list member manifests")
        policies = []
        hashes = []
        for member in manifest.members:
            member_path = _member_manifest_path(manifest_path.parent, member)
            member_policy, member_manifest = _load_policy(member_path, device, seen)
            policies.append(member_policy)
            hashes.append(member_manifest.contract_hash)
        if any(value != manifest.contract_hash for value in hashes):
            raise ModelArtifactError(
                "ensemble members must have the same contract hash as the ensemble manifest"
            )
        extra = manifest.extra or {}
        try:
            policy = EnsemblePolicy(
                policies,
                min_agreement=extra.get("min_agreement", 0.6),
                vote=extra.get("vote", "soft"),
                weights=extra.get("weights"),
            )
        except (TypeError, ValueError) as exc:
            raise ModelArtifactError(f"invalid ensemble configuration: {exc}") from exc
    else:
        raise ModelArtifactError(
            f"unsupported model type {manifest.model_type!r}; "
            "supported types are ppo, dreamer, transformer, ensemble, and dreamer_mcts"
        )
    if policy.obs_dim != manifest.obs_dim:
        raise ModelArtifactError(
            f"model obs_dim {policy.obs_dim} != manifest obs_dim {manifest.obs_dim}"
        )
    if policy.action_dim != manifest.action_dim:
        raise ModelArtifactError(
            f"model action_dim {policy.action_dim} != manifest action_dim {manifest.action_dim}"
        )
    return policy, manifest
