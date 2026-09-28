from pathlib import Path

from core.model_artifacts import ModelArtifactError, load_manifest
from models.policy import DreamerPolicy, PpoPolicy


def load_policy(manifest_path, device="cpu"):
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    model_path = manifest_path.parent / manifest.model_file
    if manifest.model_type == "ppo":
        policy = PpoPolicy(model_path)
    elif manifest.model_type == "dreamer":
        from models.dreamer_agent import DreamerV3Agent

        policy = DreamerPolicy(DreamerV3Agent.from_checkpoint(model_path, device=device))
    else:
        raise ModelArtifactError(
            f"unsupported model type {manifest.model_type!r}; "
            "transformer, ensemble, and dreamer_mcts are extension points"
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
