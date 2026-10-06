"""Subtask 3 verification: policies work on scalar action_dim=75.

- test_ppo_policy_75_discrete: build a real sb3 PPO on a Discrete(75) env, save
  it, load via PpoPolicy, and verify act() returns a valid scalar action < 75
  with finite probs.
- test_ppo_policy_backward_compat_3: existing ppo_gold_v1 artifact still loads
  via PpoPolicy and acts (action_dim=3).
- test_no_hardcoded_action_dim_in_policy_modules: the policy modules must
  derive action_dim from their underlying model (action_space.n / agent
  constructor), never hardcode it in logic. Patterns checked are
  action-dim-SPECIFIC assignments/asserts/one_hots (tensor `ndim == 3`
  checks and default-parameter values are intentionally NOT flagged).
"""

import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.policy import PpoPolicy  # noqa: E402
from models.registry import load_policy  # noqa: E402


class _Discrete75Env(gym.Env):
    """Minimal gym env with Discrete(75) actions — action_dim is the point."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(75)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        obs = np.zeros(8, dtype=np.float32)
        rew = float(1.0 if action == 37 else -0.1)  # arbitrary sparse target
        terminated = self._t >= 16
        truncated = False
        return obs, rew, terminated, truncated, {}


def test_ppo_policy_75_discrete(tmp_path):
    env = make_vec_env(lambda: _Discrete75Env(), n_envs=1)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=2, seed=42)
    model.learn(total_timesteps=128)
    model_path = tmp_path / "ppo75.zip"
    model.save(model_path)

    policy = PpoPolicy(model_path)
    assert policy.action_dim == 75, f"PpoPolicy.action_dim {policy.action_dim} != 75"
    assert policy.obs_dim == 8
    policy.reset()
    obs = np.zeros(8, dtype=np.float32)
    out = policy.act(obs)
    assert 0 <= out.action < 75, f"action {out.action} out of [0,75)"
    assert out.probs.shape == (75,), f"probs shape {out.probs.shape} != (75,)"
    assert np.isfinite(out.probs).all(), "probs contain NaN/inf"
    assert np.isclose(out.probs.sum(), 1.0, atol=1e-6), "probs not normalized"
    assert np.isfinite(out.confidence)
    # argmax of probs matches action
    assert int(np.argmax(out.probs)) == out.action


def test_ppo_policy_backward_compat_3():
    manifest_path = REPO / "artifacts" / "models" / "ppo_gold_v1" / "manifest.json"
    assert manifest_path.exists(), "ppo_gold_v1 manifest missing"
    policy, manifest = load_policy(str(manifest_path), device="cpu")
    assert isinstance(policy, PpoPolicy)
    assert policy.action_dim == 3
    assert manifest.action_dim == 3
    obs = np.zeros(policy.obs_dim, dtype=np.float32)
    out = policy.act(obs)
    assert 0 <= out.action < 3
    assert np.isfinite(out.probs).all()


def test_no_hardcoded_action_dim_in_policy_modules():
    # Action-dim-specific hardcoding patterns that would BREAK a 75-action model.
    # Note: `== 3` / `== 2` alone are NOT flagged because transformer_policy.py
    # legitimately uses `tensor.ndim == 3` for rank checks and `action_dim=3`
    # as an overridable constructor default.
    bad_patterns = (
        "self.action_dim == 3",
        "self.action_dim == 2",
        "action_dim == 3",
        "action_dim == 2",
        "action_dim != 3",
        "action_dim != 2",
        "num_classes=3",
        "num_classes=2",
        "randint(0, 3",
        "randint(0, 2",
        "one_hot(action_idx, 3)",
        "one_hot(action_idx, 2)",
        "np.eye(3)",
        "np.eye(2)",
    )
    for rel in ("models/policy.py", "models/transformer_policy.py", "models/mcts.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        for bad in bad_patterns:
            assert bad not in src, f"{rel} contains hardcoded action-dim pattern {bad!r}"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
