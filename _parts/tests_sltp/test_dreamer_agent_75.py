"""Subtask 2 verification: DreamerV3 agent with action_dim=75 (no code change expected).

- test_train_100_steps_75_actions: train 100 steps on a tiny random slice with
  action_dim=75; losses finite (no NaN/inf); sampled action idx decode to valid
  (dir, sl_frac, tp_frac) via SL_BUCKETS/TP_BUCKETS.
- test_act_returns_scalar_75_onehot: act() returns a (75,) one-hot; RSSM flow
  works for 75-wide prev_action.
- test_replay_buffer_75_onehot: ReplayBuffer add/sample flows a (75,) one-hot
  action shape unchanged.
- test_existing_3_action_artifact_acts: existing dreamer artifact (action_dim=3)
  still loads and acts (backward compat, registry path).
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from env.dreamer_trading_env import SL_BUCKETS, TP_BUCKETS  # noqa: E402
from models.dreamer_agent import DreamerV3Agent, ReplayBuffer  # noqa: E402


def _tiny_data(n=220, window=64, n_features=9):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    r = (rng.standard_normal(n) * 0.002).astype(np.float32)
    return X, r, window


def test_train_100_steps_75_actions():
    torch.manual_seed(0)
    np.random.seed(0)
    X, r, window = _tiny_data()
    agent = DreamerV3Agent(
        obs_dim=window * X.shape[1] + 5,
        action_dim=75,
        device="cpu",
        embed_dim=64,
        hidden_dim=64,
        stoch_dim=8,
        num_categories=8,
        horizon=6,
        seq_len=8,
    )
    # Fill the replay buffer with random one-hot (75,) transitions.
    obs = np.zeros(agent.obs_dim, dtype=np.float32)
    for _ in range(80):
        action = np.zeros(75, dtype=np.float32)
        action[np.random.randint(0, 75)] = 1.0
        agent.replay_buffer.add(
            obs + np.random.standard_normal(agent.obs_dim).astype(np.float32) * 0.01,
            action,
            float(np.random.standard_normal() * 0.01),
            False,
        )
    assert len(agent.replay_buffer) >= 80
    assert agent.replay_buffer.action.shape[1] == 75, "replay buffer action axis != 75"

    metrics_list = []
    for _ in range(100):
        result = agent.train_step(batch_size=8)
        if result is not None:
            metrics_list.append(result)
    assert metrics_list, "no train_step returned metrics (replay buffer sampling failed)"
    for metrics in metrics_list:
        for key, value in metrics.items():
            assert np.isfinite(value), f"non-finite {key} = {value}"

    # Sample actions from the actor and check decode to valid dir/sl/tp.
    obs_t = torch.randn(4, agent.obs_dim, device="cpu")
    with torch.no_grad():
        embed = agent.encoder(obs_t)
        h, z = agent.rssm.initial_state(4, agent.device)
        h, z, _, _ = agent.rssm.observe(embed, torch.zeros(4, 75), h, z)
        state = agent.rssm.get_state(h, z)
        dist = agent.actor.dist(state)
        idx = dist.sample().cpu().numpy()
        probs = torch.softmax(agent.actor(state), dim=-1).cpu().numpy()
    assert idx.shape == (4,), "actor sample shape wrong"
    assert np.all(idx >= 0) and np.all(idx < 75), f"idx out of [0,75): {idx}"
    for i in range(4):
        dir_idx = int(idx[i]) // 25
        sl_idx = (int(idx[i]) // 5) % 5
        tp_idx = int(idx[i]) % 5
        assert dir_idx in (0, 1, 2)
        assert SL_BUCKETS[sl_idx] in SL_BUCKETS
        assert TP_BUCKETS[tp_idx] in TP_BUCKETS
    assert np.isfinite(probs).all(), "actor probs contain NaN/inf"


def test_act_returns_scalar_75_onehot():
    torch.manual_seed(1)
    agent = DreamerV3Agent(
        obs_dim=11,
        action_dim=75,
        device="cpu",
        embed_dim=32,
        hidden_dim=32,
        stoch_dim=4,
        num_categories=4,
        horizon=4,
        seq_len=8,
    )
    obs = np.random.standard_normal(11).astype(np.float32)
    action, (h, z) = agent.act(obs, deterministic=True)
    assert action.shape == (75,), f"act() returned {action.shape}, expected (75,)"
    assert int(action.sum()) == 1, "act() output is not one-hot"
    idx = int(np.argmax(action))
    assert 0 <= idx < 75
    # prev_action flows through RSSM (75-wide one-hot) on second call.
    action2, _ = agent.act(obs, h, z, deterministic=True)
    assert action2.shape == (75,)


def test_replay_buffer_75_onehot():
    buf = ReplayBuffer(capacity=32, seq_len=4)
    obs = np.zeros(9, dtype=np.float32)
    for i in range(16):
        action = np.zeros(75, dtype=np.float32)
        action[i % 75] = 1.0
        buf.add(obs, action, float(i % 3) / 3.0, i % 8 == 0)
    batch = buf.sample(4)
    assert batch is not None, "buffer did not sample"
    assert batch["action"].shape == (4, 4, 75), (
        f"action batch shape {batch['action'].shape}, expected (4,4,75)"
    )
    # Every sampled action is a valid one-hot vector in [0,75).
    for row in batch["action"].reshape(-1, 75):
        assert int(row.sum()) == 1
        assert int(np.argmax(row)) < 75


def test_existing_3_action_artifact_acts():
    # Load the committed 3-action dreamer artifact via the registry and run one
    # act() to confirm backward compatibility is untouched.
    from models.registry import load_policy

    manifest_path = (
        REPO
        / "artifacts"
        / "models"
        / "dreamer_20261003T230304213562Z"
        / "manifest.json"
    )
    if not manifest_path.exists():
        manifest_path = (
            REPO / "artifacts" / "models" / "dreamer_20261003T230535411560Z" / "manifest.json"
        )
    assert manifest_path.exists(), "no dreamer artifact manifest found"

    policy, manifest = load_policy(str(manifest_path), device="cpu")
    assert policy.action_dim == 3, f"expected action_dim 3, got {policy.action_dim}"
    assert manifest.action_dim == 3
    obs = np.zeros(policy.obs_dim, dtype=np.float32)
    out = policy.act(obs)
    assert out.action in (0, 1, 2), f"3-action policy produced {out.action}"
    assert np.isfinite(out.probs).all()


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
