import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from backtest.engine import prepare_ohlc
from backtest.model_signals import model_signal_frame
from core.model_artifacts import (
    ModelArtifactError,
    ModelManifest,
    load_manifest,
    save_manifest,
)
from core.observation import AccountState, build_observation, obs_dim
from env.dreamer_trading_env import RealisticTradingEnv
from models.dreamer_agent import DreamerV3Agent
from models.dreamer_components import symlog
from models.policy import DreamerPolicy, PolicyOutput
from models.registry import load_policy
from tests.helpers import sample_df
from train.data import prepare_data
from train.evaluate import evaluate_policy, promotion_gate
from train.train_dreamer import train as train_dreamer
from train.train_ppo import train as train_ppo


def _csv(path, n=620):
    frame = sample_df(n=n)
    frame.to_csv(path, index=False)
    return frame


def _train_end(frame):
    return frame["time"].iloc[420].isoformat()


def test_observation_matches_realistic_environment():
    features = np.arange(48, dtype=np.float32).reshape(12, 4)
    env = RealisticTradingEnv(features, np.zeros(len(features), dtype=np.float32), window=4)
    env.pos = -1
    env.trade_pnl = 0.025
    env.bars_in_trade = 8
    env.drawdown = 0.12
    env.equity = 0.93
    account = AccountState(-1, 0.025, 8, 0.12, 0.93)
    expected = build_observation(features[env.t - env.window:env.t], account)
    np.testing.assert_array_equal(env._get_obs(), expected)
    assert expected.shape == (obs_dim(4, 4),)


def test_prepare_data_fits_scaler_on_train_only(tmp_path):
    frame = _csv(tmp_path / "original.csv")
    changed = frame.copy()
    split = 420
    changed.loc[split:, "close"] *= 1.8
    changed.loc[split:, "high"] *= 1.8
    changed.loc[split:, "low"] *= 1.8
    changed.loc[split:, "open"] *= 1.8
    changed.to_csv(tmp_path / "changed.csv", index=False)

    first = prepare_data(tmp_path / "original.csv", _train_end(frame), window=8)
    second = prepare_data(tmp_path / "changed.csv", _train_end(frame), window=8)
    assert first.contract["scaler"] == second.contract["scaler"]
    assert first.contract["hash"] == second.contract["hash"]
    assert len(first.X_train) == len(first.r_train) == len(first.ts_train)
    assert len(first.X_test) == len(first.r_test) == len(first.ts_test)
    assert np.isfinite(first.X_train).all()
    assert np.isfinite(first.X_test).all()
    assert np.isfinite(first.r_train).all()
    assert np.isfinite(first.r_test).all()
    assert first.ts_train[-1] < np.datetime64(_train_end(frame))
    assert first.ts_test[0] >= np.datetime64(_train_end(frame))


def _manifest(directory, model_type="ppo"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.pt").write_bytes(b"model")
    contract = {"hash": "abc123"}
    (directory / "feature_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    manifest = ModelManifest(
        model_type=model_type,
        model_file="model.pt",
        contract_file="feature_contract.json",
        contract_hash="abc123",
        window=8,
        n_features=3,
        obs_dim=29,
        action_dim=3,
        allow_short=True,
        symbol="XAUUSD",
        timeframe="H1",
        train_start="2020-01-01",
        train_end="2022-01-01",
        test_end=None,
        created_at="2024-01-01T00:00:00+00:00",
        git_commit=None,
        hyperparams={},
    )
    return save_manifest(manifest, directory)


def test_manifest_round_trip_and_validation(tmp_path):
    path = _manifest(tmp_path / "artifact")
    loaded = load_manifest(path)
    assert loaded.model_type == "ppo"
    assert loaded.obs_dim == 29

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contract_hash"] = "wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="hash mismatch"):
        load_manifest(path)

    payload["contract_hash"] = "abc123"
    payload["obs_dim"] = 28
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="obs_dim mismatch"):
        load_manifest(path)

    payload["obs_dim"] = 29
    path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "artifact" / "model.pt").unlink()
    with pytest.raises(ModelArtifactError, match="model file not found"):
        load_manifest(path)


def test_world_model_loss_matches_extracted_phase():
    torch.manual_seed(12)
    agent = DreamerV3Agent(
        obs_dim=11,
        action_dim=3,
        embed_dim=8,
        hidden_dim=16,
        stoch_dim=2,
        num_categories=3,
        horizon=1,
        seq_len=3,
        use_amp=False,
    )
    batch_size, sequence = 2, 3
    actions = F.one_hot(torch.randint(0, 3, (batch_size, sequence)), 3).float()
    batch = {
        "obs": np.random.default_rng(4).normal(size=(batch_size, sequence, 11)).astype(np.float32),
        "action": actions.numpy(),
        "reward": np.random.default_rng(5).normal(size=(batch_size, sequence)).astype(np.float32),
        "done": np.zeros((batch_size, sequence), dtype=np.float32),
    }

    def legacy_loss():
        obs = torch.as_tensor(batch["obs"], device=agent.device)
        action = torch.as_tensor(batch["action"], device=agent.device)
        reward = torch.as_tensor(batch["reward"], device=agent.device)
        done = torch.as_tensor(batch["done"], device=agent.device)
        B, T = obs.shape[:2]
        with agent._autocast():
            embed = agent.encoder(obs.reshape(B * T, -1)).reshape(B, T, -1)
            h, z = agent.rssm.initial_state(B, agent.device)
            states = []
            kl_losses = []
            for t in range(T):
                if t > 0:
                    keep = (1.0 - done[:, t - 1]).unsqueeze(-1)
                    h = h * keep
                    z = z * keep
                h, z, prior, posterior = agent.rssm.observe(
                    embed[:, t], action[:, t], h, z
                )
                states.append(agent.rssm.get_state(h, z))
                kl_losses.append(
                    agent.rssm.kl_loss(
                        prior,
                        posterior,
                        agent.free_nats,
                        agent.kl_dyn_scale,
                        agent.kl_rep_scale,
                    )
                )
            states = torch.stack(states, dim=1).reshape(B * T, -1)
            recon = ((agent.decoder(states).float() - symlog(obs.reshape(B * T, -1))) ** 2).sum(-1).mean()
            reward_loss = F.mse_loss(
                agent.reward_predictor(states).float(), symlog(reward.reshape(B * T))
            )
            kl_loss = torch.stack(kl_losses).mean()
            return recon + reward_loss + kl_loss

    torch.manual_seed(33)
    old_loss = legacy_loss()
    torch.manual_seed(33)
    new_loss, metrics = agent.compute_world_model_loss(batch)
    torch.testing.assert_close(new_loss, old_loss)
    torch.testing.assert_close(metrics["world_model_loss"], old_loss)


def test_policy_observe_executed_and_reset():
    torch.manual_seed(7)
    agent = DreamerV3Agent(
        obs_dim=11,
        action_dim=3,
        embed_dim=8,
        hidden_dim=16,
        stoch_dim=2,
        num_categories=3,
        horizon=1,
        seq_len=3,
        use_amp=False,
    )
    policy = DreamerPolicy(agent)
    obs = np.linspace(-1, 1, 11, dtype=np.float32)
    next_obs = obs[::-1].copy()

    initial = policy.act(obs)
    policy.observe_executed(0)
    after_flat = policy.act(next_obs)
    policy.reset()
    policy.act(obs)
    policy.observe_executed(1)
    after_long = policy.act(next_obs)
    assert not np.allclose(after_flat.info["latent"], after_long.info["latent"])
    policy.reset()
    repeated = policy.act(obs)
    np.testing.assert_allclose(initial.probs, repeated.probs)
    np.testing.assert_allclose(initial.info["latent"], repeated.info["latent"])


class FlatPolicy:
    action_dim = 3
    obs_dim = 13

    def reset(self):
        return None

    def act(self, obs):
        return PolicyOutput(0, np.array([1.0, 0.0, 0.0]), 1.0)

    def observe_executed(self, action):
        return None


def test_evaluation_flat_policy_and_promotion_gate():
    features = np.zeros((80, 2), dtype=np.float32)
    returns = np.linspace(-0.002, 0.003, 80, dtype=np.float32)
    timestamps = pd.date_range("2023-01-01", periods=80, freq="h").to_numpy()
    metrics = evaluate_policy(
        FlatPolicy(),
        features,
        returns,
        timestamps,
        {"window": 4, "random_start": False},
    )
    assert metrics["trades"] == 0
    assert metrics["buy_hold_return_pct"] == pytest.approx(
        np.expm1(returns[4:4 + metrics["bars"]].sum()) * 100.0
    )
    assert promotion_gate(
        {"sharpe": 0.7, "max_dd_pct": 12.0, "trades": 22, "total_return_pct": 2.0}
    )
    assert not promotion_gate(
        {"sharpe": 0.2, "max_dd_pct": 30.0, "trades": 4, "total_return_pct": -2.0}
    )


def test_ppo_and_dreamer_training_artifacts_and_resume(tmp_path):
    frame = _csv(tmp_path / "bars.csv")
    train_end = _train_end(frame)

    ppo_dir = train_ppo(
        SimpleNamespace(
            data=tmp_path / "bars.csv",
            train_end=train_end,
            window=8,
            timesteps=32,
            n_envs=1,
            n_steps=16,
            batch_size=8,
            chunk_steps=32,
            hidden_sizes=[16],
            artifact_root=tmp_path / "artifacts",
            run_name="ppo-smoke",
            allow_short=False,
            seed=10,
        )
    )
    for filename in ("model.zip", "feature_contract.json", "manifest.json", "evaluation.json"):
        assert (ppo_dir / filename).is_file()
    ppo_policy, _ = load_policy(ppo_dir / "manifest.json")
    ppo_output = ppo_policy.act(
        np.zeros(ppo_policy.obs_dim, dtype=np.float32)
    )
    assert ppo_output.probs.sum() == pytest.approx(1.0)
    assert ppo_output.action < ppo_policy.action_dim
    signals = model_signal_frame(
        ppo_dir / "manifest.json",
        frame,
        frame["time"].iloc[420],
        frame["time"].iloc[-1],
    )
    assert signals.index.equals(pd.DatetimeIndex(frame["time"]))
    assert set(signals.unique()).issubset({-1, 0, 1})
    assert "signal" in prepare_ohlc(frame.assign(signal=signals.to_numpy()))

    dreamer_args = {
        "data": tmp_path / "bars.csv",
        "train_end": train_end,
        "window": 8,
        "steps": 2,
        "prefill": 8,
        "batch_size": 1,
        "train_every": 1,
        "save_every": 100,
        "seq_len": 4,
        "embed_dim": 8,
        "hidden_dim": 16,
        "stoch_dim": 2,
        "num_categories": 3,
        "horizon": 1,
        "artifact_root": tmp_path / "artifacts",
        "allow_short": True,
        "seed": 11,
    }
    dreamer_dir = train_dreamer(SimpleNamespace(**dreamer_args, run_name="dreamer-smoke"))
    for filename in ("model.pt", "feature_contract.json", "manifest.json", "evaluation.json"):
        assert (dreamer_dir / filename).is_file()
    dreamer_policy, _ = load_policy(dreamer_dir / "manifest.json")
    dreamer_output = dreamer_policy.act(
        np.zeros(dreamer_policy.obs_dim, dtype=np.float32)
    )
    assert dreamer_output.probs.sum() == pytest.approx(1.0)
    assert dreamer_output.action < dreamer_policy.action_dim
    before = DreamerV3Agent.from_checkpoint(dreamer_dir / "model.pt").training_step

    resume_args = {
        **dreamer_args,
        "resume": dreamer_dir / "model.pt",
        "prefill": 4,
        "steps": 1,
        "run_name": "dreamer-resume",
    }
    resumed = train_dreamer(SimpleNamespace(**resume_args))
    after = DreamerV3Agent.from_checkpoint(resumed / "model.pt").training_step
    assert after > before
