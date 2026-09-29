import argparse
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from core.model_artifacts import artifact_dir
from env.dreamer_trading_env import (
    DEFAULT_ENV_KWARGS,
    EVAL_ENV_OVERRIDES,
    RealisticTradingEnv,
)
from models.policy import TransformerPolicy
from models.transformer_policy import (
    TransformerAgentWrapper,
    compute_gae,
)
from train.common import write_model_artifact
from train.data import prepare_data
from train.evaluate import evaluate_policy, write_evaluation


def _arg(args, name, default):
    return getattr(args, name, default)


def train(args) -> Path:
    seed = int(_arg(args, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = prepare_data(
        _arg(args, "data", "data/xauusd_h1.csv"),
        train_end=_arg(args, "train_end", "2022-01-01"),
        window=int(_arg(args, "window", 64)),
        macro_csv=_arg(args, "macro", None),
        test_end=_arg(args, "test_end", None),
    )
    allow_short = bool(_arg(args, "allow_short", True))
    action_dim = 3 if allow_short else 2
    rollout_steps = max(1, int(_arg(args, "rollout_steps", 128)))
    batch_size = max(1, int(_arg(args, "batch_size", 64)))
    configured_minibatches = int(_arg(args, "num_minibatches", 4))
    env = RealisticTradingEnv(
        data.X_train,
        data.r_train,
        timestamps=data.ts_train,
        **{
            **DEFAULT_ENV_KWARGS,
            "window": data.window,
            "allow_short": allow_short,
            "seed": seed,
        },
    )
    agent = TransformerAgentWrapper(
        action_dim=action_dim,
        n_features=len(data.feature_names),
        window=data.window,
        hidden_dim=int(_arg(args, "hidden_dim", 128)),
        num_heads=int(_arg(args, "num_heads", 4)),
        num_layers=int(_arg(args, "num_layers", 2)),
        learning_rate=float(_arg(args, "learning_rate", 3e-4)),
        update_epochs=int(_arg(args, "update_epochs", 4)),
        num_minibatches=min(configured_minibatches, math.ceil(rollout_steps / batch_size)),
        device="cpu",
    )
    total_steps = int(_arg(args, "steps", 100_000))
    save_every = int(_arg(args, "save_every", 0))
    if total_steps < 1:
        raise ValueError("steps must be positive")
    out_dir = artifact_dir(
        Path(_arg(args, "artifact_root", "artifacts/models")),
        "transformer",
        _arg(args, "run_name", None),
    )

    obs = env.reset()
    steps_done = 0
    while steps_done < total_steps:
        count = min(rollout_steps, total_steps - steps_done)
        observations = []
        actions = []
        old_logp = []
        values = []
        rewards = []
        dones = []
        for _ in range(count):
            observations.append(obs.copy())
            action, logp, value = agent.act(obs, deterministic=False)
            one_hot = np.zeros(action_dim, dtype=np.float32)
            one_hot[action] = 1.0
            obs, reward, done, _ = env.step(one_hot)
            actions.append(action)
            old_logp.append(logp)
            values.append(value)
            rewards.append(reward)
            dones.append(float(done))
            if done:
                obs = env.reset()
        if dones[-1]:
            last_value = 0.0
        else:
            was_training = agent.critic.training
            agent.critic.eval()
            with torch.no_grad():
                last_value = float(agent.critic(agent._tokens(obs)).item())
            agent.critic.train(was_training)
        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            last_value,
            gamma=float(_arg(args, "gamma", 0.99)),
            lam=float(_arg(args, "gae_lambda", 0.95)),
        )
        minibatch = {
            "obs": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "old_logp": np.asarray(old_logp, dtype=np.float32),
            "advantages": advantages.cpu().numpy(),
            "returns": returns.cpu().numpy(),
        }
        agent.train_step(minibatch)
        steps_done += count
        if save_every > 0 and steps_done % save_every == 0:
            agent.save(out_dir / f"checkpoint_{steps_done}.pt")

    model_path = out_dir / "model.pt"
    agent.save(model_path)
    write_model_artifact(
        out_dir,
        model_type="transformer",
        model_file=model_path.name,
        contract=data.contract,
        window=data.window,
        n_features=len(data.feature_names),
        obs_dim=agent.obs_dim,
        action_dim=action_dim,
        allow_short=allow_short,
        symbol=_arg(args, "symbol", "XAUUSD"),
        timeframe=_arg(args, "timeframe", "H1"),
        train_start=data.ts_train[0],
        train_end=_arg(args, "train_end", "2022-01-01"),
        test_end=_arg(args, "test_end", None),
        hyperparams={
            "steps": total_steps,
            "rollout_steps": rollout_steps,
            "batch_size": batch_size,
            "hidden_dim": agent.hidden_dim,
            "num_heads": agent.num_heads,
            "num_layers": agent.num_layers,
            "update_epochs": agent.update_epochs,
            "num_minibatches": agent.num_minibatches,
        },
    )
    if len(data.X_test) <= data.window + 2:
        raise ValueError("test period must contain more than window+2 feature rows")
    policy = TransformerPolicy(agent)
    metrics = evaluate_policy(
        policy,
        data.X_test,
        data.r_test,
        data.ts_test,
        {
            **DEFAULT_ENV_KWARGS,
            **EVAL_ENV_OVERRIDES,
            "window": data.window,
            "allow_short": allow_short,
        },
    )
    write_evaluation(
        out_dir,
        metrics,
        data.ts_test[0],
        data.ts_test[-1],
        data.contract["hash"],
    )
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a Transformer PPO policy")
    parser.add_argument("--data", default="data/xauusd_h1.csv")
    parser.add_argument("--macro")
    parser.add_argument("--train-end", dest="train_end", default="2022-01-01")
    parser.add_argument("--test-end", dest="test_end")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    parser.add_argument("--artifact-root", default="artifacts/models")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--allow-short", dest="allow_short", action="store_true")
    actions.add_argument("--long-only", dest="allow_short", action="store_false")
    parser.set_defaults(allow_short=True)
    args = parser.parse_args(argv)
    print(train(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
