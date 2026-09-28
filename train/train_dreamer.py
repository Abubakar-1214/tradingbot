import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_artifacts import ModelManifest, artifact_dir, save_manifest
from env.dreamer_trading_env import (
    DEFAULT_ENV_KWARGS,
    EVAL_ENV_OVERRIDES,
    RealisticTradingEnv,
)
from models.dreamer_agent import DreamerV3Agent
from models.policy import DreamerPolicy
from train.data import prepare_data
from train.evaluate import evaluate_policy, write_evaluation


def _arg(args, name, default):
    return getattr(args, name, default)


def _git_commit():
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


def train(args) -> Path:
    seed = int(_arg(args, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _arg(args, "device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    window = int(_arg(args, "window", 64))
    train_end = _arg(args, "train_end", "2022-01-01")
    data = prepare_data(
        _arg(args, "data", "data/xauusd_1h.csv"),
        train_end=train_end,
        window=window,
        macro_csv=_arg(args, "macro", None),
        test_end=_arg(args, "test_end", None),
    )
    allow_short = bool(_arg(args, "allow_short", True))
    action_dim = 3 if allow_short else 2
    env_kwargs = {
        **DEFAULT_ENV_KWARGS,
        "window": data.window,
        "allow_short": allow_short,
    }
    env = RealisticTradingEnv(
        data.X_train,
        data.r_train,
        timestamps=data.ts_train,
        **env_kwargs,
        seed=seed,
    )
    obs_dim = int(env.observation_space)
    artifact_root = Path(_arg(args, "artifact_root", "artifacts/models"))
    out_dir = artifact_dir(artifact_root, "dreamer", _arg(args, "run_name", None))
    contract_path = out_dir / "feature_contract.json"
    contract_path.write_text(json.dumps(data.contract, indent=2), encoding="utf-8")

    resume = _arg(args, "resume", None)
    if resume:
        agent = DreamerV3Agent.from_checkpoint(resume, device=device)
        if agent.obs_dim != obs_dim or agent.action_dim != action_dim:
            raise ValueError("resume checkpoint dimensions do not match the prepared dataset")
    else:
        agent = DreamerV3Agent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            embed_dim=int(_arg(args, "embed_dim", 256)),
            hidden_dim=int(_arg(args, "hidden_dim", 512)),
            stoch_dim=int(_arg(args, "stoch_dim", 32)),
            num_categories=int(_arg(args, "num_categories", 32)),
            horizon=int(_arg(args, "horizon", 15)),
            seq_len=int(_arg(args, "seq_len", 64)),
        )

    prefill_steps = int(_arg(args, "prefill", 5000))
    obs = env.reset()
    h = z = None
    for _ in range(prefill_steps):
        if resume:
            action, (h, z) = agent.act(obs, h, z, deterministic=False)
        else:
            action = np.zeros(action_dim, dtype=np.float32)
            action[np.random.randint(action_dim)] = 1.0
        next_obs, reward, done, _ = env.step(action)
        agent.replay_buffer.add(obs, action, reward, done)
        obs = env.reset() if done else next_obs
        if done:
            h = z = None

    steps = int(_arg(args, "steps", 100_000))
    train_every = max(1, int(_arg(args, "train_every", 4)))
    batch_size = int(_arg(args, "batch_size", 16))
    save_every = max(1, int(_arg(args, "save_every", 10_000)))
    obs = env.reset()
    h = z = None
    for step in range(steps):
        action, (h, z) = agent.act(obs, h, z, deterministic=False)
        next_obs, reward, done, _ = env.step(action)
        agent.replay_buffer.add(obs, action, reward, done)
        obs = env.reset() if done else next_obs
        if done:
            h = z = None
        if step % train_every == 0:
            agent.train_step(batch_size=batch_size)
        if (step + 1) % save_every == 0:
            agent.save(out_dir / f"checkpoint_{agent.training_step}.pt")

    model_path = out_dir / "model.pt"
    agent.save(model_path)
    manifest = ModelManifest(
        model_type="dreamer",
        model_file=model_path.name,
        contract_file=contract_path.name,
        contract_hash=data.contract["hash"],
        window=data.window,
        n_features=len(data.feature_names),
        obs_dim=obs_dim,
        action_dim=action_dim,
        allow_short=allow_short,
        symbol=_arg(args, "symbol", "XAUUSD"),
        timeframe=_arg(args, "timeframe", "H1"),
        train_start=str(data.ts_train[0]),
        train_end=str(train_end),
        test_end=_arg(args, "test_end", None),
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit(),
        hyperparams={
            "steps": steps,
            "prefill": prefill_steps,
            "batch_size": batch_size,
            "train_every": train_every,
            "embed_dim": int(_arg(args, "embed_dim", 256)),
            "hidden_dim": int(_arg(args, "hidden_dim", 512)),
            "stoch_dim": int(_arg(args, "stoch_dim", 32)),
            "num_categories": int(_arg(args, "num_categories", 32)),
            "horizon": int(_arg(args, "horizon", 15)),
        },
    )
    save_manifest(manifest, out_dir)
    if len(data.X_test) <= data.window + 2:
        raise ValueError("test period must contain more than window+2 feature rows")
    policy = DreamerPolicy(agent)
    eval_metrics = evaluate_policy(
        policy,
        data.X_test,
        data.r_test,
        data.ts_test,
        {**env_kwargs, **EVAL_ENV_OVERRIDES},
    )
    write_evaluation(
        out_dir,
        eval_metrics,
        data.ts_test[0],
        data.ts_test[-1],
        data.contract["hash"],
    )
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Dreamer on the shared trading environment")
    parser.add_argument("--data", default="data/xauusd_1h.csv")
    parser.add_argument("--macro")
    parser.add_argument("--train-end", dest="train_end", default="2022-01-01")
    parser.add_argument("--test-end", dest="test_end")
    parser.add_argument("--run-name")
    parser.add_argument("--artifact-root", default="artifacts/models")
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--prefill", type=int, default=5_000)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--stoch-dim", type=int, default=32)
    parser.add_argument("--num-categories", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--allow-short", dest="allow_short", action="store_true")
    actions.add_argument("--long-only", dest="allow_short", action="store_false")
    parser.set_defaults(allow_short=True)
    args = parser.parse_args(argv)
    print(train(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
