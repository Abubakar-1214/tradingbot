import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from core.model_artifacts import artifact_dir, load_manifest
from env.dreamer_trading_env import DEFAULT_ENV_KWARGS, RealisticTradingEnv
from models.adversarial_training import MarketMakerAgent, SelfPlayTrainer
from models.dreamer_agent import DreamerV3Agent
from models.policy import DreamerPolicy
from train.common import write_model_artifact
from train.data import prepare_data
from train.evaluate import evaluate_policy, write_evaluation


def _manifest_path(value):
    path = Path(value)
    return path / "manifest.json" if path.is_dir() else path


def train(args) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest_path = _manifest_path(args.manifest)
    base_manifest = load_manifest(manifest_path)
    if base_manifest.model_type != "dreamer":
        raise ValueError("--manifest must describe a Dreamer artifact")
    data = prepare_data(
        args.data,
        train_end=base_manifest.train_end,
        window=base_manifest.window,
        macro_csv=args.macro,
        test_end=base_manifest.test_end,
    )
    if data.contract["hash"] != base_manifest.contract_hash:
        raise ValueError("prepared feature contract does not match the Dreamer manifest")
    agent = DreamerV3Agent.from_checkpoint(
        manifest_path.parent / base_manifest.model_file,
        device="cpu",
    )
    allow_short = base_manifest.allow_short
    env = RealisticTradingEnv(
        data.X_train,
        data.r_train,
        timestamps=data.ts_train,
        **{
            **DEFAULT_ENV_KWARGS,
            "window": data.window,
            "allow_short": allow_short,
            "seed": args.seed,
        },
    )
    market_maker = MarketMakerAgent(
        state_dim=env.observation_space,
        hidden_dim=args.mm_hidden_dim,
        learning_rate=args.mm_learning_rate,
        manip_cost=args.manip_cost,
        max_manipulation_rate=args.max_manipulation_rate,
    )
    trainer = SelfPlayTrainer(
        agent,
        market_maker,
        env,
        train_every=args.train_every,
        batch_size=args.batch_size,
    )
    trainer.train(
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
    )
    out_dir = artifact_dir(
        Path(args.artifact_root),
        "dreamer_adversarial",
        args.run_name or None,
    )
    model_path = out_dir / "model.pt"
    agent.save(model_path)
    policy = DreamerPolicy(agent)
    metrics = evaluate_policy(
        policy,
        data.X_test,
        data.r_test,
        data.ts_test,
        {"window": data.window, "allow_short": allow_short},
    )
    write_model_artifact(
        out_dir,
        model_type="dreamer",
        model_file=model_path.name,
        contract=data.contract,
        window=data.window,
        n_features=len(data.feature_names),
        obs_dim=agent.obs_dim,
        action_dim=agent.action_dim,
        allow_short=allow_short,
        symbol=base_manifest.symbol,
        timeframe=base_manifest.timeframe,
        train_start=data.ts_train[0],
        train_end=base_manifest.train_end,
        test_end=base_manifest.test_end,
        hyperparams={
            **base_manifest.hyperparams,
            "adversarial_epochs": args.epochs,
            "adversarial_steps_per_epoch": args.steps_per_epoch,
            "train_every": args.train_every,
            "mm_hidden_dim": args.mm_hidden_dim,
            "max_manipulation_rate": args.max_manipulation_rate,
        },
        extra={
            **base_manifest.extra,
            "adversarial": True,
            "adversarial_finetuned": True,
            "clean_evaluation": True,
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
    parser = argparse.ArgumentParser(description="Fine-tune Dreamer with adversarial market-maker self-play")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data", default="data/xauusd_1h.csv")
    parser.add_argument("--macro")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mm-hidden-dim", type=int, default=128)
    parser.add_argument("--mm-learning-rate", type=float, default=3e-4)
    parser.add_argument("--manip-cost", type=float, default=0.01)
    parser.add_argument("--max-manipulation-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    parser.add_argument("--artifact-root", default="artifacts/models")
    args = parser.parse_args(argv)
    print(train(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
