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
from models.dreamer_agent import DreamerV3Agent
from models.meta_learning import MAMLTrader, MarketRegimeGenerator
from models.policy import DreamerPolicy
from train.common import write_model_artifact
from train.data import prepare_data
from train.evaluate import evaluate_policy, write_evaluation


def _manifest_path(value):
    path = Path(value)
    return path / "manifest.json" if path.is_dir() else path


def train(args) -> Path:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    manifest_path = _manifest_path(args.manifest)
    base_manifest = load_manifest(manifest_path)
    if base_manifest.model_type != "dreamer":
        raise ValueError("--manifest must describe a Dreamer artifact")
    train_end = args.train_end or base_manifest.train_end
    test_end = args.test_end or base_manifest.test_end
    data = prepare_data(
        args.data,
        train_end=train_end,
        window=base_manifest.window,
        macro_csv=args.macro,
        test_end=test_end,
    )
    if data.contract["hash"] != base_manifest.contract_hash:
        raise ValueError("prepared feature contract does not match the Dreamer manifest")
    regimes = MarketRegimeGenerator.generate_regimes(
        data.X_train,
        data.r_train,
        data.ts_train,
        data.window,
        min_len=args.min_len,
    )
    if not regimes:
        raise ValueError(
            "no contiguous training-data regime segments meet min_len; reduce --min-len"
        )
    agent = DreamerV3Agent.from_checkpoint(
        manifest_path.parent / base_manifest.model_file,
        device="cpu",
    )
    maml = MAMLTrader(
        agent,
        meta_lr=args.meta_lr,
        adapt_lr=args.adapt_lr,
        adapt_steps=args.adapt_steps,
    )
    losses = maml.meta_train(
        regimes,
        num_epochs=args.epochs,
        tasks_per_batch=args.tasks_per_batch,
        batch_size=args.batch_size,
    )
    out_dir = artifact_dir(
        Path(args.artifact_root),
        "dreamer_meta",
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
        {"window": data.window, "allow_short": base_manifest.allow_short},
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
        allow_short=base_manifest.allow_short,
        symbol=base_manifest.symbol,
        timeframe=base_manifest.timeframe,
        train_start=data.ts_train[0],
        train_end=train_end,
        test_end=test_end,
        hyperparams={
            **base_manifest.hyperparams,
            "meta_epochs": args.epochs,
            "tasks_per_batch": args.tasks_per_batch,
            "meta_lr": args.meta_lr,
            "adapt_lr": args.adapt_lr,
            "adapt_steps": args.adapt_steps,
            "meta_loss": losses[-1]["meta_loss"],
        },
        extra={**base_manifest.extra, "meta_trained": True},
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
    parser = argparse.ArgumentParser(description="Meta-train a Dreamer world model on causal regimes")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data", default="data/xauusd_h1.csv")
    parser.add_argument("--macro")
    parser.add_argument("--train-end")
    parser.add_argument("--test-end")
    parser.add_argument("--min-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--tasks-per-batch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--meta-lr", type=float, default=1e-4)
    parser.add_argument("--adapt-lr", type=float, default=1e-3)
    parser.add_argument("--adapt-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    parser.add_argument("--artifact-root", default="artifacts/models")
    args = parser.parse_args(argv)
    print(train(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
