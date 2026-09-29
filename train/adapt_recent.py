import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_artifacts import artifact_dir, load_manifest
from models.dreamer_agent import DreamerV3Agent
from models.meta_learning import MAMLTrader, build_task_buffer
from models.policy import DreamerPolicy
from train.common import write_model_artifact
from train.data import prepare_data
from train.evaluate import evaluate_policy, write_evaluation


def _manifest_path(value):
    path = Path(value)
    return path / "manifest.json" if path.is_dir() else path


def adapt(args) -> Path:
    manifest_path = _manifest_path(args.manifest)
    base_manifest = load_manifest(manifest_path)
    if base_manifest.model_type != "dreamer":
        raise ValueError("--manifest must describe a Dreamer artifact")
    if args.bars < 1:
        raise ValueError("--bars must be positive")
    data = prepare_data(
        args.data,
        train_end=base_manifest.train_end,
        window=base_manifest.window,
        macro_csv=args.macro,
        test_end=base_manifest.test_end,
    )
    if data.contract["hash"] != base_manifest.contract_hash:
        raise ValueError("prepared feature contract does not match the Dreamer manifest")
    bars = min(args.bars, len(data.X_train))
    recent_features = data.X_train[-bars:]
    recent_returns = data.r_train[-bars:]
    agent = DreamerV3Agent.from_checkpoint(
        manifest_path.parent / base_manifest.model_file,
        device="cpu",
    )
    if bars < data.window + agent.replay_buffer.seq_len + 2:
        raise ValueError(
            "recent training window must contain at least window + sequence length + 2 bars"
        )
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    buffer = build_task_buffer(
        agent,
        {
            "features": recent_features,
            "returns": recent_returns,
            "timestamps": data.ts_train[-bars:],
            "window": data.window,
        },
        steps=len(recent_features) - data.window - 1,
        seed=42,
    )
    adapted_agent = MAMLTrader(
        agent,
        meta_lr=args.meta_lr,
        adapt_lr=args.adapt_lr,
        adapt_steps=args.steps,
    ).fast_adapt(
        buffer,
        steps=args.steps,
        batch_size=args.batch_size,
    )
    out_dir = artifact_dir(
        Path(args.artifact_root),
        "dreamer_adapted",
        args.run_name or None,
    )
    model_path = out_dir / "model.pt"
    adapted_agent.save(model_path)
    policy = DreamerPolicy(adapted_agent)
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
        obs_dim=adapted_agent.obs_dim,
        action_dim=adapted_agent.action_dim,
        allow_short=base_manifest.allow_short,
        symbol=base_manifest.symbol,
        timeframe=base_manifest.timeframe,
        train_start=data.ts_train[0],
        train_end=base_manifest.train_end,
        test_end=base_manifest.test_end,
        hyperparams={
            **base_manifest.hyperparams,
            "adaptation_bars": bars,
            "adaptation_steps": args.steps,
            "adapt_lr": args.adapt_lr,
        },
        extra={
            **base_manifest.extra,
            "adapted_from": str(manifest_path.resolve()),
            "meta_adapted": True,
            "reevaluated": True,
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
    parser = argparse.ArgumentParser(description="Adapt a Dreamer artifact on recent training history")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bars", type=int, required=True)
    parser.add_argument("--data", default="data/xauusd_h1.csv")
    parser.add_argument("--macro")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--meta-lr", type=float, default=1e-4)
    parser.add_argument("--adapt-lr", type=float, default=1e-3)
    parser.add_argument("--run-name")
    parser.add_argument("--artifact-root", default="artifacts/models")
    args = parser.parse_args(argv)
    print(adapt(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
