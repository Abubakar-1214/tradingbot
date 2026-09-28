import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_artifacts import artifact_dir
from models.ensemble import EnsemblePolicy
from models.registry import load_policy
from train.common import write_model_artifact
from train.evaluate import _evaluation_frame, evaluate_policy, write_evaluation


def _member_path(value):
    path = Path(value)
    return path / "manifest.json" if path.is_dir() else path


def _train_members(args):
    if args.base_type == "ppo":
        from train.train_ppo import train as train_base
    elif args.base_type == "dreamer":
        from train.train_dreamer import train as train_base
    else:
        from train.train_transformer import train as train_base
    result = []
    for seed in range(args.members):
        values = vars(args).copy()
        values["run_name"] = f"{args.run_name or 'ensemble'}_{args.base_type}_{seed}"
        values["seed"] = seed
        result.append(train_base(SimpleNamespace(**values)) / "manifest.json")
    return result


def train(args) -> Path:
    member_paths = (
        [_member_path(path) for path in args.from_manifests]
        if args.from_manifests
        else _train_members(args)
    )
    if not member_paths:
        raise ValueError("at least one member manifest is required")
    loaded = [load_policy(path) for path in member_paths]
    policies = [item[0] for item in loaded]
    manifests = [item[1] for item in loaded]
    contract_hash = manifests[0].contract_hash
    if any(manifest.contract_hash != contract_hash for manifest in manifests):
        raise ValueError("ensemble members must share the same feature contract hash")
    reference = manifests[0]
    for manifest in manifests[1:]:
        if (
            manifest.window != reference.window
            or manifest.n_features != reference.n_features
            or manifest.obs_dim != reference.obs_dim
        ):
            raise ValueError("ensemble members must share observation dimensions")
    requested_weights = args.weights or None
    ensemble = EnsemblePolicy(
        policies,
        min_agreement=args.min_agreement,
        vote=args.vote,
        weights=requested_weights,
    )
    out_dir = artifact_dir(
        Path(args.artifact_root),
        "ensemble",
        args.run_name or None,
    )
    model_file = "ensemble.json"
    (out_dir / model_file).write_text(
        json.dumps(
            {
                "min_agreement": ensemble.min_agreement,
                "vote": ensemble.vote,
                "weights": ensemble.weights.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    first_manifest_path = member_paths[0]
    contract_path = first_manifest_path.parent / reference.contract_file
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    member_refs = [
        os.path.relpath(path.resolve(), out_dir.resolve()) for path in member_paths
    ]
    write_model_artifact(
        out_dir,
        model_type="ensemble",
        model_file=model_file,
        contract=contract,
        window=reference.window,
        n_features=reference.n_features,
        obs_dim=reference.obs_dim,
        action_dim=ensemble.action_dim,
        allow_short=any(manifest.allow_short for manifest in manifests),
        symbol=reference.symbol,
        timeframe=reference.timeframe,
        train_start=reference.train_start,
        train_end=reference.train_end,
        test_end=reference.test_end,
        hyperparams={"members": len(member_paths)},
        members=member_refs,
        extra={
            "min_agreement": ensemble.min_agreement,
            "vote": ensemble.vote,
            "weights": ensemble.weights.tolist(),
        },
    )
    X_test, r_test, ts_test = _evaluation_frame(
        args.data,
        contract,
        reference.window,
        args.eval_start or args.train_end,
        args.test_end,
        args.macro,
    )
    if len(X_test) <= reference.window + 2:
        raise ValueError("test period must contain more than window+2 feature rows")
    metrics = evaluate_policy(
        ensemble,
        X_test,
        r_test,
        ts_test,
        {"window": reference.window, "allow_short": ensemble.action_dim == 3},
    )
    write_evaluation(
        out_dir,
        metrics,
        ts_test[0],
        ts_test[-1],
        contract_hash,
    )
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train or assemble a trading-policy ensemble")
    parser.add_argument("--base-type", choices=["dreamer", "ppo", "transformer"], default="dreamer")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--from", dest="from_manifests", nargs="+")
    parser.add_argument("--min-agreement", type=float, default=0.6)
    parser.add_argument("--vote", choices=["soft", "hard"], default="soft")
    parser.add_argument("--weights", type=float, nargs="*")
    parser.add_argument("--data", default="data/xauusd_1h.csv")
    parser.add_argument("--macro")
    parser.add_argument("--train-end", default="2022-01-01")
    parser.add_argument("--eval-start")
    parser.add_argument("--test-end")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--prefill", type=int, default=5_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--stoch-dim", type=int, default=32)
    parser.add_argument("--num-categories", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    parser.add_argument("--artifact-root", default="artifacts/models")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--allow-short", dest="allow_short", action="store_true")
    actions.add_argument("--long-only", dest="allow_short", action="store_false")
    parser.set_defaults(allow_short=True)
    args = parser.parse_args(argv)
    if args.members < 1:
        parser.error("--members must be positive")
    print(train(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
