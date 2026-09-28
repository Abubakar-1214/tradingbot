import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_artifacts import artifact_dir, load_manifest
from train.common import write_model_artifact


def make_manifest(args) -> Path:
    dreamer_manifest_path = Path(args.dreamer)
    if dreamer_manifest_path.is_dir():
        dreamer_manifest_path /= "manifest.json"
    dreamer = load_manifest(dreamer_manifest_path)
    if dreamer.model_type != "dreamer":
        raise ValueError("--dreamer must reference a Dreamer manifest")
    source_dir = dreamer_manifest_path.parent
    source_model = source_dir / dreamer.model_file
    contract = json.loads(
        (source_dir / dreamer.contract_file).read_text(encoding="utf-8")
    )
    out_dir = artifact_dir(
        Path(args.artifact_root),
        "dreamer_mcts",
        args.run_name or None,
    )
    model_file = os.path.relpath(source_model.resolve(), out_dir.resolve())
    write_model_artifact(
        out_dir,
        model_type="dreamer_mcts",
        model_file=model_file,
        contract=contract,
        window=dreamer.window,
        n_features=dreamer.n_features,
        obs_dim=dreamer.obs_dim,
        action_dim=dreamer.action_dim,
        allow_short=dreamer.allow_short,
        symbol=dreamer.symbol,
        timeframe=dreamer.timeframe,
        train_start=dreamer.train_start,
        train_end=dreamer.train_end,
        test_end=dreamer.test_end,
        hyperparams=dreamer.hyperparams,
        extra={
            "mcts_simulations": int(args.simulations),
            "c_puct": float(args.c_puct),
        },
    )
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create a Dreamer MCTS artifact manifest")
    parser.add_argument("--dreamer", required=True)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--artifact-root", default="artifacts/models")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    if args.simulations < 1 or args.c_puct < 0:
        parser.error("simulations must be positive and c-puct must be non-negative")
    print(make_manifest(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
