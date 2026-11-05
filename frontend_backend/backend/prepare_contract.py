from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the leak-free feature contract")
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--macro")
    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    macro_path = Path(args.macro) if args.macro else None
    if macro_path and not macro_path.is_absolute():
        macro_path = ROOT / macro_path

    from train.data import prepare_data

    prepared = prepare_data(data_path, train_end=args.train_end, window=args.window, macro_csv=macro_path)
    output = ROOT / "train" / "feature_contract.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="feature_contract.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(prepared.contract, handle, indent=2)
            handle.write("\n")
        os.replace(temp_name, output)
    except Exception:
        try: os.unlink(temp_name)
        except OSError: pass
        raise
    report = {"source": data_path.relative_to(ROOT).as_posix(), "contract": output.relative_to(ROOT).as_posix(),
              "window": prepared.window, "train_end": args.train_end, "macro_source": macro_path.relative_to(ROOT).as_posix() if macro_path else None,
              "n_features": len(prepared.feature_names), "obs_dim": prepared.window * len(prepared.feature_names) + 5,
              "contract_hash": prepared.contract.get("hash"),
              "train_rows": len(prepared.X_train), "validation_rows": len(prepared.X_test),
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "leakage_check": "Feature pipeline fit on train window only; validation transformed with saved train statistics."}
    report_path = ROOT / "logs" / "prep_latest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
