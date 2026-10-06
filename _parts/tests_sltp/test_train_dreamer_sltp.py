"""Subtask 8 verification: train_dreamer --sl-tp-model flag + evaluate_policy
sl_tp_action inference.

- test_cli_flag_parses_sl_tp_model: the argparse flag exists and sets
  sl_tp_model=True (with --long-only preserved).
- test_train_with_flag_manifest_model_75: train() with sl_tp_model=True on a
  tiny slice -> artifact manifest sl_tp_mode='model' + action_dim=75.
- test_train_without_flag_manifest_rules_3: train() without the flag ->
  manifest sl_tp_mode='rules' + action_dim=3 (legacy unchanged).
- test_evaluate_policy_infers_sltp_action: evaluate_policy constructs the env
  with sl_tp_action=True when policy.action_dim > 3 (composite 75-action
  space), and leaves sl_tp_action False for a legacy 3-action policy.
- test_evaluate_model_artifact_runs: a tiny 75-action dreamer artifact
  produced by the flag evaluates on a tiny test slice without crashing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.model_artifacts import load_manifest  # noqa: E402
from tests.helpers import sample_df  # noqa: E402
from train.data import prepare_data  # noqa: E402
from train.train_dreamer import train as train_dreamer  # noqa: E402


def _write_tiny_bars(tmp_path, n=620):
    """Deterministic tiny OHLC CSV (mirrors tests/test_live_integration.py)."""
    frame = sample_df(n=n)
    data_path = tmp_path / "bars.csv"
    frame.to_csv(data_path, index=False)
    return data_path, frame


def _train_kwargs(data_path, frame, *, sl_tp_model, tmp_path, seed=14):
    train_end = frame["time"].iloc[420].isoformat()
    return {
        "data": str(data_path),
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
        "artifact_root": str(tmp_path / "models"),
        "run_name": f"dreamer-{'model' if sl_tp_model else 'rules'}",
        "allow_short": True,
        "seed": seed,
        "sl_tp_model": sl_tp_model,
    }


def _ns(**kwargs):
    from types import SimpleNamespace

    return SimpleNamespace(**kwargs)


def test_cli_flag_parses_sl_tp_model(monkeypatch):
    import train.train_dreamer as td

    captured = {}

    def fake_train(args):
        captured["sl_tp_model"] = args.sl_tp_model
        captured["allow_short"] = args.allow_short
        return Path("fake-artifact")

    monkeypatch.setattr(td, "train", fake_train)
    assert td.main(["--sl-tp-model", "--long-only", "--steps", "2"]) == 0
    assert captured["sl_tp_model"] is True
    assert captured["allow_short"] is False

    captured.clear()
    assert td.main(["--allow-short", "--steps", "2"]) == 0
    assert captured["sl_tp_model"] is False
    assert captured["allow_short"] is True


def test_train_with_flag_manifest_model_75(tmp_path):
    data_path, frame = _write_tiny_bars(tmp_path)
    out_dir = train_dreamer(_ns(**_train_kwargs(
        data_path, frame, sl_tp_model=True, tmp_path=tmp_path
    )))
    manifest = load_manifest(out_dir / "manifest.json")
    assert manifest.sl_tp_mode == "model", (
        f"expected sl_tp_mode='model', got {manifest.sl_tp_mode!r}"
    )
    assert manifest.action_dim == 75, f"expected action_dim 75, got {manifest.action_dim}"
    assert manifest.hyperparams.get("steps") == 2
    # The training env used the composite space: evaluation.json exists
    # (evaluate ran on the tiny test slice without crashing).
    assert (out_dir / "evaluation.json").is_file()


def test_train_without_flag_manifest_rules_3(tmp_path):
    data_path, frame = _write_tiny_bars(tmp_path)
    out_dir = train_dreamer(_ns(**_train_kwargs(
        data_path, frame, sl_tp_model=False, tmp_path=tmp_path
    )))
    manifest = load_manifest(out_dir / "manifest.json")
    assert manifest.sl_tp_mode == "rules", (
        f"expected sl_tp_mode='rules', got {manifest.sl_tp_mode!r}"
    )
    assert manifest.action_dim == 3, f"expected action_dim 3, got {manifest.action_dim}"
    assert manifest.allow_short is True
    assert (out_dir / "evaluation.json").is_file()


def test_evaluate_policy_infers_sltp_action(monkeypatch):
    from train import evaluate as ev

    captured = {}
    orig_env = ev.RealisticTradingEnv

    class CapturingEnv:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self._env = orig_env(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._env, name)

    monkeypatch.setattr(ev, "RealisticTradingEnv", CapturingEnv)

    rng = np.random.default_rng(0)
    window, n_features = 8, 3
    n = 100
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    r = (rng.standard_normal(n) * 0.001).astype(np.float32)
    ts = np.datetime64("2020-01-01", "h") + np.arange(n).astype("timedelta64[h]")

    # A 75-action policy -> env must be built with sl_tp_action=True so the
    # composite action space (75) matches the policy's action_dim.
    from models.dreamer_agent import DreamerV3Agent
    from models.policy import DreamerPolicy

    agent75 = DreamerV3Agent(
        obs_dim=window * n_features + 5,
        action_dim=75,
        device="cpu",
        embed_dim=8,
        hidden_dim=16,
        stoch_dim=2,
        num_categories=3,
        horizon=1,
        seq_len=4,
    )
    metrics = ev.evaluate_policy(
        DreamerPolicy(agent75),
        X, r, ts,
        {"window": window, "allow_short": True},
    )
    assert captured["sl_tp_action"] is True, "75-action policy must build an sl_tp_action env"
    assert metrics["bars"] > 0 and np.isfinite(metrics["final_equity"])

    # Legacy 3-action policy -> sl_tp_action stays False (rules mode).
    captured.clear()
    agent3 = DreamerV3Agent(
        obs_dim=window * n_features + 5,
        action_dim=3,
        device="cpu",
        embed_dim=8,
        hidden_dim=16,
        stoch_dim=2,
        num_categories=3,
        horizon=1,
        seq_len=4,
    )
    metrics3 = ev.evaluate_policy(
        DreamerPolicy(agent3),
        X, r, ts,
        {"window": window, "allow_short": True},
    )
    assert captured.get("sl_tp_action", False) is False
    assert metrics3["bars"] > 0 and np.isfinite(metrics3["final_equity"])


def test_evaluate_model_artifact_runs(tmp_path):
    """End-to-end: a flag-trained 75-action artifact evaluates on a tiny slice."""
    data_path, frame = _write_tiny_bars(tmp_path)
    out_dir = train_dreamer(_ns(**_train_kwargs(
        data_path, frame, sl_tp_model=True, tmp_path=tmp_path
    )))
    manifest = load_manifest(out_dir / "manifest.json")

    from models.registry import load_policy

    policy, loaded_manifest = load_policy(str(out_dir / "manifest.json"), device="cpu")
    assert loaded_manifest.action_dim == 75
    assert policy.action_dim == 75

    data = prepare_data(
        str(data_path),
        train_end=frame["time"].iloc[420].isoformat(),
        window=8,
        test_end=None,
    )
    from train.evaluate import evaluate_policy

    metrics = evaluate_policy(
        policy,
        data.X_test,
        data.r_test,
        data.ts_test,
        {"window": 8, "allow_short": True},
    )
    assert metrics["bars"] > 0
    assert np.isfinite(metrics["total_return_pct"])
    assert np.isfinite(metrics["sharpe"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
