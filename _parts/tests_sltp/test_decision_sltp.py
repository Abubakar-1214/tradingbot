"""Subtask 7 verification: Decision carries SL/TP fractions; ModelSignalSource
decodes 75-action composite indices into bucket fractions; DecisionEngine
threads fractions through every filter branch.

- test_model_mode_decide_returns_bucket_fractions: stub 75-action artifact
  (sl_tp_mode=model) -> decide() returns Decision with sl_frac/tp_frac from
  SL_BUCKETS/TP_BUCKETS for composite idx (dir*25 + sl_idx*5 + tp_idx).
- test_model_mode_short_mapping_clears_fractions: direction 2 mapped to flat
  when ALLOW_SHORT=false -> fractions None.
- test_rules_mode_decide_returns_none_fractions: legacy 3-action artifact
  (sl_tp_mode=rules) -> Decision sl_frac/tp_frac None.
- test_decision_engine_filter_preserves_fractions: raw Decision fractions
  survive filter() (entry branch).
- test_decision_engine_filter_blocked_branch_preserves_fractions:
  LOW_CONFIDENCE branch also carries fractions (no silent discard).
- test_decision_engine_filter_policyoutput_fractions_none: PolicyOutput raw
  (no fractions) -> filter() emits None fractions.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dataclasses import replace
from datetime import datetime, timezone

import live.model_signal as live_model_signal
from core.model_artifacts import ModelManifest, save_manifest
from live.decision_engine import Decision, DecisionEngine
from live.model_signal import ModelSignalSource
from models.policy import PolicyOutput
from tests.helpers import make_config, sample_df


def _make_artifact(tmp_path, action_dim, sl_tp_mode, window=4, n_features=3):
    """Stub artifact dir: garbage model.pt + feature_contract.json + manifest.json."""
    d = tmp_path / "artifact"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pt").write_bytes(b"\x00" * 8)
    contract_hash = "cafe1234" * 4
    contract = {
        "hash": contract_hash,
        "feature_names": ["f1", "f2", "f3"],
        "pipeline_params": {},
    }
    (d / "feature_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    manifest = ModelManifest(
        model_type="dreamer",
        model_file="model.pt",
        contract_file="feature_contract.json",
        contract_hash=contract_hash,
        window=window,
        n_features=n_features,
        obs_dim=window * n_features + 5,
        action_dim=action_dim,
        allow_short=True,
        symbol="XAUUSD",
        timeframe="H1",
        train_start="2020-01-01",
        train_end="2022-01-01",
        test_end=None,
        created_at="2026-01-01T00:00:00+00:00",
        git_commit=None,
        hyperparams={},
        sl_tp_mode=sl_tp_mode,
    )
    save_manifest(manifest, d)
    return d / "manifest.json", manifest


class FakePolicy:
    """Minimal TradingPolicy stub with a scripted act() sequence."""

    def __init__(self, action_dim, obs_dim, outputs):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.outputs = list(outputs)
        self.executed_actions = []
        self.act_calls = 0

    def reset(self):
        return None

    def act(self, obs):
        self.act_calls += 1
        if not self.outputs:
            raise AssertionError("FakePolicy.act called more times than scripted")
        return self.outputs.pop(0)

    def observe_executed(self, action):
        self.executed_actions.append(int(action))


def _policy_output(action, action_dim):
    probs = np.zeros(action_dim, dtype=np.float64)
    probs[action] = 1.0
    return PolicyOutput(action, probs, 1.0, {})


def _source(cfg, monkeypatch, manifest, action_dim, outputs):
    fake = FakePolicy(action_dim, manifest.obs_dim, outputs)

    def fake_load_policy(_path):
        return fake, manifest

    monkeypatch.setattr(live_model_signal, "load_policy", fake_load_policy)
    monkeypatch.setattr(
        live_model_signal,
        "transform_feature_pipeline",
        lambda *a, **k: (np.zeros((40, manifest.n_features)), None),
    )
    return ModelSignalSource(cfg, burn_in=0), fake


def _model_cfg(manifest_path, allow_short=True):
    cfg = make_config()
    from core.config import ModelConfig

    return replace(
        cfg,
        feature=replace(cfg.feature, window=4, drop_warmup=0),
        model=ModelConfig(
            signal_source="model",
            manifest_path=str(manifest_path),
            live_history_bars=260,
            macro_csv=None,
        ),
        broker=replace(cfg.broker, allow_short=allow_short),
    )


def _bars():
    return sample_df(n=60)


def test_model_mode_decide_returns_bucket_fractions(tmp_path, monkeypatch):
    manifest_path, manifest = _make_artifact(
        tmp_path, action_dim=75, sl_tp_mode="model"
    )
    # composite idx 38 = dir 1 (long) * 25 + sl_idx 2 * 5 + tp_idx 3
    # => sl_frac = SL_BUCKETS[2] = 0.02, tp_frac = TP_BUCKETS[3] = 0.03
    cfg = _model_cfg(manifest_path)
    source, fake = _source(
        cfg,
        monkeypatch,
        manifest,
        75,
        [_policy_output(38, 75)],
    )
    decision = source.decide(_bars(), None) if False else source.decide(
        _bars(), __import__("core.observation", fromlist=["AccountState"]).AccountState(0, 0.0, 0, 0.0, 1.0)
    )
    assert decision.action == 1
    assert decision.reason == "MODEL_SIGNAL"
    assert decision.sl_frac == pytest.approx(0.02)
    assert decision.tp_frac == pytest.approx(0.03)


def test_model_mode_short_mapping_clears_fractions(tmp_path, monkeypatch):
    manifest_path, manifest = _make_artifact(
        tmp_path, action_dim=75, sl_tp_mode="model"
    )
    # composite idx 50 = dir 2 (short) * 25 + sl 0 + tp 0
    cfg = _model_cfg(manifest_path, allow_short=False)
    source, fake = _source(
        cfg,
        monkeypatch,
        manifest,
        75,
        [_policy_output(50, 75)],
    )
    from core.observation import AccountState

    decision = source.decide(_bars(), AccountState(0, 0.0, 0, 0.0, 1.0))
    assert decision.action == 0
    assert decision.info["short_mapped_to_flat"] is True
    assert decision.sl_frac is None
    assert decision.tp_frac is None


def test_rules_mode_decide_returns_none_fractions(tmp_path, monkeypatch):
    manifest_path, manifest = _make_artifact(
        tmp_path, action_dim=3, sl_tp_mode="rules"
    )
    cfg = _model_cfg(manifest_path, allow_short=True)
    source, fake = _source(
        cfg,
        monkeypatch,
        manifest,
        3,
        [_policy_output(1, 3)],
    )
    from core.observation import AccountState

    decision = source.decide(_bars(), AccountState(0, 0.0, 0, 0.0, 1.0))
    assert decision.action == 1
    assert decision.sl_frac is None
    assert decision.tp_frac is None


def test_decision_engine_filter_preserves_fractions():
    engine = DecisionEngine(
        replace(make_config().behavior, session_filter=False, use_kelly=False)
    )
    raw = Decision(1, 0.9, 1.0, "MODEL_SIGNAL", {}, 0.02, 0.03)
    out = engine.filter(raw, datetime.now(timezone.utc), 0, None, [])
    assert out.action == 1
    assert out.sl_frac == pytest.approx(0.02)
    assert out.tp_frac == pytest.approx(0.03)
    assert 0.0 < out.size_multiplier <= 1.0


def test_decision_engine_filter_blocked_branch_preserves_fractions():
    engine = DecisionEngine(
        replace(make_config().behavior, session_filter=False, use_kelly=False)
    )
    raw = Decision(1, 0.3, 1.0, "MODEL_SIGNAL", {}, 0.02, 0.03)
    out = engine.filter(raw, datetime.now(timezone.utc), 0, None, [])
    assert out.reason == "LOW_CONFIDENCE"
    # fractions must NOT be silently discarded even on a blocked branch
    assert out.sl_frac == pytest.approx(0.02)
    assert out.tp_frac == pytest.approx(0.03)
    assert out.size_multiplier == 0.0


def test_decision_engine_filter_policyoutput_fractions_none():
    engine = DecisionEngine(
        replace(make_config().behavior, session_filter=False, use_kelly=False)
    )
    raw = PolicyOutput(1, np.asarray([0.0, 1.0]), 0.9, {})
    out = engine.filter(raw, datetime.now(timezone.utc), 0, None, [])
    assert out.action == 1
    assert out.sl_frac is None
    assert out.tp_frac is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
