"""Subtask 9 verification: SL/TP dual-mode integration tests.

(a) Backward-compat smoke — the REAL 3-action dreamer artifact
    (artifacts/models/dreamer_20261003T230304213562Z, manifest has NO
    sl_tp_mode key -> defaults to "rules") in SLTP_MODE=rules, dry-run
    through LiveTrader + MockBroker: Decision carries None fractions and the
    order submit-line SL/TP come from the ATR rule path (entry -+ atr*mult)
    — no model fractions attached.

(b) Model-mode end-to-end — stub 75-action artifact (sl_tp_mode=model) ->
    ModelSignalSource -> Decision carries sl_frac/tp_frac -> TradeExecutor
    attaches clamped SL/TP prices entry*(1-+frac) via MockBroker.  Also
    driven through LiveTrader.on_closed_bar to prove the live wiring.

(c) Startup guard — both mismatch combos raise clear ValueError at the
    startup path (enforce_model_promotion_gate -> enforce_sltp_compatibility):
    SLTP_MODE=model + rules artifact ("cannot supply.*SL/TP") and
    SLTP_MODE=rules + model artifact ("SILENTLY DISCARD").
"""

import json
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.config import (  # noqa: E402
    AppConfig,
    BrokerConfig,
    ModelConfig,
    TradingBehaviorConfig,
    load_config,
)
from core.model_artifacts import ModelManifest, load_manifest, save_manifest  # noqa: E402
from core.observation import AccountState  # noqa: E402
from live import live_trade_mt5  # noqa: E402
from live.live_trade_mt5 import LiveTrader  # noqa: E402
from live.mock_broker import MockBroker  # noqa: E402
from live.model_signal import ModelSignalSource  # noqa: E402
from models.policy import PolicyOutput  # noqa: E402
from tests.helpers import (  # noqa: E402
    EQUITY,
    FakeClock,
    build_executor,
    calm_md,
    make_config,
    sample_df,
)

import live.model_signal as live_model_signal  # noqa: E402

BROKER_KW = dict(symbol="XAUUSD", allow_short=True, sl_atr_mult=2.0, tp_atr_mult=3.0)

# --------------------------------------------------------------------------- #
# Shared helpers (patterns taken from test_executor_sltp / test_decision_sltp)
# --------------------------------------------------------------------------- #


def _last_submit_sl_tp(ex):
    """MockBroker logs TWO lines per order: 'submit ... sl=.. tp=..' then
    '  -> filled ticket=.. at ..'.  Scan for the submit line (the fill line
    has no sl/tp fields)."""
    for line in reversed(ex.broker.order_log()):
        if line.startswith("submit "):
            m = re.search(r"sl=(\S+) tp=(\S+)$", line)
            assert m, f"submit line missing sl/tp: {line!r}"
            return float(m.group(1)), float(m.group(2))
    raise AssertionError("no submit line found in order log")


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


def _write_stub_artifact(tmp_path, sl_tp_mode, action_dim):
    """Create a minimal valid artifact directory for the startup guard tests."""
    d = tmp_path / "artifact"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pt").write_bytes(b"\x00" * 8)
    contract_hash = "c0ffee00" * 4
    (d / "feature_contract.json").write_text(
        json.dumps({"hash": contract_hash, "features": ["f1"]}), encoding="utf-8"
    )
    manifest = ModelManifest(
        model_type="dreamer",
        model_file="model.pt",
        contract_file="feature_contract.json",
        contract_hash=contract_hash,
        window=64,
        n_features=15,
        obs_dim=64 * 15 + 5,
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
    return d / "manifest.json"


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
    """Signal-side config (feature window 4, drop_warmup 0, model source)."""
    cfg = make_config()
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


def _executor_model_cfg(bounds=(0.005, 0.05, 0.005, 0.05)):
    """Executor-side config with SLTP_MODE=model + fraction bounds."""
    base = make_config()
    sl_min, sl_max, tp_min, tp_max = bounds
    return AppConfig(
        risk=base.risk,
        broker=BrokerConfig(**BROKER_KW),
        behavior=TradingBehaviorConfig(
            sl_tp_mode="model",
            sl_tp_min_frac=sl_min,
            sl_tp_max_frac=sl_max,
            tp_min_frac=tp_min,
            tp_max_frac=tp_max,
        ),
    )


def _legacy_dreamer_manifest() -> Path:
    """The committed 3-action dreamer artifact (rules mode by default)."""
    primary = (
        REPO
        / "artifacts"
        / "models"
        / "dreamer_20261003T230304213562Z"
        / "manifest.json"
    )
    if primary.exists():
        return primary
    return (
        REPO
        / "artifacts"
        / "models"
        / "dreamer_20261003T230535411560Z"
        / "manifest.json"
    )


def _live_paths(cfg, tmp_path):
    return replace(
        cfg.paths,
        local_state_file=tmp_path / "bot_state.json",
        risk_state_db=tmp_path / "risk.db",
        kill_switch_path=tmp_path / "KILL_SWITCH",
        log_dir=tmp_path / "logs",
    )


def _live_behavior(cfg, **kw):
    overrides = dict(
        session_filter=False,
        use_kelly=False,
        cooldown_bars_after_loss=0,
        partial_close_fraction=0.0,
        max_bars_in_trade=0,
        breakeven_at_r=100.0,
        trail_start_r=100.0,
    )
    overrides.update(kw)
    return replace(cfg.behavior, **overrides)


# --------------------------------------------------------------------------- #
# (a) Backward-compat smoke — real 3-action artifact, rules mode
# --------------------------------------------------------------------------- #


def test_backward_compat_rules_smoke_real_artifact(tmp_path, monkeypatch):
    manifest_path = _legacy_dreamer_manifest()
    assert manifest_path.exists(), "no legacy dreamer artifact manifest found"

    # Core backward-compat property: manifest without sl_tp_mode key -> rules.
    manifest = load_manifest(str(manifest_path))
    assert manifest.action_dim == 3
    assert manifest.sl_tp_mode == "rules"

    cfg = make_config()
    cfg = replace(
        cfg,
        model=ModelConfig(
            signal_source="model",
            manifest_path=str(manifest_path),
            live_history_bars=260,
            macro_csv=None,
        ),
        broker=replace(cfg.broker, **BROKER_KW),
        behavior=_live_behavior(cfg),
        paths=_live_paths(cfg, tmp_path),
    )

    source = ModelSignalSource(cfg, burn_in=0)

    def scripted_act(obs):
        # Long entry with confidence 1.0 — legacy 3-action space.
        return PolicyOutput(1, np.asarray([0.0, 1.0, 0.0]), 1.0, {})

    monkeypatch.setattr(source.policy, "act", scripted_act)

    broker = MockBroker(symbol="XAUUSD", balance=EQUITY, seed=23)
    broker.connect()
    broker.set_mid(2000.0)
    trader = LiveTrader(cfg, broker, source)

    df = sample_df(n=620)  # >= drop_warmup(200) + window(64) + slack
    events = trader.on_closed_bar(df, bar_time=str(df.iloc[-1]["time"]))
    entries = [e for e in events if e["event"] == "ENTRY"]
    assert len(entries) == 1, f"expected one entry, got {events}"

    # Rules mode: order SL/TP must come from the ATR path, NOT model fractions.
    sl, tp = _last_submit_sl_tp(trader.executor)
    entry = broker.get_tick("XAUUSD")["ask"]
    atr = trader.executor.compute_atr(df)
    exp_sl, exp_tp = entry - atr * 2.0, entry + atr * 3.0
    assert abs(sl - exp_sl) < 1e-9, f"SL {sl} != ATR path {exp_sl}"
    assert abs(tp - exp_tp) < 1e-9, f"TP {tp} != ATR path {exp_tp}"
    # PositionInfo carries the same ATR-derived levels.
    pos = broker.get_positions("XAUUSD")[0]
    assert abs(float(pos.sl) - exp_sl) < 1e-9
    assert abs(float(pos.tp) - exp_tp) < 1e-9


# --------------------------------------------------------------------------- #
# (b) Model-mode end-to-end — 75-action artifact -> Decision -> order SL/TP
# --------------------------------------------------------------------------- #


def test_model_mode_end_to_end_decision_to_executor(tmp_path, monkeypatch):
    manifest_path, manifest = _make_artifact(
        tmp_path, action_dim=75, sl_tp_mode="model"
    )
    assert manifest.sl_tp_mode == "model" and manifest.action_dim == 75

    cfg = _model_cfg(manifest_path)
    source, fake = _source(cfg, monkeypatch, manifest, 75, [_policy_output(38, 75)])
    # composite idx 38 = dir 1 (long) * 25 + sl_idx 2 * 5 + tp_idx 3
    # => sl_frac = 0.02, tp_frac = 0.03
    decision = source.decide(sample_df(n=60), AccountState(0, 0.0, 0, 0.0, 1.0))
    assert decision.action == 1
    assert decision.reason == "MODEL_SIGNAL"
    assert decision.sl_frac == pytest.approx(0.02)
    assert decision.tp_frac == pytest.approx(0.03)

    ex_cfg = _executor_model_cfg()
    clock = FakeClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
    ex = build_executor(clock, ex_cfg, tmp_path, equity=EQUITY)
    ex.broker.set_mid(2000.0)
    df = sample_df()
    md = calm_md(ex, df)
    ok, reason, result = ex.execute_entry(
        decision.action,
        EQUITY,
        df,
        market_data=md,
        bar_time="2024-01-01T00:00:00",
        size_multiplier=decision.size_multiplier,
        sl_frac=decision.sl_frac,
        tp_frac=decision.tp_frac,
    )
    assert ok, reason
    entry = ex.broker.get_tick("XAUUSD")["ask"]  # buy fills at ask
    exp_sl, exp_tp = entry * (1 - 0.02), entry * (1 + 0.03)
    sl, tp = _last_submit_sl_tp(ex)
    assert abs(sl - exp_sl) < 1e-9, f"SL {sl} != expected {exp_sl}"
    assert abs(tp - exp_tp) < 1e-9, f"TP {tp} != expected {exp_tp}"
    pos = ex.broker.get_positions("XAUUSD")[0]
    assert abs(float(pos.sl) - exp_sl) < 1e-9
    assert abs(float(pos.tp) - exp_tp) < 1e-9


def test_model_mode_end_to_end_via_live_loop(tmp_path, monkeypatch):
    manifest_path, manifest = _make_artifact(
        tmp_path, action_dim=75, sl_tp_mode="model"
    )
    # Start from the signal-side model config (feature window 4, drop_warmup 0)
    # so the stub's decide() produces a real model decision, then override the
    # live-relevant slices: broker params, model-mode behavior, tmp paths.
    cfg = _model_cfg(manifest_path)
    cfg = replace(
        cfg,
        broker=replace(cfg.broker, **BROKER_KW),
        behavior=_live_behavior(cfg, sl_tp_mode="model"),
        paths=_live_paths(cfg, tmp_path),
    )

    source, fake = _source(cfg, monkeypatch, manifest, 75, [_policy_output(38, 75)])
    broker = MockBroker(symbol="XAUUSD", balance=EQUITY, seed=23)
    broker.connect()
    broker.set_mid(2000.0)
    trader = LiveTrader(cfg, broker, source)

    df = sample_df(n=60)
    events = trader.on_closed_bar(df, bar_time=str(df.iloc[-1]["time"]))
    entries = [e for e in events if e["event"] == "ENTRY"]
    assert len(entries) == 1, f"expected one entry, got {events}"
    assert entries[0]["signal"] == 1

    # The live loop threaded decision.sl_frac/tp_frac into the order.
    sl, tp = _last_submit_sl_tp(trader.executor)
    entry = broker.get_tick("XAUUSD")["ask"]
    exp_sl, exp_tp = entry * (1 - 0.02), entry * (1 + 0.03)
    assert abs(sl - exp_sl) < 1e-9, f"SL {sl} != expected {exp_sl}"
    assert abs(tp - exp_tp) < 1e-9, f"TP {tp} != expected {exp_tp}"
    pos = broker.get_positions("XAUUSD")[0]
    assert abs(float(pos.sl) - exp_sl) < 1e-9
    assert abs(float(pos.tp) - exp_tp) < 1e-9


# --------------------------------------------------------------------------- #
# (c) Startup guard — both mismatch combos raise clear ValueErrors
# --------------------------------------------------------------------------- #


def test_startup_guard_model_artifact_in_rules_mode_raises(tmp_path):
    manifest_path = _write_stub_artifact(tmp_path, sl_tp_mode="model", action_dim=75)
    cfg = load_config(_env={
        "SLTP_MODE": "rules",
        "SIGNAL_SOURCE": "model",
        "MODEL_MANIFEST": str(manifest_path),
        "REQUIRE_PROMOTED_MODEL": "false",
    })
    with pytest.raises(ValueError, match="SILENTLY DISCARD"):
        live_trade_mt5.enforce_model_promotion_gate(cfg)


def test_startup_guard_rules_artifact_in_model_mode_raises(tmp_path):
    manifest_path = _write_stub_artifact(tmp_path, sl_tp_mode="rules", action_dim=3)
    cfg = load_config(_env={
        "SLTP_MODE": "model",
        "SIGNAL_SOURCE": "model",
        "MODEL_MANIFEST": str(manifest_path),
        "REQUIRE_PROMOTED_MODEL": "false",
    })
    with pytest.raises(ValueError, match="cannot supply.*SL/TP"):
        live_trade_mt5.enforce_model_promotion_gate(cfg)


def test_startup_guard_matching_modes_pass(tmp_path):
    rules_path = _write_stub_artifact(tmp_path / "r1", sl_tp_mode="rules", action_dim=3)
    cfg_rules = load_config(_env={
        "SLTP_MODE": "rules",
        "SIGNAL_SOURCE": "model",
        "MODEL_MANIFEST": str(rules_path),
        "REQUIRE_PROMOTED_MODEL": "false",
    })
    live_trade_mt5.enforce_model_promotion_gate(cfg_rules)  # no raise

    model_path = _write_stub_artifact(tmp_path / "m1", sl_tp_mode="model", action_dim=75)
    cfg_model = load_config(_env={
        "SLTP_MODE": "model",
        "SIGNAL_SOURCE": "model",
        "MODEL_MANIFEST": str(model_path),
        "REQUIRE_PROMOTED_MODEL": "false",
    })
    live_trade_mt5.enforce_model_promotion_gate(cfg_model)  # no raise


if __name__ == "__main__":
    import pytest as pt

    sys.exit(pt.main([__file__, "-v"]))
