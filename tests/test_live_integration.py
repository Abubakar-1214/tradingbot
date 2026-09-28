from __future__ import annotations

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.config import ModelConfig, load_config
from core.model_artifacts import ModelArtifactError
from core.observation import AccountState
from live import live_trade_mt5
from live.live_trade_mt5 import LiveTrader, Mt5BarSource, SyntheticBarSource
from live.mock_broker import MockBroker
from live.model_signal import ModelSignalSource
from models.policy import PolicyOutput
from tests.helpers import EQUITY, make_config, sample_df
from train.data import load_bars
from train.train_dreamer import train as train_dreamer
from train.train_ppo import train as train_ppo


@pytest.fixture(scope="module")
def live_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("live-artifacts")
    frame = sample_df(n=620)
    data_path = root / "bars.csv"
    frame.to_csv(data_path, index=False)
    macro = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=40, freq="D"),
        "dxy_close": np.linspace(100.0, 103.0, 40),
    })
    macro_path = root / "macro.csv"
    macro.to_csv(macro_path, index=False)
    train_end = frame["time"].iloc[420].isoformat()
    ppo_dir = train_ppo(SimpleNamespace(
        data=data_path,
        macro=macro_path,
        train_end=train_end,
        window=8,
        timesteps=16,
        n_envs=1,
        n_steps=16,
        batch_size=8,
        chunk_steps=16,
        hidden_sizes=[8],
        artifact_root=root / "models",
        run_name="ppo-live",
        allow_short=False,
        seed=13,
    ))
    dreamer_dir = train_dreamer(SimpleNamespace(
        data=data_path,
        train_end=train_end,
        window=8,
        steps=2,
        prefill=8,
        batch_size=1,
        train_every=1,
        save_every=100,
        seq_len=4,
        embed_dim=8,
        hidden_dim=16,
        stoch_dim=2,
        num_categories=3,
        horizon=1,
        artifact_root=root / "models",
        run_name="dreamer-live",
        allow_short=True,
        seed=14,
    ))
    return {
        "root": root,
        "frame": frame,
        "macro": macro_path,
        "ppo": ppo_dir,
        "dreamer": dreamer_dir,
    }


def _model_cfg(manifest, macro_csv=None, allow_short=False):
    cfg = make_config()
    return replace(
        cfg,
        feature=replace(cfg.feature, window=8),
        model=ModelConfig(
            signal_source="model",
            manifest_path=manifest,
            live_history_bars=260,
            macro_csv=macro_csv,
        ),
        broker=replace(cfg.broker, allow_short=allow_short),
    )


def test_model_signal_sources_use_saved_contract_and_window(live_artifacts, monkeypatch):
    frame = live_artifacts["frame"]
    ppo_manifest = live_artifacts["ppo"] / "manifest.json"
    ppo_cfg = _model_cfg(ppo_manifest, live_artifacts["macro"])
    ppo_source = ModelSignalSource(ppo_cfg)
    assert ppo_source.decide(
        frame.iloc[:20],
        AccountState(0, 0.0, 0, 0.0, 1.0),
    ).reason == "INSUFFICIENT_HISTORY"

    observations = []
    original_act = ppo_source.policy.act

    def capture(obs):
        observations.append(np.asarray(obs).copy())
        return original_act(obs)

    monkeypatch.setattr(ppo_source.policy, "act", capture)
    decision = ppo_source.decide(
        frame,
        AccountState(0, 0.0, 0, 0.0, 1.0),
    )
    assert decision.action in (0, 1)
    assert 0.0 <= decision.confidence <= 1.0
    assert observations[-1].size == ppo_source.manifest.obs_dim
    assert any(name.startswith("macro_") for name in ppo_source.feature_names)

    monkeypatch.setattr(
        ppo_source.policy,
        "act",
        lambda obs: PolicyOutput(2, np.asarray([0.0, 1.0]), 1.0, {}),
    )
    with pytest.raises(ValueError, match="unsupported short action"):
        ppo_source.decide(
            frame,
            AccountState(0, 0.0, 0, 0.0, 1.0),
        )
    monkeypatch.setattr(
        ppo_source.policy,
        "act",
        lambda obs: PolicyOutput(3, np.asarray([0.0, 1.0]), 1.0, {}),
    )
    with pytest.raises(ValueError, match="outside action_dim"):
        ppo_source.decide(
            frame,
            AccountState(0, 0.0, 0, 0.0, 1.0),
        )

    no_macro_cfg = _model_cfg(ppo_manifest)
    with pytest.raises(ValueError, match="MACRO_CSV"):
        ModelSignalSource(no_macro_cfg)
    missing_macro_cfg = _model_cfg(
        ppo_manifest, live_artifacts["root"] / "missing-macro.csv"
    )
    with pytest.raises(ValueError, match="no usable data"):
        ModelSignalSource(missing_macro_cfg)

    dreamer_source = ModelSignalSource(
        _model_cfg(live_artifacts["dreamer"] / "manifest.json")
    )
    assert dreamer_source.manifest.action_dim == 3
    monkeypatch.setattr(
        dreamer_source.policy,
        "act",
        lambda obs: PolicyOutput(
            2,
            np.asarray([0.0, 0.0, 1.0]),
            1.0,
            {},
        ),
    )
    mapped = dreamer_source.decide(
        frame,
        AccountState(0, 0.0, 0, 0.0, 1.0),
    )
    assert mapped.action == 0
    assert mapped.info["short_mapped_to_flat"] is True


def test_model_source_rejects_hash_and_action_space_mismatches(
    live_artifacts, tmp_path
):
    source_dir = tmp_path / "bad-artifact"
    shutil.copytree(live_artifacts["ppo"], source_dir)
    manifest_path = source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract_hash"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="contract hash mismatch"):
        ModelSignalSource(_model_cfg(manifest_path, live_artifacts["macro"]))

    with pytest.raises(ValueError, match="three-action model"):
        ModelSignalSource(
            _model_cfg(
                live_artifacts["ppo"] / "manifest.json",
                live_artifacts["macro"],
                allow_short=True,
            )
        )


def test_live_model_loop_routes_orders_through_executor_and_feeds_execution(
    live_artifacts, tmp_path, monkeypatch
):
    cfg = _model_cfg(
        live_artifacts["ppo"] / "manifest.json",
        live_artifacts["macro"],
    )
    cfg = replace(
        cfg,
        behavior=replace(
            cfg.behavior,
            session_filter=False,
            use_kelly=False,
            cooldown_bars_after_loss=0,
            partial_close_fraction=0.0,
            max_bars_in_trade=0,
            breakeven_at_r=100.0,
            trail_start_r=100.0,
        ),
        paths=replace(
            cfg.paths,
            local_state_file=tmp_path / "bot_state.json",
            risk_state_db=tmp_path / "risk.db",
            kill_switch_path=tmp_path / "KILL_SWITCH",
            log_dir=tmp_path / "logs",
        ),
    )
    broker = MockBroker(symbol="XAUUSD", balance=EQUITY, seed=23)
    broker.connect()
    broker.set_mid(2000.0)
    source = ModelSignalSource(cfg, burn_in=0)
    feedback = []
    outputs = iter([
        (1, 0.99, {}),
        (0, 0.1, {}),
        (0, 0.99, {"consensus": False}),
        (0, 0.99, {}),
    ])

    def act(obs):
        action, confidence, info = next(outputs)
        probs = np.asarray([0.99, 0.01] if action == 0 else [0.01, 0.99])
        return PolicyOutput(action, probs, confidence, info)

    monkeypatch.setattr(source.policy, "act", act)
    original_observe = source.policy.observe_executed

    def observe(action):
        feedback.append(int(action))
        original_observe(action)

    monkeypatch.setattr(source.policy, "observe_executed", observe)
    trader = LiveTrader(cfg, broker, source)
    assert trader.warmup_bars == 260
    risk_calls = []
    original_check = trader.risk.check_trade

    def check_trade(*args, **kwargs):
        risk_calls.append(args[0] if args else None)
        return original_check(*args, **kwargs)

    monkeypatch.setattr(trader.risk, "check_trade", check_trade)
    bar_events = []
    account_states = []
    original_on_closed_bar = trader.on_closed_bar

    def record_events(*args, **kwargs):
        events = original_on_closed_bar(*args, **kwargs)
        bar_events.append(events)
        positions = broker.get_positions("XAUUSD", cfg.broker.magic)
        account_states.append(
            trader._account_state(positions, trader._current_equity())
        )
        return events

    monkeypatch.setattr(trader, "on_closed_bar", record_events)
    starting_balance = broker.account_info().balance
    processed = trader.run_forever(
        SyntheticBarSource(seed=9), poll_seconds=0.0, max_bars=4
    )
    assert processed == 4
    assert len(broker._order_log) == 2
    assert len(risk_calls) >= len(broker._order_log)
    assert feedback[-4:] == [1, 1, 1, 0]
    assert not any(event["event"] == "CLOSE" for event in bar_events[1])
    assert not any(event["event"] == "CLOSE" for event in bar_events[2])
    assert any(
        event.get("reason") == "MODEL_EXIT"
        for event in bar_events[3]
        if event["event"] == "CLOSE"
    )
    assert account_states[0].position == 1
    assert account_states[1].bars_in_trade == 1
    assert account_states[-1].position == 0
    assert trader.risk.trade_history[-1]["pnl"] == pytest.approx(
        broker.account_info().balance - starting_balance
    )
    saved_state = json.loads(cfg.paths.local_state_file.read_text(encoding="utf-8"))
    assert saved_state["reference_equity"] == pytest.approx(EQUITY)


def test_live_model_gate_lists_failures_before_mt5_initialization(
    live_artifacts, tmp_path, monkeypatch
):
    artifact = tmp_path / "unpromoted"
    shutil.copytree(live_artifacts["ppo"], artifact)
    evaluation_path = artifact / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["passed"] = False
    evaluation["contract_hash"] = "wrong"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    cfg = load_config(_env={
        "TRADING_MODE": "live",
        "SIGNAL_SOURCE": "model",
        "MODEL_MANIFEST": str(artifact / "manifest.json"),
        "RISK_STATE_DB": str(tmp_path / "risk.db"),
        "MT5_LOGIN": "1",
        "MT5_PASSWORD": "test",
        "MT5_SERVER": "test-server",
        "REQUIRE_PROMOTED_MODEL": "false",
    })

    class Mt5MustNotInitialize:
        def __init__(self, *args, **kwargs):
            raise AssertionError("MT5 broker initialized before the promotion gate")

    monkeypatch.setattr(live_trade_mt5, "Mt5Broker", Mt5MustNotInitialize)
    with pytest.raises(RuntimeError) as exc:
        live_trade_mt5._build_broker(cfg)
    message = str(exc.value)
    assert "evaluation passed is not true" in message
    assert "evaluation contract_hash does not match" in message
    assert "REFUSED TO START" in message

    evaluation_path.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="JSON object"):
        live_trade_mt5._build_broker(cfg)


def test_demo_gate_can_warn_when_promotion_is_not_required(
    live_artifacts, tmp_path, caplog
):
    cfg = load_config(_env={
        "TRADING_MODE": "demo",
        "SIGNAL_SOURCE": "model",
        "MODEL_MANIFEST": str(tmp_path / "missing" / "manifest.json"),
        "REQUIRE_PROMOTED_MODEL": "false",
    })
    live_trade_mt5.enforce_model_promotion_gate(cfg)
    assert "Model promotion gate warning" in caplog.text
    required_cfg = load_config(_env={
        "TRADING_MODE": "demo",
        "SIGNAL_SOURCE": "model",
        "MODEL_MANIFEST": str(tmp_path / "missing" / "manifest.json"),
        "REQUIRE_PROMOTED_MODEL": "true",
    })
    with pytest.raises(RuntimeError, match="promotion gate refused"):
        live_trade_mt5.enforce_model_promotion_gate(required_cfg)


def test_exporter_csv_shape_is_accepted_by_load_bars(tmp_path):
    exported = tmp_path / "xauusd_h1.csv"
    pd.DataFrame({
        "time": ["2024-01-01T00:00:00+00:00"],
        "open": [2000.0],
        "high": [2002.0],
        "low": [1998.0],
        "close": [2001.0],
        "volume": [123],
    }).to_csv(exported, index=False)
    bars = load_bars(exported)
    assert list(bars.columns)[:6] == [
        "time", "open", "high", "low", "close", "volume"
    ]
    assert len(bars) == 1


def test_mt5_bars_convert_server_time_to_utc():
    class FakeMT5:
        TIMEFRAME_H1 = 1

        @staticmethod
        def copy_rates_from_pos(symbol, timeframe, start_pos, count):
            return [{
                "time": 1704067200,
                "open": 2000.0,
                "high": 2001.0,
                "low": 1999.0,
                "close": 2000.5,
                "tick_volume": 12,
            }]

    bars = Mt5BarSource(
        FakeMT5(), "XAUUSD", "H1", utc_offset_hours=2.0
    ).last_closed_bars(1)
    assert bars.iloc[0]["time"] == pd.Timestamp("2023-12-31T22:00:00Z")
    assert bars.iloc[0]["volume"] == 12
