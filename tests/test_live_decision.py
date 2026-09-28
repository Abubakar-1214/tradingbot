from dataclasses import replace
from datetime import datetime, timezone

import pytest

from live.broker import PositionInfo
from live.decision_engine import Decision, DecisionEngine
from live.trade_manager import Close, ModifySL, PartialClose, TradeManager
from models.policy import PolicyOutput
from tests.helpers import make_config


def _behavior(**changes):
    return replace(make_config().behavior, **changes)


def _policy(action, confidence=0.9, **info):
    return PolicyOutput(action, [1.0], confidence, info)


def test_decision_engine_confidence_consensus_and_no_signal_hold():
    engine = DecisionEngine(_behavior(session_filter=False, use_kelly=False))
    low = engine.filter(_policy(2, 0.4), datetime.now(timezone.utc), 1, None, [])
    assert (low.action, low.reason) == (1, "LOW_CONFIDENCE")
    split = engine.filter(
        _policy(2, consensus=False),
        datetime.now(timezone.utc),
        1,
        None,
        [],
    )
    assert (split.action, split.reason) == (1, "NO_CONSENSUS")
    below_agreement = engine.filter(
        _policy(2, consensus=True, agreement=0.3),
        datetime.now(timezone.utc),
        1,
        None,
        [],
    )
    assert below_agreement.reason == "NO_CONSENSUS"
    no_signal = engine.filter(
        Decision(0, 1.0, reason="NO_SIGNAL"),
        datetime.now(timezone.utc),
        2,
        None,
        [],
    )
    assert (no_signal.action, no_signal.reason) == (2, "NO_SIGNAL")


@pytest.mark.parametrize(
    "stamp,blocked",
    [
        ("2024-01-05T20:00:00+00:00", True),
        ("2024-01-07T00:30:00+00:00", True),
        ("2024-01-08T00:30:00+00:00", True),
        ("2024-01-08T01:00:00+00:00", False),
        ("2024-01-08T21:45:00+00:00", True),
        ("2024-01-08T23:00:00+00:00", False),
    ],
)
def test_session_filter_uses_utc_windows(stamp, blocked):
    engine = DecisionEngine(_behavior(use_kelly=False))
    decision = engine.filter(
        _policy(1),
        datetime.fromisoformat(stamp),
        0,
        None,
        [],
    )
    assert (decision.reason == "SESSION_BLOCKED") is blocked
    if blocked:
        exit_decision = engine.filter(
            _policy(0),
            datetime.fromisoformat(stamp),
            1,
            None,
            [],
        )
        assert exit_decision.action == 0


def test_cooldown_blocks_entries_but_never_exits():
    engine = DecisionEngine(
        _behavior(session_filter=False, cooldown_bars_after_loss=3, use_kelly=False)
    )
    entry = engine.filter(_policy(1), datetime.now(timezone.utc), 0, 2, [])
    exit_decision = engine.filter(_policy(0), datetime.now(timezone.utc), 1, 0, [])
    assert (entry.action, entry.reason) == (0, "COOLDOWN_AFTER_LOSS")
    assert exit_decision.action == 0


def test_confidence_size_multiplier_is_bounded():
    engine = DecisionEngine(_behavior(session_filter=False, use_kelly=False))
    decision = engine.filter(_policy(1, 0.6), datetime.now(timezone.utc), 0, None, [])
    assert 0.25 <= decision.size_multiplier <= 1.0
    assert decision.size_multiplier == pytest.approx(0.25)
    full = engine.filter(_policy(1, 1.0), datetime.now(timezone.utc), 0, None, [])
    assert full.size_multiplier == 1.0


def test_kelly_minimum_no_edge_and_positive_edge_scale():
    engine = DecisionEngine(
        _behavior(session_filter=False, use_kelly=True, kelly_min_trades=30),
        risk_per_trade=0.02,
    )
    no_history = engine.filter(
        _policy(1, 1.0), datetime.now(timezone.utc), 0, None, []
    )
    assert no_history.size_multiplier == 1.0
    negative = [{"pnl": -10, "equity": 10_000} for _ in range(24)]
    negative += [{"pnl": 1, "equity": 10_000} for _ in range(6)]
    no_edge = engine.filter(
        _policy(1), datetime.now(timezone.utc), 0, None, negative
    )
    assert (no_edge.action, no_edge.size_multiplier, no_edge.reason) == (
        0,
        0.0,
        "NO_EDGE",
    )
    positive = [{"pnl": 100, "equity": 10_000} for _ in range(16)]
    positive += [{"pnl": -100, "equity": 10_000} for _ in range(14)]
    scaled = engine.filter(
        _policy(1), datetime.now(timezone.utc), 0, None, positive
    )
    assert 0.0 < scaled.size_multiplier < 1.0


def _position(side="buy", volume=0.10, sl=90.0):
    return PositionInfo(1, "XAUUSD", side, volume, 100.0, sl, None, 234000)


def test_trade_manager_breakeven_trailing_and_partial_close():
    cfg = make_config()
    manager = TradeManager(cfg)
    pos = _position()
    meta = {"initial_sl": 90.0, "partial_done": False}
    actions = manager.manage(pos, 5.0, 4, {"close": 110.0}, 1.0, meta)
    assert any(isinstance(action, ModifySL) and action.sl > 100.0 for action in actions)
    partials = [action for action in actions if isinstance(action, PartialClose)]
    assert partials == [PartialClose(1, 0.05)]
    assert not any(
        isinstance(action, PartialClose)
        for action in manager.manage(
            pos, 5.0, 5, {"close": 111.0}, 1.0,
            {"initial_sl": 90.0, "partial_done": True},
        )
    )

    already_tighter = _position(sl=115.0)
    actions = manager.manage(
        already_tighter, 5.0, 4, {"close": 116.0}, 1.0, meta
    )
    assert not any(isinstance(action, ModifySL) for action in actions)


def test_trade_manager_short_trailing_partial_floor_and_minimum_skip():
    cfg = make_config()
    manager = TradeManager(cfg)
    short = _position(side="sell", volume=0.13, sl=110.0)
    actions = manager.manage(
        short, 5.0, 5, {"close": 84.0}, 1.0,
        {"initial_sl": 110.0, "partial_done": False},
    )
    trail = next(action for action in actions if isinstance(action, ModifySL))
    partial = next(action for action in actions if isinstance(action, PartialClose))
    assert trail.sl < 110.0
    assert partial.volume == pytest.approx(0.06)

    tighter_short = _position(side="sell", volume=0.13, sl=85.0)
    actions = manager.manage(
        tighter_short, 5.0, 5, {"close": 84.0}, 1.0,
        {"initial_sl": 110.0, "partial_done": True},
    )
    assert not any(isinstance(action, ModifySL) for action in actions)

    small = _position(volume=0.01)
    actions = manager.manage(
        small, 5.0, 1, {"close": 110.0}, 1.0,
        {"initial_sl": 90.0, "partial_done": False},
    )
    assert not any(isinstance(action, PartialClose) for action in actions)


def test_trade_manager_time_stop():
    cfg = make_config()
    cfg = replace(
        cfg,
        behavior=replace(cfg.behavior, max_bars_in_trade=2),
    )
    manager = TradeManager(cfg)
    actions = manager.manage(
        _position(), 5.0, 2, {"close": 95.0}, 1.0,
        {"initial_sl": 90.0, "partial_done": False},
    )
    assert actions == [Close(1, "TIME_STOP")]
