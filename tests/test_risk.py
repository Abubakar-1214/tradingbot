"""RiskSupervisor pytest: real-size approved, oversized rejected, breakers persist across restart."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from models.risk_supervisor import RiskSupervisor


def _sup(db_path, now=None, **cfg_kw):
    cfg = {
        "max_daily_loss": 0.05,
        "max_position": 0.10,
        "max_drawdown": 0.15,
        "max_consecutive_losses": 5,
        "max_trades_per_day": 20,
        "min_trade_interval_sec": 300.0,
        "vol_threshold": 3.0,
        "max_spread": 0.0005,
    }
    cfg.update(cfg_kw)
    return RiskSupervisor(config=cfg, db_path=db_path, now_fn=now)


def _md(**kw):
    d = {"volatility": 1.5, "spread": 0.0003, "is_high_impact_event": False,
         "is_event_window": False, "dxy_momentum": -0.005}
    d.update(kw)
    return d


def test_normal_trade_approved(tmp_path) -> None:
    sup = _sup(tmp_path / "risk.db")
    ok, reason = sup.check_trade(1, requested_size=0.05,
                                 state={"position": 0, "equity": 10_000.0},
                                 market_data=_md())
    assert ok
    assert reason == "APPROVED"


def test_oversized_rejected(tmp_path) -> None:
    sup = _sup(tmp_path / "risk.db")
    ok, reason = sup.check_trade(1, requested_size=0.50,
                                 state={"equity": 10_000.0}, market_data=_md())
    assert not ok
    assert "POSITION_TOO_LARGE" in reason
    # 9% under the 10% cap is accepted.
    ok, _ = sup.check_trade(1, requested_size=0.09,
                            state={"equity": 10_000.0}, market_data=_md())
    assert ok
    # Fraction > 1 is invalid.
    ok, reason = sup.check_trade(1, requested_size=1.5,
                                 state={"equity": 10_000.0}, market_data=_md())
    assert not ok and "INVALID_SIZE" in reason


def test_daily_loss_breaker_persists_across_restart(tmp_path) -> None:
    db = tmp_path / "risk.db"
    sup = _sup(db)
    sup.update_state(pnl=-700.0, equity=9_300.0, is_win=False)
    ok, reason = sup.check_trade(1, requested_size=0.05,
                                 state={"equity": 9_300.0}, market_data=_md())
    assert not ok and "CIRCUIT_BREAKER" in reason

    # Fresh process, same DB -> breaker survives restart.
    sup2 = _sup(db)
    st = sup2.get_statistics()
    assert abs(st["daily_pnl"] - (-700.0)) < 1e-9
    ok, reason = sup2.check_trade(1, requested_size=0.05,
                                  state={"equity": 9_300.0}, market_data=_md())
    assert not ok


def test_halt_survives_restart(tmp_path) -> None:
    db = tmp_path / "risk.db"
    now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    clock = [now]

    def fake_now():
        return clock[0]

    sup = _sup(db, now=fake_now)
    sup.emergency_shutdown()
    assert sup.halt_remaining_seconds() > 86400

    sup2 = _sup(db, now=fake_now)
    ok, reason = sup2.check_trade(1, requested_size=0.05,
                                  state={"equity": 10_000.0},
                                  market_data=_md())
    assert not ok and "HALTED" in reason

    # After the halt window expires the account may trade again.
    clock[0] = now + timedelta(days=366)
    sup3 = _sup(db, now=fake_now)
    ok, reason = sup3.check_trade(1, requested_size=0.05,
                                  state={"equity": 10_000.0},
                                  market_data=_md())
    assert ok


def test_consecutive_losses_breaker_at_limit(tmp_path) -> None:
    db = tmp_path / "risk.db"
    clock = [datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)]

    def fake_now():
        return clock[0]

    sup = _sup(db, now=fake_now)
    for i in range(4):
        clock[0] += timedelta(minutes=10)
        sup.update_state(pnl=-50.0, equity=9_950.0 - 50.0 * i, is_win=False)
    assert sup.consecutive_losses == 4
    # Advance past cooldown so this isolates the consecutive-loss breaker.
    clock[0] += timedelta(minutes=10)
    ok, _ = sup.check_trade(1, requested_size=0.05,
                            state={"equity": 9_750.0}, market_data=_md())
    assert ok
    sup.update_state(pnl=-50.0, equity=9_700.0, is_win=False)
    ok, reason = sup.check_trade(1, requested_size=0.05,
                                 state={"equity": 9_700.0}, market_data=_md())
    assert not ok and "TOO_MANY_LOSSES" in reason
    # A win resets the streak.
    clock[0] += timedelta(minutes=10)
    sup.update_state(pnl=100.0, equity=9_800.0, is_win=True)
    assert sup.consecutive_losses == 0


def test_correlation_guard_off_by_default(tmp_path) -> None:
    sup = _sup(tmp_path / "risk.db")
    ok, _ = sup.check_trade(1, requested_size=0.05,
                            state={"equity": 10_000.0},
                            market_data=_md(dxy_momentum=0.05))
    assert ok


def test_correlation_guard_on_blocks_long(tmp_path) -> None:
    sup = _sup(tmp_path / "risk.db", correlation_guard_enabled=True,
               correlation_asset="DXY", correlation_block_long_on_up=0.01)
    ok, reason = sup.check_trade(1, requested_size=0.05,
                                 state={"equity": 10_000.0},
                                 market_data=_md(dxy_momentum=0.05))
    assert not ok and "CORRELATION_GUARD" in reason
    ok, _ = sup.check_trade(1, requested_size=0.05,
                            state={"equity": 10_000.0},
                            market_data=_md(dxy_momentum=-0.05))
    assert ok


def test_max_trades_and_cooldown(tmp_path) -> None:
    db = tmp_path / "risk.db"
    clock = [datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)]

    def fake_now():
        return clock[0]

    sup = _sup(db, now=fake_now, max_trades_per_day=2)
    for _ in range(2):
        ok, _ = sup.check_trade(1, requested_size=0.05,
                                state={"equity": 10_000.0}, market_data=_md())
        assert ok
        sup.update_state(pnl=10.0, equity=10_010.0, is_win=True)
        clock[0] += timedelta(minutes=10)
    ok, reason = sup.check_trade(1, requested_size=0.05,
                                 state={"equity": 10_020.0}, market_data=_md())
    assert not ok and "MAX_TRADES" in reason


def test_cooldown_blocks_immediate_retrade(tmp_path) -> None:
    db = tmp_path / "risk.db"
    clock = [datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)]

    def fake_now():
        return clock[0]

    sup = _sup(db, now=fake_now)
    ok, _ = sup.check_trade(1, requested_size=0.05,
                            state={"equity": 10_000.0}, market_data=_md())
    assert ok
    sup.update_state(pnl=10.0, equity=10_010.0, is_win=True)
    ok, reason = sup.check_trade(1, requested_size=0.05,
                                 state={"equity": 10_010.0}, market_data=_md())
    assert not ok and "COOLDOWN" in reason


def test_spread_and_volatility_breakers(tmp_path) -> None:
    sup = _sup(tmp_path / "risk.db")
    ok, reason = sup.check_trade(1, requested_size=0.05,
                                 state={"equity": 10_000.0},
                                 market_data=_md(spread=0.003))
    assert not ok and "SPREAD_TOO_WIDE" in reason

    ok, reason = sup.check_trade(1, requested_size=0.05,
                                 state={"equity": 10_000.0, "position": 0},
                                 market_data=_md(volatility=9.0))
    assert not ok and "HIGH_VOLATILITY" in reason


def test_event_window_halves_size(tmp_path) -> None:
    sup = _sup(tmp_path / "risk.db")
    ok, reason = sup.check_trade(1, requested_size=0.09,
                                 state={"equity": 10_000.0},
                                 market_data=_md(is_event_window=True))
    assert not ok and "EVENT_RISK" in reason
