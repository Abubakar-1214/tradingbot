"""TradeExecutor pytest: RiskSupervisor wired into EVERY order path + SL/TP on fills.

Lifts the scenario builders from scripts/verify_risk_integration.py so the pytest
suite independently proves the same acceptance criteria, plus a 3-candle full-loop
smoke test over a seeded MockBroker.

Bug fixes this cycle (audit P0-1/P0-6 wiring proof):
  * The probe test no longer reaches for ``ex.risk.now_fn`` (RiskSupervisor
    stores the injectable clock as PRIVATE ``_now_fn`` — no public ``now_fn``
    attribute).  The test keeps the ``FakeClock`` instance it injects into the
    supervisor and advances THAT; the assertion still GENUINELY verifies
    ``check_trade`` is invoked on every order path (entry, scale, modify, close)
    via a counting wrapper around ``self.risk.check_trade``.
  * The smoke test slices >= 40 bars (ATR needs len(df) >= 15; 10 bars returned
    ATR=0 -> ``ATR_UNAVAILABLE``) and exercises a full order cycle end-to-end:
    risk-approved entry with SL/TP attached -> benign bar (no stop/TP breach)
    -> manual close -> local state persisted.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from live.broker import OrderRequest, PositionInfo
from live.mock_broker import MockBroker
from live.trade_executor import TradeExecutor
from models.risk_supervisor import RiskSupervisor
from tests.helpers import (
    EQUITY,
    FakeClock,
    SmallSizer,
    build_executor,
    calm_md,
    make_config,
    sample_df,
)


class ProbeExecutor(TradeExecutor):
    """TradeExecutor with a check_trade invocation counter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_calls: int = 0
        self._wire_probe()

    def _wire_probe(self) -> None:
        original_check = self.risk.check_trade

        def counting_check(signal, size, state, market_data):
            self.check_calls += 1
            return original_check(signal, size, state, market_data)

        self.risk.check_trade = counting_check  # type: ignore[method-assign]


def _clock(start="2024-01-08T09:00:00") -> FakeClock:
    return FakeClock(datetime.fromisoformat(start).replace(tzinfo=timezone.utc))


def _probe_executor(tmp_path):
    """ProbeExecutor on a fresh seeded MockBroker + temp SQLite risk state.

    Returns ``(executor, clock)`` — the clock is the SAME instance injected into
    the RiskSupervisor (stored as private ``_now_fn``); tests advance it directly
    instead of reaching into the supervisor's internals.
    """
    clock = _clock()
    cfg = make_config()
    broker = MockBroker(symbol="XAUUSD", balance=EQUITY, seed=42)
    broker.connect()
    broker.set_mid(2000.0)
    risk = RiskSupervisor(config=cfg.risk, db_path=tmp_path / "risk.db", now_fn=clock)
    risk.initial_equity = EQUITY
    risk.current_equity = EQUITY
    risk.daily_start_equity = EQUITY
    risk.peak_equity = EQUITY
    ex = ProbeExecutor(broker, risk, cfg, sizer=SmallSizer(),
                       local_state_file=tmp_path / "bot_state.json")
    return ex, clock


def test_all_order_paths_call_check_trade(tmp_path) -> None:
    ex, clock = _probe_executor(tmp_path)
    df = sample_df()
    ok, reason, res = ex.execute_entry(1, equity=EQUITY, df=df,
                                       market_data=calm_md(ex, df),
                                       bar_time="2024-01-08T10:00:00")
    assert ok, reason
    assert res is not None and res.ticket is not None
    assert ex.check_calls == 1

    clock.advance(minutes=10)
    ok2, reason2, _ = ex.execute_scale(1, equity=EQUITY, df=df,
                                       market_data=calm_md(ex, df),
                                       bar_time="2024-01-08T10:10:00")
    assert ok2, reason2
    assert ex.check_calls == 2

    ticket = next(iter(ex._positions))
    ok3, reason3, _ = ex.modify_sl_tp(ticket, sl=1980.0, tp=2040.0)
    assert ok3, reason3
    assert ex.check_calls == 3

    ok4, reason4, _ = ex.execute_close(ticket)
    assert ok4, reason4
    assert ex.check_calls == 4


def test_normal_entry_approved_and_filled(tmp_path) -> None:
    ex = build_executor(_clock(), make_config(), tmp_path)
    df = sample_df()
    ok, reason, res = ex.execute_entry(1, equity=EQUITY, df=df,
                                       market_data=calm_md(ex, df),
                                       bar_time="2024-01-08T10:00:00")
    assert ok, reason
    assert res is not None and res.ok and res.ticket is not None
    positions = ex.broker.get_positions()
    assert len(positions) == 1
    assert positions[0].sl is not None and positions[0].tp is not None


def test_consecutive_losses_block_entry(tmp_path) -> None:
    clock = _clock()
    ex = build_executor(clock, make_config(), tmp_path)
    df = sample_df()
    for i in range(5):
        ex.risk.update_state(pnl=-300.0, equity=EQUITY - 300.0 * (i + 1),
                             is_win=False)
        clock.advance(minutes=10)
    ok, reason, _ = ex.execute_entry(1, equity=ex.risk.current_equity, df=df,
                                     market_data=calm_md(ex, df),
                                     bar_time="2024-01-08T11:00:00")
    assert not ok and "TOO_MANY_LOSSES" in reason


def test_daily_loss_breaker_blocks_entry(tmp_path) -> None:
    ex = build_executor(_clock(), make_config(), tmp_path)
    ex.risk.update_state(pnl=-7_000.0, equity=93_000.0, is_win=False)
    ok, reason, _ = ex.execute_entry(1, equity=93_000.0, df=sample_df(),
                                     market_data=calm_md(ex, sample_df()),
                                     bar_time="2024-01-08T12:00:00")
    assert not ok and "CIRCUIT_BREAKER" in reason


def test_drawdown_breaker_blocks_entry(tmp_path) -> None:
    ex = build_executor(_clock(), make_config(), tmp_path)
    ex.risk.update_state(pnl=+2_500.0, equity=102_500.0, is_win=True)
    ok, reason, _ = ex.execute_entry(1, equity=82_000.0, df=sample_df(),
                                     market_data=calm_md(ex, sample_df()),
                                     bar_time="2024-01-08T13:00:00")
    assert not ok and "MAX_DRAWDOWN" in reason


def test_cooldown_blocks_immediate_retrade(tmp_path) -> None:
    clock = _clock()
    ex = build_executor(clock, make_config(), tmp_path)
    df = sample_df()
    ok0, _, _ = ex.execute_entry(1, equity=EQUITY, df=df,
                                 market_data=calm_md(ex, df),
                                 bar_time="2024-01-08T14:00:00")
    assert ok0
    ex.risk.update_state(pnl=+150.0, equity=EQUITY, is_win=True)
    ok, reason, _ = ex.execute_entry(1, equity=EQUITY, df=df,
                                     market_data=calm_md(ex, df),
                                     bar_time="2024-01-08T14:01:00")
    assert not ok and "COOLDOWN" in reason


def test_correlation_guard_blocks_long_when_enabled(tmp_path) -> None:
    ex = build_executor(_clock(),
                        make_config(correlation_guard_enabled=True,
                                    correlation_block_long_on_up=0.01),
                        tmp_path)
    md = calm_md(ex, sample_df())
    md["dxy_momentum"] = 0.02
    ok, reason, _ = ex.execute_entry(1, equity=EQUITY, df=sample_df(),
                                     market_data=md,
                                     bar_time="2024-01-08T15:00:00")
    assert not ok and "CORRELATION_GUARD" in reason


def test_short_disabled_by_default(tmp_path) -> None:
    ex = build_executor(_clock(), make_config(), tmp_path)
    ok, reason, _ = ex.execute_entry(2, equity=EQUITY, df=sample_df(),
                                     market_data=calm_md(ex, sample_df()),
                                     bar_time="2024-01-08T16:00:00")
    assert not ok and "SHORT_DISABLED" in reason


def test_short_enabled_when_allow_short(tmp_path) -> None:
    cfg = make_config()
    cfg = replace(cfg, broker=replace(cfg.broker, allow_short=True))
    ex = build_executor(_clock(), cfg, tmp_path)
    ok, reason, res = ex.execute_entry(2, equity=EQUITY, df=sample_df(),
                                       market_data=calm_md(ex, sample_df()),
                                       bar_time="2024-01-08T16:00:00")
    assert ok, reason
    assert res is not None and res.ok
    positions = ex.broker.get_positions()
    assert positions[0].side == "sell"
    assert positions[0].sl is not None and positions[0].tp is not None


def test_every_filled_order_carries_sl_tp(tmp_path) -> None:
    clock = _clock()
    ex = build_executor(clock, make_config(), tmp_path)
    df = sample_df()
    ok_a, _, ra = ex.execute_entry(1, equity=EQUITY, df=df,
                                   market_data=calm_md(ex, df),
                                   bar_time="2024-01-08T10:00:00")
    clock.advance(minutes=10)
    ok_b, _, rb = ex.execute_entry(1, equity=EQUITY, df=df,
                                   market_data=calm_md(ex, df),
                                   bar_time="2024-01-08T10:10:00")
    assert ok_a and ok_b and ra.ok and rb.ok
    positions = ex.broker.get_positions()
    assert len(positions) >= 2
    assert all(p.sl is not None for p in positions)
    assert all(p.tp is not None for p in positions)
    clock.advance(minutes=10)
    ok_s, _, rs = ex.execute_scale(1, equity=EQUITY, df=df,
                                   market_data=calm_md(ex, df),
                                   bar_time="2024-01-08T10:20:00")
    assert ok_s and rs.ok
    positions2 = ex.broker.get_positions()
    assert all(p.sl is not None for p in positions2)
    assert all(p.tp is not None for p in positions2)


def test_startup_reconcile_halts_on_drift(tmp_path) -> None:
    ex = build_executor(_clock(), make_config(), tmp_path)
    rec = ex.startup_reconcile()
    assert rec.ok
    assert not ex.halted
    ex.broker.submit_order(OrderRequest(symbol="XAUUSD", side="buy", volume=0.1,
                                        sl=1990.0, tp=2030.0, comment="x",
                                        magic=234000, token="recon-x"))
    rec2 = ex.startup_reconcile()
    assert not rec2.ok
    assert ex.halted and "reconciliation drift" in ex.halt_reason


def test_three_candle_full_loop_smoke(tmp_path) -> None:
    """3-candle full loop: entry on closed candle 1, benign bar 2, close on 3.

    Uses >= 40 bars — ATR needs len(df) >= 15 (period 14 + 1); a 10-bar slice
    produced ATR=0 -> ATR_UNAVAILABLE.  40 bars yield a stable ATR so the entry
    is sized, SL/TP are attached, and the cycle runs end-to-end.
    """
    clock = _clock()
    ex = build_executor(clock, make_config(), tmp_path)
    df = sample_df(n=50)

    ok1, reason1, res1 = ex.execute_entry(1, equity=EQUITY, df=df.iloc[:40],
                                          market_data=calm_md(ex, df.iloc[:40]),
                                          bar_time="2024-01-08T10:00:00")
    assert ok1, reason1
    assert res1 is not None and res1.ok and res1.ticket is not None
    ticket = res1.ticket

    # Benign bar: neither SL nor TP breached (SL ~ -2 ATR, TP ~ +3 ATR).
    fills = ex.broker.advance(high=2003.0, low=1997.0, close=2001.0)
    assert fills == []
    assert len(ex.broker.get_positions()) == 1

    ok3, reason3, res3 = ex.execute_close(ticket)
    assert ok3, reason3
    assert res3.ok
    assert ex.broker.get_positions() == []
    assert (tmp_path / "bot_state.json").exists()


def test_executor_determinism(tmp_path) -> None:
    """Same seed + config -> identical approval and fill price."""
    results = []
    for i in range(2):
        clock = _clock()
        ex = build_executor(clock, make_config(), tmp_path / f"det{i}")
        ok, reason, res = ex.execute_entry(1, equity=EQUITY, df=sample_df(),
                                           market_data=calm_md(ex, sample_df()),
                                           bar_time="2024-01-08T10:00:00")
        results.append((ok, reason, res.fill_price))
    assert results[0][0] == results[1][0]
    assert results[0][2] == results[1][2]


def test_size_multiplier_scales_volume_and_rejects_below_minimum(tmp_path) -> None:
    class LargeSizer:
        def compute_position_size(self, atr, price, equity):
            return 0.08

    ex = build_executor(_clock(), make_config(), tmp_path / "scaled")
    ex.sizer = LargeSizer()
    base_fraction, base_volume = ex.fraction_and_volume(EQUITY, 2000.0, 4.0)
    half_fraction, half_volume = ex.fraction_and_volume(
        EQUITY, 2000.0, 4.0, size_multiplier=0.5
    )
    assert half_fraction == pytest.approx(base_fraction * 0.5)
    assert half_volume == pytest.approx(base_volume * 0.5)

    small = build_executor(_clock(), make_config(), tmp_path / "small")
    df = sample_df()
    ok, reason, _ = small.execute_entry(
        1,
        equity=EQUITY,
        df=df,
        market_data=calm_md(small, df),
        bar_time="2024-01-08T10:00:00",
        size_multiplier=0.25,
    )
    assert not ok and "BELOW_MIN_LOT" in reason


def test_trade_metadata_and_reference_equity_persist(tmp_path) -> None:
    clock = _clock()
    cfg = make_config()
    ex = build_executor(clock, cfg, tmp_path)
    df = sample_df()
    ok, reason, result = ex.execute_entry(
        1,
        equity=EQUITY,
        df=df,
        market_data=calm_md(ex, df),
        bar_time="2024-01-08T10:00:00",
    )
    assert ok, reason
    metadata = ex.get_trade_meta(result.ticket)
    assert metadata["initial_sl"] < ex._positions[result.ticket].open_price
    assert metadata["entry_atr"] > 0
    assert metadata["bars_held"] == 0
    assert metadata["partial_done"] is False
    assert metadata["open_bar_time"] == "2024-01-08T10:00:00"

    loaded = TradeExecutor(
        ex.broker,
        ex.risk,
        cfg,
        sizer=SmallSizer(),
        local_state_file=tmp_path / "bot_state.json",
    )
    assert loaded.get_trade_meta(result.ticket) == metadata
    assert loaded.reference_equity == pytest.approx(EQUITY)


def test_legacy_state_without_new_fields_loads(tmp_path) -> None:
    broker = MockBroker(symbol="XAUUSD", balance=EQUITY, seed=42)
    broker.connect()
    risk = RiskSupervisor(
        config=make_config().risk,
        db_path=tmp_path / "legacy-risk.db",
        now_fn=_clock(),
    )
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "last_bar_time": "2024-01-08T10:00:00",
                "positions": {"99": PositionInfo(
                    99, "XAUUSD", "buy", 0.01, 2000.0, 1990.0, 2020.0, 234000
                ).to_dict()},
            }
        ),
        encoding="utf-8",
    )
    loaded = TradeExecutor(
        broker, risk, make_config(), sizer=SmallSizer(), local_state_file=path
    )
    assert loaded.get_trade_meta(99) == {}
    assert loaded.reference_equity == pytest.approx(EQUITY)
