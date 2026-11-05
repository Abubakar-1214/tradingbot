import time
import pandas as pd
import pytest

from core.config import AppConfig, BrokerConfig, PathConfig, RiskConfig, ScalpConfig
from live.broker import PositionInfo
from live.mock_broker import MockBroker
from live.live_trade_mt5 import LiveTrader, RuleSignalSource


def _make_trader(tmp_path, target_profit=0.40, allow_loss_cut=False, allow_same_candle=True, reenter_bars=0, reenter_delay=0.01):
    cfg = AppConfig(
        broker=BrokerConfig(symbol="XAUUSD", magic=123, allow_short=True, use_sl_tp=False),
        risk=RiskConfig(allow_min_lot_override=True, min_trade_interval_sec=0.0, max_trades_per_day=500),
        paths=PathConfig(risk_state_db=tmp_path / "risk.db", local_state_file=tmp_path / "bot_state.json"),
        scalp=ScalpConfig(
            enabled=True,
            target_profit_usd=target_profit,
            max_loss_usd=0.0,
            allow_loss_cut=allow_loss_cut,
            allow_same_candle_reenter=allow_same_candle,
            reenter_bars_wait=reenter_bars,
            reenter_delay_sec=reenter_delay,
            poll_seconds=0.1,
        ),
    )
    broker = MockBroker(symbol="XAUUSD", balance=25.0)
    broker.connect()
    signal_source = RuleSignalSource(fast=5, slow=10, enter_on_trend=True)
    trader = LiveTrader(cfg, broker, signal_source, warmup_bars=20)
    trader.executor._positions.clear()
    trader.executor.trade_meta.clear()
    return trader, broker, cfg


def _make_bars(trend="down", count=30):
    base = 2000.0
    rows = []
    for i in range(count):
        if trend == "down":
            p = base - i * 0.5
        else:
            p = base + i * 0.5
        rows.append({
            "time": pd.Timestamp("2026-10-30 12:00:00") + pd.Timedelta(minutes=i),
            "open": p,
            "high": p + 0.2,
            "low": p - 0.2,
            "close": p,
            "volume": 100.0,
        })
    return pd.DataFrame(rows)


def test_intra_candle_profit_lock(tmp_path):
    trader, broker, cfg = _make_trader(tmp_path, target_profit=0.40)
    df = _make_bars(trend="down")

    pos = PositionInfo(
        ticket=101,
        symbol="XAUUSD",
        side="sell",
        volume=0.01,
        open_price=2000.0,
        sl=None,
        tp=None,
        magic=123,
        profit=0.45,  # Hits profit target!
    )
    broker._positions[101] = pos
    trader.executor._positions[101] = pos

    trader._handle_intra_candle(df, bar_time="2026-10-30 12:30:00")

    # Position should be closed to lock profit!
    assert 101 not in broker._positions
    assert 101 not in trader.executor._positions
    # After close, _bars_since_close must be reset to 0
    assert trader._bars_since_close == 0


def test_intra_candle_never_cuts_on_loss(tmp_path):
    # User requirement: Do NOT close trade under a loss!
    trader, broker, cfg = _make_trader(tmp_path, allow_loss_cut=False)
    df = _make_bars(trend="down")

    pos = PositionInfo(
        ticket=102,
        symbol="XAUUSD",
        side="sell",
        volume=0.01,
        open_price=2000.0,
        sl=None,
        tp=None,
        magic=123,
        profit=-1.50,  # In loss / pullback!
    )
    broker._positions[102] = pos
    trader.executor._positions[102] = pos

    trader._handle_intra_candle(df, bar_time="2026-10-30 12:30:00")

    # Position MUST REMAIN OPEN! Not closed!
    assert 102 in broker._positions
    assert 102 in trader.executor._positions


def test_intra_candle_skips_2_candles_when_configured(tmp_path):
    # When allow_same_candle=False and reenter_bars=2:
    trader, broker, cfg = _make_trader(tmp_path, allow_same_candle=False, reenter_bars=2, reenter_delay=0.01)
    df = _make_bars(trend="down")

    # Simulate position just closed on bar 12:20:00
    trader._bars_since_close = 0
    trader._last_close_bar_time = "2026-10-30 12:20:00"
    trader._last_scalp_close_time = time.time() - 1.0

    # Case 1: Same candle (12:20:00) -> MUST NOT ENTER!
    trader._handle_intra_candle(df, bar_time="2026-10-30 12:20:00")
    assert len(broker.get_positions()) == 0

    # Case 2: 1 candle has passed (_bars_since_close = 1) -> MUST NOT ENTER!
    trader._bars_since_close = 1
    trader._handle_intra_candle(df, bar_time="2026-10-30 12:21:00")
    assert len(broker.get_positions()) == 0

    # Case 3: 2 candles have passed (_bars_since_close = 2) -> NOW ENTER!
    trader._bars_since_close = 2
    trader._handle_intra_candle(df, bar_time="2026-10-30 12:22:00")
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].side == "sell"


def test_intra_candle_allows_same_candle_reenter(tmp_path):
    # User requirement: Watch same candle and open new trade if opportunity exists!
    trader, broker, cfg = _make_trader(tmp_path, allow_same_candle=True, reenter_bars=0, reenter_delay=0.01)
    df = _make_bars(trend="down")

    # Simulate position just closed on bar 12:20:00
    trader._bars_since_close = 0
    trader._last_close_bar_time = "2026-10-30 12:20:00"
    trader._last_scalp_close_time = time.time() - 0.05  # Past delay

    # Same candle (12:20:00) with fresh downtrend opportunity -> MUST ENTER!
    trader._handle_intra_candle(df, bar_time="2026-10-30 12:20:00")
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].side == "sell"


def test_htf_trend_filter_blocks_mismatches(tmp_path):
    # If HTF is SELL, and M1 tries to BUY -> Filter blocks it!
    trader, broker, cfg = _make_trader(tmp_path, reenter_bars=0, reenter_delay=0.01)

    class MockHtfSource:
        def last_closed_bars(self, n):
            # Creates an uptrend for HTF (Fast > Slow = 1) with enough bars for slow SMA
            return _make_bars(trend="up", count=70)

    trader.htf_source = MockHtfSource()

    # M1 bars are downtrend (SELL = 2)
    df_m1_down = _make_bars(trend="down")

    # M1 wants to SELL, but HTF is BUY -> MUST NOT ENTER!
    trader._bars_since_close = 5
    trader._handle_intra_candle(df_m1_down, bar_time="2026-10-30 12:30:00")
    assert len(broker.get_positions()) == 0

    # Now provide M1 uptrend bars (BUY = 1) -> Aligns with HTF -> MUST ENTER!
    df_m1_up = _make_bars(trend="up")
    trader._handle_intra_candle(df_m1_up, bar_time="2026-10-30 12:31:00")
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].side == "buy"
