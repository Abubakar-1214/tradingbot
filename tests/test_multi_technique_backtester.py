"""
Unit tests for AGEX AI Phase 6 Historical Backtesting & Validation Engine.
Verifies:
1. Metric computation formulas (Win Rate, Expectancy, Profit Factor, Drawdown).
2. Realistic cost modeling (Spread, Slippage, Commission).
3. 5-minute cycle grouping and causal isolation (no future data leakage).
4. All versions (Baseline to Version H) execution.
5. Walk-forward in-sample vs out-of-sample partitioning.
"""

from datetime import datetime, timedelta
import pandas as pd
import pytest

from core.multi_technique_backtester import (
    AgexMultiTechniqueBacktester,
    TradeRecord,
    compute_metrics_from_trades,
    determine_market_session,
)


def create_synthetic_m1_candles(num_minutes: int = 100, trend: float = 0.1) -> pd.DataFrame:
    """Creates a deterministic synthetic M1 dataset for testing."""
    records = []
    base_time = datetime(2026, 11, 2, 0, 0, 0)  # Asian session start
    price = 2000.0

    for i in range(num_minutes):
        bar_time = base_time + timedelta(minutes=i)
        bar_open = price
        # Add micro fluctuation + trend
        fluct = 0.2 if (i % 2 == 0) else -0.15
        bar_close = bar_open + fluct + trend
        bar_high = max(bar_open, bar_close) + 0.1
        bar_low = min(bar_open, bar_close) - 0.1

        records.append({
            "time": bar_time,
            "open": bar_open,
            "high": bar_high,
            "low": bar_low,
            "close": bar_close,
            "tick_volume": 100 + (i % 10) * 10,
            "spread": 240.0,
        })
        price = bar_close

    return pd.DataFrame(records)


def test_metric_computations_known_values():
    """Verifies Win Rate, Profit Factor, Expectancy, and Drawdown formulas."""
    trades = [
        TradeRecord(
            trade_id=1,
            cycle_time=datetime(2026, 11, 2, 8, 5),
            decision="BUY",
            entry_price=2000.0,
            exit_price=2002.0,
            spread=0.24,
            slippage=0.02,
            commission=0.03,
            gross_pnl=2.0,
            net_pnl=1.97,
            is_win=True,
            session="LONDON",
            regime="NORMAL",
        ),
        TradeRecord(
            trade_id=2,
            cycle_time=datetime(2026, 11, 2, 8, 10),
            decision="SELL",
            entry_price=2002.0,
            exit_price=2003.0,
            spread=0.24,
            slippage=0.02,
            commission=0.03,
            gross_pnl=-1.0,
            net_pnl=-1.03,
            is_win=False,
            session="LONDON",
            regime="NORMAL",
        ),
    ]

    metrics = compute_metrics_from_trades(
        trades=trades,
        total_cycles=5,
        version_name="TEST_VERSION",
        description="Test description",
        initial_balance=100.0,
    )

    assert metrics.total_trades == 2
    assert metrics.total_candles == 5
    assert metrics.no_trade_count == 3
    assert metrics.no_trade_pct == 60.0  # 3/5 = 60%
    assert metrics.wins == 1
    assert metrics.losses == 1
    assert metrics.win_rate == 50.0

    # Net profit: 1.97 - 1.03 = +0.94
    assert pytest.approx(metrics.net_profit, 0.01) == 0.94
    # Profit factor: 1.97 / 1.03 = ~1.91
    assert pytest.approx(metrics.profit_factor, 0.05) == 1.91
    # Expectancy: (0.5 * 1.97) - (0.5 * 1.03) = 0.47
    assert pytest.approx(metrics.expectancy, 0.02) == 0.47
    # Max DD: After trade 1 equity is 101.97. After trade 2 equity is 100.94. DD = 1.03
    assert pytest.approx(metrics.max_drawdown_usd, 0.02) == 1.03


def test_market_session_determination():
    """Verifies Asian, London, and NY session classification."""
    asian = datetime(2026, 11, 2, 4, 30)
    london = datetime(2026, 11, 2, 10, 0)
    ny = datetime(2026, 11, 2, 16, 15)
    off = datetime(2026, 11, 2, 22, 0)

    assert determine_market_session(asian) == "ASIAN"
    assert determine_market_session(london) == "LONDON"
    assert determine_market_session(ny) == "NEW_YORK"
    assert determine_market_session(off) == "OFF_HOURS"


def test_5min_cycles_causal_separation():
    """Ensures 4 observation bars and 1 target 5th bar have strict causal boundaries."""
    df_m1 = create_synthetic_m1_candles(num_minutes=35)
    tester = AgexMultiTechniqueBacktester()
    cycles = tester.group_m1_into_5min_cycles(df_m1)

    assert len(cycles) > 0
    for obs_4, target_5 in cycles:
        assert len(obs_4) == 4
        assert obs_4["time"].iloc[-1] < target_5["time"]
        # Observation bars are minutes 0, 1, 2, 3
        # Target is minute 4
        m_obs = [t.minute % 5 for t in obs_4["time"]]
        assert m_obs == [0, 1, 2, 3]
        assert target_5["time"].minute % 5 == 4


def test_simulate_trade_execution_costs():
    """Verifies that realistic spread, slippage, and commission are deducted."""
    tester = AgexMultiTechniqueBacktester(
        symbol="XAUUSDm",
        contract_size=100.0,
        lot_size=0.01,
        fixed_spread_usd=0.24,
        slippage_usd=0.02,
        commission_usd=0.03,
    )

    bar = pd.Series({
        "open": 2000.0,
        "high": 2005.0,
        "low": 1999.0,
        "close": 2002.0,
        "spread": 240.0,  # 0.24 USD
    })

    # BUY trade
    # Entry: open (2000.0) + half_spread (0.12) + slippage (0.02) = 2000.14
    # Exit: close (2002.0) - half_spread (0.12) = 2001.88
    # Price change = 2001.88 - 2000.14 = +1.74
    # Gross PnL on 0.01 lot = 1.74
    # Net PnL = 1.74 - commission (0.03) = 1.71
    trade = tester.simulate_trade_execution("BUY", bar, datetime(2026, 11, 2, 8, 4), 1)
    assert trade is not None
    assert trade.decision == "BUY"
    assert pytest.approx(trade.entry_price, 0.01) == 2000.14
    assert pytest.approx(trade.exit_price, 0.01) == 2001.88
    assert pytest.approx(trade.gross_pnl, 0.02) == 1.74
    assert pytest.approx(trade.net_pnl, 0.02) == 1.71
    assert trade.is_win is True


def test_backtest_all_versions_run_cleanly():
    """Verifies that all versions execute without runtime errors and produce metrics."""
    df_m1 = create_synthetic_m1_candles(num_minutes=150, trend=0.08)
    tester = AgexMultiTechniqueBacktester(lot_size=0.01)

    versions = [
        "BASELINE",
        "VERSION_A",
        "VERSION_B",
        "VERSION_C",
        "VERSION_D",
        "VERSION_E",
        "VERSION_F",
        "VERSION_G",
        "VERSION_H",
    ]

    results = tester.run_backtest_all_versions(df_m1, versions=versions)
    assert len(results) == len(versions)

    for v in versions:
        m = results[v]
        assert m.version_name == v
        assert m.total_candles == 30  # 150 minutes / 5 = 30 cycles
        assert 0.0 <= m.win_rate <= 100.0
        assert 0.0 <= m.no_trade_pct <= 100.0
        assert m.total_trades + m.no_trade_count == m.total_candles

    md_table = tester.generate_comparison_markdown_table(results)
    assert "| Version |" in md_table
    assert "**BASELINE**" in md_table
    assert "**VERSION_H**" in md_table


def test_walk_forward_validation():
    """Verifies walk-forward in-sample and out-of-sample partitioning."""
    df_m1 = create_synthetic_m1_candles(num_minutes=150, trend=0.05)
    tester = AgexMultiTechniqueBacktester(lot_size=0.01)

    wf = tester.run_walk_forward_validation(df_m1, train_ratio=0.70, version="BASELINE")
    assert "train_cycles" in wf
    assert "test_cycles" in wf
    assert wf["train_cycles"] > 0
    assert wf["test_cycles"] > 0
    assert wf["train_cycles"] + wf["test_cycles"] == 30
    assert "train_metrics" in wf
    assert "test_metrics" in wf


def test_rolling_7_candle_cycles_grouping():
    """Verifies that rolling 7-candle grouping produces exact 7 observation bars and 1 target bar."""
    df_m1 = create_synthetic_m1_candles(num_minutes=30)
    tester = AgexMultiTechniqueBacktester()
    cycles = tester.group_m1_into_rolling_cycles(df_m1, window=7)

    assert len(cycles) == 30 - 7
    for obs_7, target_1 in cycles:
        assert len(obs_7) == 7
        assert obs_7["time"].iloc[-1] < target_1["time"]
        assert (target_1["time"] - obs_7["time"].iloc[-1]).total_seconds() == 60


def test_rolling_7_candle_backtest_with_cooldown():
    """Verifies that rolling backtest runs cleanly with window=7 and post-trade cooldown."""
    df_m1 = create_synthetic_m1_candles(num_minutes=100, trend=0.06)
    tester = AgexMultiTechniqueBacktester(lot_size=0.01)

    results = tester.run_backtest_all_versions(
        df_m1,
        versions=["BASELINE", "VERSION_A", "VERSION_H"],
        cycle_mode="ROLLING",
        rolling_window=7,
        cooldown_bars=2,
    )

    assert "BASELINE" in results
    assert "VERSION_H" in results
    vh = results["VERSION_H"]
    assert vh.total_candles == 100 - 7
    assert 0 <= vh.win_rate <= 100


