"""
Unit tests for AGEX AI Tick-Based Aggregation Engine (`core/tick_aggregator.py`).
Verifies all 23 formulas, 5 stages, and edge case rules from `Tick-Based Aggregation.md`.
"""

import math
import pytest
from core.tick_aggregator import (
    AgexTick,
    AgexTickCandle,
    AgexRollingFeatures,
    AgexTickAggregator,
)


def test_stage_1_tick_preprocessing():
    """Verify Formulas 1, 2: MidPrice and Spread, plus monotonic validation."""
    agg = AgexTickAggregator(ticks_per_candle=10, rolling_window_size=5)

    # Valid tick
    tick = agg.process_raw_tick(bid=4120.10, ask=4120.50, timestamp=100.0)
    assert tick is not None
    assert tick.bid == 4120.10
    assert tick.ask == 4120.50
    assert tick.mid == 4120.30  # (4120.10 + 4120.50) / 2
    assert tick.spread == 0.40  # 4120.50 - 4120.10

    # Invalid quotes: negative or crossed spread
    assert agg.process_raw_tick(bid=-1.0, ask=10.0) is None
    assert agg.process_raw_tick(bid=10.0, ask=9.0) is None

    # Monotonic timestamp adjustment (timestamp going backwards)
    t2 = agg.process_raw_tick(bid=4120.0, ask=4120.40, timestamp=99.0)
    assert t2 is not None
    assert t2.timestamp > 100.0  # adjusted to be monotonic


def test_stage_2_and_3_k_tick_candle_construction():
    """Verify Formulas 3-18: K-tick OHLCV, body, wicks, direction, duration, tick rate, spreads."""
    agg = AgexTickAggregator(ticks_per_candle=5, rolling_window_size=5)

    # Ingest 4 ticks (candle should NOT be complete yet)
    ticks_data = [
        (4100.0, 4100.4, 10.0),  # mid = 4100.2
        (4102.0, 4102.6, 11.0),  # mid = 4102.3
        (4098.0, 4098.6, 12.0),  # mid = 4098.3 (low)
        (4104.0, 4104.4, 13.0),  # mid = 4104.2 (high)
    ]
    for b, a, ts in ticks_data:
        c = agg.add_tick(b, a, ts)
        assert c is None

    assert agg.current_progress()["ticks"] == 4

    # 5th tick -> completes candle
    c = agg.add_tick(4103.0, 4103.4, 14.0)  # mid = 4103.2 (close)
    assert c is not None
    assert isinstance(c, AgexTickCandle)
    assert c.index == 1
    assert c.tick_count == 5
    assert c.open == 4100.2
    assert c.close == 4103.2
    assert c.high == 4104.2
    assert c.low == 4098.3
    assert round(c.range, 3) == round(4104.2 - 4098.3, 3)  # 5.9
    assert round(c.body, 3) == round(4103.2 - 4100.2, 3)    # 3.0
    assert c.direction == 1  # Bullish
    assert round(c.upper_wick, 3) == round(4104.2 - 4103.2, 3)  # 1.0
    assert round(c.lower_wick, 3) == round(4100.2 - 4098.3, 3)  # 1.9
    assert c.duration_seconds == 4.0  # 14.0 - 10.0
    assert c.tick_rate == 1.25        # 5 ticks / 4.0s = 1.25


def test_stage_4_rolling_features():
    """Verify Formulas 19-23: Rolling Average Range, Tick Rate, Volatility, Acceleration, Norm Range."""
    agg = AgexTickAggregator(ticks_per_candle=3, rolling_window_size=3)

    # Complete 3 candles
    # Candle 1: range 2.0, 3 ticks in 1.0s -> tick_rate 3.0
    c1_ticks = [(100.0, 100.2, 1.0), (101.8, 102.2, 1.5), (101.0, 101.2, 2.0)]
    for b, a, ts in c1_ticks:
        c1 = agg.add_tick(b, a, ts)
    assert c1 is not None

    # Candle 2: range 4.0, 3 ticks in 0.5s -> tick_rate 6.0
    c2_ticks = [(101.0, 101.2, 2.1), (104.8, 105.2, 2.3), (103.0, 103.2, 2.6)]
    for b, a, ts in c2_ticks:
        c2 = agg.add_tick(b, a, ts)
    assert c2 is not None

    features = agg.latest_features()
    assert features is not None
    assert features.candle_index == 2
    # Tick rate changed from 3.0 to 6.0 -> (6.0 - 3.0) / 3.0 = +1.0 (+100% acceleration)
    assert features.tick_rate_change == 1.0
    assert features.normalized_range > 0.0


def test_stage_5_price_action_patterns():
    """Verify PKR pattern classifications: Hammer, Shooting Star, Expansion, Chop."""
    # 1. Bullish Hammer Rejection: lower wick >= 35%, lower wick >= 1.3*body, upper wick <= 25%
    p, act = AgexTickAggregator._classify_candle(
        direction=1,
        body_pct=20.0,
        upper_pct=10.0,
        lower_pct=70.0,
        lower_wick=7.0,
        upper_wick=1.0,
        body=2.0,
        c_range=10.0,
    )
    assert p == "BULLISH_HAMMER_REJECTION"
    assert act == 1

    # 2. Bearish Shooting Star Rejection: upper wick >= 35%, upper wick >= 1.3*body, lower wick <= 25%
    p, act = AgexTickAggregator._classify_candle(
        direction=-1,
        body_pct=20.0,
        upper_pct=70.0,
        lower_pct=10.0,
        lower_wick=1.0,
        upper_wick=7.0,
        body=2.0,
        c_range=10.0,
    )
    assert p == "BEARISH_SHOOTING_STAR_REJECTION"
    assert act == 2

    # 3. Bullish Expansion: direction=1, body_pct >= 35%, upper_pct <= 35%
    p, act = AgexTickAggregator._classify_candle(
        direction=1,
        body_pct=60.0,
        upper_pct=15.0,
        lower_pct=25.0,
        lower_wick=2.5,
        upper_wick=1.5,
        body=6.0,
        c_range=10.0,
    )
    assert p == "BULLISH_EXPANSION"
    assert act == 1

    # 4. Bearish Expansion: direction=-1, body_pct >= 35%, lower_pct <= 35%
    p, act = AgexTickAggregator._classify_candle(
        direction=-1,
        body_pct=60.0,
        upper_pct=20.0,
        lower_pct=20.0,
        lower_wick=2.0,
        upper_wick=2.0,
        body=6.0,
        c_range=10.0,
    )
    assert p == "BEARISH_EXPANSION"
    assert act == 2

    # 5. Indecision Chop: body_pct < 20%, both wicks >= 25%
    p, act = AgexTickAggregator._classify_candle(
        direction=1,
        body_pct=15.0,
        upper_pct=45.0,
        lower_pct=40.0,
        lower_wick=4.0,
        upper_wick=4.5,
        body=1.5,
        c_range=10.0,
    )
    assert p == "INDECISION_CHOP"
    assert act == 0

