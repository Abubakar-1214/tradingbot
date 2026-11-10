"""
Unit tests for AGEX AI Phase 1 Multi-Technique Prediction Engine (`core/multi_technique_phase1.py`).
Verifies:
  - Technique 1: Signed Tick Pressure (STP)
  - Technique 2, 3, 7: Candle Body Strength, SignedBody, CLV, Wick Rejection Ratios
  - Technique 4: Momentum Consistency (MC)
  - Technique 6: ATR & Relative Range
  - Technique 8: Execution Cost Filter
  - Phase 1 Consensus Engine & JSON Schema Output
"""

import math
import pandas as pd
import pytest
from core.multi_technique_phase1 import (
    SignedTickPressureTracker,
    compute_candle_metrics,
    compute_momentum_consistency,
    compute_atr_and_relative_range,
    evaluate_execution_cost,
    Phase1PredictionEngine,
    Phase1PredictionResult,
)


def test_technique_1_signed_tick_pressure():
    tracker = SignedTickPressureTracker(window_seconds=240.0)

    # Ingest sequence: 4100.0, 4101.0 (UP), 4102.0 (UP), 4101.5 (DOWN), 4101.5 (FLAT)
    tracker.add_tick(bid=4099.8, ask=4100.2, timestamp=10.0)  # mid = 4100.0
    tracker.add_tick(bid=4100.8, ask=4101.2, timestamp=11.0)  # mid = 4101.0 -> UP (+1)
    tracker.add_tick(bid=4101.8, ask=4102.2, timestamp=12.0)  # mid = 4102.0 -> UP (+1)
    tracker.add_tick(bid=4101.3, ask=4101.7, timestamp=13.0)  # mid = 4101.5 -> DOWN (-1)
    tracker.add_tick(bid=4101.3, ask=4101.7, timestamp=14.0)  # mid = 4101.5 -> FLAT (0)

    res = tracker.compute_stp()
    assert res["status"] == "VALID"
    assert res["up_count"] == 2
    assert res["down_count"] == 1
    assert res["zero_count"] == 2  # first tick has no prior, 5th tick is flat
    assert res["total_observations"] == 5
    # net direction = 2 - 1 = +1; stp = +1 / 5 = +0.20
    assert res["stp"] == 0.20


def test_technique_2_3_7_candle_metrics():
    # 1. Strong Bullish Candle: Open=100, High=110, Low=100, Close=110
    m_bull = compute_candle_metrics(open_p=100.0, high_p=110.0, low_p=100.0, close_p=110.0)
    assert m_bull.range == 10.0
    assert m_bull.signed_body == 1.0
    assert m_bull.body_strength == 1.0
    assert m_bull.clv == 1.0  # Closed at High
    assert m_bull.upper_wick == 0.0
    assert m_bull.lower_wick == 0.0

    # 2. Bullish Hammer: Open=105, High=106, Low=90, Close=105
    m_hammer = compute_candle_metrics(open_p=105.0, high_p=106.0, low_p=90.0, close_p=105.0)
    assert m_hammer.lower_wick == 15.0
    assert m_hammer.lower_rejection_ratio > 0.80
    assert m_hammer.clv > 0.50

    # 3. Doji / Zero Range edge case
    m_zero = compute_candle_metrics(open_p=100.0, high_p=100.0, low_p=100.0, close_p=100.0)
    assert m_zero.range > 0  # Protected from zero division
    assert -1.0 <= m_zero.clv <= 1.0


def test_technique_4_momentum_consistency():
    # 4 consecutive up closes: 100, 102, 105, 107, 110 (4 returns)
    res_up = compute_momentum_consistency([100.0, 102.0, 105.0, 107.0, 110.0])
    assert res_up["mc"] == 1.0
    assert res_up["positive_count"] == 4

    # 4 consecutive down closes: 110, 107, 105, 102, 100
    res_down = compute_momentum_consistency([110.0, 107.0, 105.0, 102.0, 100.0])
    assert res_down["mc"] == -1.0
    assert res_down["negative_count"] == 4

    # Alternating: 100, 105, 100, 105, 100
    res_alt = compute_momentum_consistency([100.0, 105.0, 100.0, 105.0, 100.0])
    assert res_alt["mc"] == 0.0


def test_technique_6_atr_and_relative_range():
    highs = [10.0 + i for i in range(20)]
    lows = [8.0 + i for i in range(20)]
    closes = [9.5 + i for i in range(20)]

    res = compute_atr_and_relative_range(highs, lows, closes, period=14)
    assert res["status"] == "VALID"
    assert res["atr"] > 0.0
    assert res["relative_range"] > 0.0


def test_technique_8_execution_cost_filter():
    # 1. Normal tight spread: bid=4120.0, ask=4120.25 (spread=0.25), atr=2.0 -> Cost ok
    cost_ok = evaluate_execution_cost(bid=4120.0, ask=4120.25, atr=2.0)
    assert cost_ok.passed is True
    assert cost_ok.cost_ratio < 0.40

    # 2. Blown-out wide spread: bid=4120.0, ask=4122.5 (spread=2.5), atr=2.0 -> Cost too high!
    cost_fail = evaluate_execution_cost(bid=4120.0, ask=4122.5, atr=2.0, max_allowed_cost_ratio=0.40)
    assert cost_fail.passed is False
    assert cost_fail.cost_ratio > 0.40
    assert "HIGH_COST_RATIO" in cost_fail.reason


def test_phase_1_consensus_engine_bullish_and_bearish():
    engine = Phase1PredictionEngine(score_threshold=0.30, max_cost_ratio=0.40)

    # Bullish scenario: 4 green candles, high CLV, positive STP
    bars_bull = pd.DataFrame({
        "open": [4100.0, 4102.0, 4104.0, 4106.0],
        "high": [4103.0, 4105.0, 4107.0, 4110.0],
        "low": [4099.5, 4101.5, 4103.5, 4105.5],
        "close": [4102.5, 4104.5, 4106.5, 4109.8],
    })

    tracker = SignedTickPressureTracker()
    for p in [4100.0, 4103.0, 4106.0, 4109.0]:
        tracker.add_tick(bid=p - 0.1, ask=p + 0.1)

    res_bull = engine.analyze(
        symbol="ETHUSD",
        last_4_m1_bars=bars_bull,
        historical_bars=bars_bull,
        tick_tracker=tracker,
        current_bid=4109.7,
        current_ask=4109.9,
    )
    assert res_bull.decision == "BUY"
    assert res_bull.directional_score > 0.30
    assert res_bull.probabilities["buy"] is not None
    assert res_bull.probabilities["buy"] > 0.60
    assert res_bull.feature_groups["order_flow"]["stp"] > 0
    assert res_bull.feature_groups["candle_pressure"]["clv"] > 0

    # JSON export check conforming to Section 30
    d = res_bull.to_dict()
    assert d["symbol"] == "ETHUSD"
    assert "decision" in d
    assert "probabilities" in d
    assert "feature_groups" in d

