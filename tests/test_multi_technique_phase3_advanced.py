"""
Unit tests for AGEX AI Phase 3: Advanced Market Structure, Regimes & Full 7-Group Consensus Engine.
"""

import pandas as pd
import pytest
from core.multi_technique_phase1 import SignedTickPressureTracker
from core.multi_technique_phase2_ofi import OFICalculator, OrderBookLevel, OrderBookSnapshot
from core.multi_technique_phase3_advanced import (
    compute_market_structure,
    compute_macd,
    compute_adx,
    compute_bollinger_bands,
    KalmanPriceFilter1D,
    classify_market_regime,
    UnifiedAgexConsensusEngine,
)


def test_market_structure_bos_and_pivots():
    highs = [100.0, 102.0, 105.0, 103.0, 102.0, 107.0, 106.0, 104.0, 108.0]
    lows = [98.0, 100.0, 101.0, 100.0, 99.0, 104.0, 103.0, 101.0, 105.0]
    closes = [99.0, 101.5, 104.0, 101.0, 100.0, 106.0, 104.5, 103.0, 107.5]

    res = compute_market_structure(highs, lows, closes)
    assert res.swing_high > 0.0
    assert res.swing_low > 0.0
    assert res.pivot_p > 0.0
    assert res.donchian_high >= res.donchian_low
    assert res.fib_50 > 0.0
    assert res.bos_signal in (-1, 0, 1)


def test_rsi_completely_removed_from_consensus():
    """Verifies RSI is completely removed from prediction pipeline and feature groups."""
    engine = UnifiedAgexConsensusEngine()
    df_bars = pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 103.0],
        "high": [101.5, 102.5, 103.5, 104.5],
        "low": [99.5, 100.5, 101.5, 102.5],
        "close": [101.0, 102.0, 103.0, 104.0],
        "volume": [100, 100, 100, 100],
    })
    tracker = SignedTickPressureTracker()
    for _ in range(5):
        tracker.add_tick(103.8, 104.0)

    res = engine.analyze_full(
        symbol="XAUUSD",
        last_4_m1_bars=df_bars,
        historical_bars=df_bars,
        tick_tracker=tracker,
        ofi_calculator=None,
        current_bid=103.9,
        current_ask=104.0,
    )
    # Check that RSI is completely gone from momentum feature group
    mom_features = res.feature_groups["momentum"]
    assert "rsi" not in mom_features
    assert "momentum_consistency" in mom_features
    assert "macd_histogram" in mom_features
    assert "momentum_score" in mom_features


def test_macd_and_bollinger_bands():
    closes = [100.0 + i * 0.5 for i in range(35)]
    macd_line, macd_sig, macd_hist = compute_macd(closes)
    assert macd_line > 0

    mid, upper, lower, bandwidth = compute_bollinger_bands(closes)
    assert upper > mid > lower
    assert bandwidth > 0.0


def test_kalman_filter_and_regime():
    kf = KalmanPriceFilter1D()
    prices = [100.0, 101.0, 102.0, 103.5, 105.0]
    slope = 0.0
    for p in prices:
        _, slope = kf.update(p)

    assert slope > 0.0  # Positive slope on uptrend

    regime, score = classify_market_regime(adx=30.0, relative_range=1.0, bandwidth=0.04, kalman_slope=slope)
    assert regime == "TRENDING_BULL"
    assert score > 0


def test_unified_agex_consensus_engine():
    engine = UnifiedAgexConsensusEngine(score_threshold=0.30)
    tracker = SignedTickPressureTracker()
    for p in [4100.0, 4102.0, 4105.0, 4108.0]:
        tracker.add_tick(bid=p - 0.1, ask=p + 0.1)

    # 20 bars of clear uptrend
    rows = []
    base = 4100.0
    for i in range(25):
        p = base + i * 0.6
        rows.append({"open": p, "high": p + 0.8, "low": p - 0.2, "close": p + 0.7, "volume": 100})
    df_history = pd.DataFrame(rows)
    df_last_4 = df_history.iloc[-4:]

    # OFI Calculator with bullish depth
    calc = OFICalculator()
    s1 = OrderBookSnapshot(
        timestamp=10.0,
        bids=[OrderBookLevel(price=4110.0, volume=10.0, side="bid")],
        asks=[OrderBookLevel(price=4111.0, volume=10.0, side="ask")],
    )
    s2 = OrderBookSnapshot(
        timestamp=11.0,
        bids=[OrderBookLevel(price=4110.5, volume=30.0, side="bid")],
        asks=[OrderBookLevel(price=4111.0, volume=10.0, side="ask")],
    )
    calc.add_snapshot(s1)
    calc.add_snapshot(s2)

    res = engine.analyze_full(
        symbol="XAUUSD",
        last_4_m1_bars=df_last_4,
        historical_bars=df_history,
        tick_tracker=tracker,
        ofi_calculator=calc,
        current_bid=4115.0,
        current_ask=4115.10,
    )

    assert res.decision == "BUY"
    assert res.directional_score > 0.30
    assert res.probabilities["buy"] is not None
    assert res.probabilities["buy"] > 0.60
    assert "order_flow" in res.feature_groups
    assert "market_structure" in res.feature_groups
    assert "market_regime" in res.feature_groups

    # Verify JSON serializability
    d = res.to_dict()
    assert d["symbol"] == "XAUUSD"
    assert d["decision"] == "BUY"


def test_inter_layer_conflict_detection_blocks_trade():
    """Verifies Section 9.3 conflict detection: when OFI and Candle clash, trade is blocked."""
    engine = UnifiedAgexConsensusEngine(score_threshold=0.30, max_cost_ratio=0.55)

    # Strongly bearish candle bars (open high, close low)
    df_bearish_bars = pd.DataFrame({
        "open": [4120.0, 4118.0, 4116.0, 4114.0],
        "high": [4121.0, 4119.0, 4117.0, 4115.0],
        "low": [4117.0, 4115.0, 4113.0, 4110.0],
        "close": [4118.0, 4116.0, 4114.0, 4110.5],
        "volume": [100, 100, 100, 100],
    })

    # Fake OFI calculator that reports strong Bullish (+0.80)
    class FakeBullishOFI:
        def compute_ofi_metrics(self):
            return {
                "status": "AVAILABLE",
                "normalized_ofi": 0.85,
                "depth_imbalance": 0.50,
            }

    tracker = SignedTickPressureTracker(window_seconds=240.0)
    for p in [4110.0, 4110.2, 4110.4]:
        tracker.add_tick(p - 0.05, p + 0.05)

    res = engine.analyze_full(
        symbol="XAUUSD",
        last_4_m1_bars=df_bearish_bars,
        historical_bars=df_bearish_bars,
        tick_tracker=tracker,
        ofi_calculator=FakeBullishOFI(),
        current_bid=4110.0,
        current_ask=4110.10,
    )

    assert res.decision == "NO_TRADE"
    assert any("CONFLICT_OFI_BULLISH_VS_CANDLE_BEARISH" in r for r in res.reasons)
    assert any("BLOCKED_DUE_TO_INTER_LAYER_CONFLICT" in r for r in res.reasons)


def test_vwap_computation_with_limitation_note():
    """Verifies Section 6.7: VWAP calculation and explicit forex volume limitation note."""
    from core.multi_technique_phase3_advanced import compute_vwap
    highs = [2001.0, 2002.0, 2003.0]
    lows = [1999.0, 2000.0, 2001.0]
    closes = [2000.5, 2001.5, 2002.5]
    volumes = [100.0, 200.0, 300.0]

    vwap, dev, note = compute_vwap(highs, lows, closes, volumes)
    assert vwap > 2000.0
    assert "Forex/CFD tick-volume proxy" in note

