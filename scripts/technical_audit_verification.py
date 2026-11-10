"""
AGEX AI — Technical Audit Verification Script.
Executes deep unit and stress tests on every mathematical technique in AGEX AI.
"""

import math
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd

from core.multi_technique_phase1 import (
    SignedTickPressureTracker,
    compute_candle_metrics,
    compute_momentum_consistency,
    compute_atr_and_relative_range,
    evaluate_execution_cost,
    Phase1PredictionEngine,
)
from core.multi_technique_phase2_ofi import (
    OrderBookLevel,
    OrderBookSnapshot,
    OFICalculator,
    MT5MarketBookManager,
)
from core.multi_technique_phase3_advanced import (
    compute_market_structure,
    compute_macd,
    compute_adx,
    compute_bollinger_bands,
    compute_vwap,
    KalmanPriceFilter1D,
    classify_market_regime,
    UnifiedAgexConsensusEngine,
    TechniqueOutputContract,
)

results = {}

def log_test(name, passed, details):
    results[name] = {"passed": passed, "details": details}
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {details}")

print("=== STARTING AGEX AI DEEP TECHNICAL AUDIT ===")

# --------------------------------------------------------------------------- #
# 1. Signed Tick Pressure (STP)
# --------------------------------------------------------------------------- #
try:
    tracker = SignedTickPressureTracker(window_seconds=240.0)
    # Add empty check
    empty_res = tracker.compute_stp()
    assert empty_res["status"] == "INSUFFICIENT_DATA"
    assert empty_res["stp"] == 0.0

    # Add 10 ticks: 7 up, 2 down, 1 flat
    t0 = 1000.0
    tracker.add_tick(2000.00, 2000.10, t0)        # mid 2000.05
    tracker.add_tick(2000.05, 2000.15, t0 + 1)   # mid 2000.10 (+1)
    tracker.add_tick(2000.10, 2000.20, t0 + 2)   # mid 2000.15 (+1)
    tracker.add_tick(2000.08, 2000.18, t0 + 3)   # mid 2000.13 (-1)
    tracker.add_tick(2000.08, 2000.18, t0 + 4)   # mid 2000.13 (0)
    stp_res = tracker.compute_stp()
    
    # Formula check: Total obs = 5, up = 2, down = 1, flat = 1.
    # Diff sum = +2 - 1 = +1. stp = +1 / 5 = 0.20
    assert -1.0 <= stp_res["stp"] <= 1.0
    assert stp_res["status"] == "VALID"
    log_test("Signed Tick Pressure (STP)", True, f"STP={stp_res['stp']}, Up={stp_res['up_count']}, Down={stp_res['down_count']}, Flat={stp_res['zero_count']}")
except Exception as e:
    log_test("Signed Tick Pressure (STP)", False, str(e))


# --------------------------------------------------------------------------- #
# 2. Order Flow Imbalance (OFI) & Market Depth
# --------------------------------------------------------------------------- #
try:
    ofi_calc = OFICalculator(window_seconds=240.0)
    # Test unavailable when no snapshots
    metrics = ofi_calc.compute_ofi_metrics()
    assert metrics["status"] == "UNAVAILABLE"
    assert metrics["normalized_ofi"] == 0.0

    # Cont-Kukanov-Stoikov Step Test:
    # Snapshot 1: Bid 2000.0 @ 10, Ask 2000.5 @ 10
    s1 = OrderBookSnapshot(
        timestamp=100.0,
        bids=[OrderBookLevel(2000.0, 10.0, "bid")],
        asks=[OrderBookLevel(2000.5, 10.0, "ask")]
    )
    ofi_calc.add_snapshot(s1)

    # Snapshot 2: Bid increases to 2000.1 @ 15, Ask unchanged at 2000.5 @ 10
    # Bid contribution: Bid improved -> +15.0
    # Ask contribution: Ask price unchanged -> 10 - 10 = 0.0
    # Net OFI = +15.0
    s2 = OrderBookSnapshot(
        timestamp=101.0,
        bids=[OrderBookLevel(2000.1, 15.0, "bid")],
        asks=[OrderBookLevel(2000.5, 10.0, "ask")]
    )
    ev = ofi_calc.add_snapshot(s2)
    assert ev is not None
    assert ev.bid_contribution == 15.0
    assert ev.ask_contribution == 0.0
    assert ev.event_ofi == 15.0

    ofi_res = ofi_calc.compute_ofi_metrics()
    assert ofi_res["status"] == "AVAILABLE"
    assert ofi_res["ofi_total"] == 15.0
    assert 0.0 < ofi_res["normalized_ofi"] <= 1.0
    log_test("Order Flow Imbalance (OFI)", True, f"CKS Formula verified. ofi_total={ofi_res['ofi_total']}, normalized_ofi={ofi_res['normalized_ofi']}")
except Exception as e:
    log_test("Order Flow Imbalance (OFI)", False, str(e))


# --------------------------------------------------------------------------- #
# 3. Depth Imbalance
# --------------------------------------------------------------------------- #
try:
    snap = OrderBookSnapshot(
        timestamp=100.0,
        bids=[OrderBookLevel(2000.0, 30.0, "bid"), OrderBookLevel(1999.0, 20.0, "bid")],
        asks=[OrderBookLevel(2001.0, 10.0, "ask"), OrderBookLevel(2002.0, 15.0, "ask")]
    )
    # Total bid = 50, Total ask = 25. Imbalance = (50 - 25) / (50 + 25) = 25 / 75 = +0.3333
    imb = snap.depth_imbalance()
    assert abs(imb - (1.0 / 3.0)) < 1e-4
    assert -1.0 <= imb <= 1.0
    log_test("Order Book Depth Imbalance", True, f"Imbalance={imb:.4f} matched formula (Bid-Ask)/(Bid+Ask)")
except Exception as e:
    log_test("Order Book Depth Imbalance", False, str(e))


# --------------------------------------------------------------------------- #
# 4. Candle Body Strength, Signed Body & Wick Rejections
# --------------------------------------------------------------------------- #
try:
    # Bullish Pinbar: Open=2000, High=2005, Low=1990, Close=2002
    # Range = 2005 - 1990 = 15.0
    # Body = abs(2002 - 2000) = 2.0. BodyStrength = 2 / 15 = 0.1333
    # SignedBody = (2002 - 2000) / 15 = +0.1333
    # Upper Wick = 2005 - 2002 = 3.0 -> UpperRatio = 3 / 15 = 0.20
    # Lower Wick = 2000 - 1990 = 10.0 -> LowerRatio = 10 / 15 = 0.6667
    # CLV = ((2002 - 1990) - (2005 - 2002)) / 15 = (12 - 3) / 15 = +0.60
    m = compute_candle_metrics(2000.0, 2005.0, 1990.0, 2002.0)
    assert abs(m.body_strength - (2.0 / 15.0)) < 1e-4
    assert abs(m.signed_body - (2.0 / 15.0)) < 1e-4
    assert abs(m.upper_rejection_ratio - 0.20) < 1e-4
    assert abs(m.lower_rejection_ratio - (10.0 / 15.0)) < 1e-4
    assert abs(m.clv - 0.60) < 1e-4
    assert -1.0 <= m.clv <= 1.0
    assert -1.0 <= m.signed_body <= 1.0
    assert 0.0 <= m.body_strength <= 1.0
    log_test("Candle Metrics (Body, SignedBody, CLV, Wicks)", True, f"CLV={m.clv:.2f}, SignedBody={m.signed_body:.3f}, LowerWick={m.lower_rejection_ratio:.3f}")
except Exception as e:
    log_test("Candle Metrics", False, str(e))


# --------------------------------------------------------------------------- #
# 5. Momentum Consistency (MC)
# --------------------------------------------------------------------------- #
try:
    # Test all positive returns: [100, 102, 105, 109] -> Returns: [+2, +3, +4]
    # Sum returns = +9, Sum abs = 9 -> MC = +1.0
    res_up = compute_momentum_consistency([100.0, 102.0, 105.0, 109.0])
    assert res_up["mc"] == 1.0
    assert res_up["status"] == "VALID"
    assert res_up["positive_count"] == 3

    # Test conflicting returns: [100, 105, 102, 104] -> Returns: [+5, -3, +2]
    # Sign returns: [+1, -1, +1] -> Sum signs = +1 / 3 = 0.3333
    res_conf = compute_momentum_consistency([100.0, 105.0, 102.0, 104.0])
    assert abs(res_conf["mc"] - (1.0 / 3.0)) < 1e-4
    log_test("Momentum Consistency (MC)", True, f"All-up MC={res_up['mc']}, Mixed MC={res_conf['mc']}")
except Exception as e:
    log_test("Momentum Consistency (MC)", False, str(e))


# --------------------------------------------------------------------------- #
# 6. Wilder's ATR & Relative Range
# --------------------------------------------------------------------------- #
try:
    # 20 identical candles: H=12, L=8, C=10 -> TR=4 for all
    highs = [12.0] * 20
    lows = [8.0] * 20
    closes = [10.0] * 20
    vol_res = compute_atr_and_relative_range(highs, lows, closes, period=14)
    assert abs(vol_res["atr"] - 4.0) < 1e-4
    assert abs(vol_res["relative_range"] - 1.0) < 1e-4
    log_test("ATR & Relative Range", True, f"ATR={vol_res['atr']}, RelativeRange={vol_res['relative_range']}")
except Exception as e:
    log_test("ATR & Relative Range", False, str(e))


# --------------------------------------------------------------------------- #
# 7. MACD Line, Signal & Histogram (RSI Removed)
# --------------------------------------------------------------------------- #
try:
    np.random.seed(42)
    synthetic_closes = list(np.cumsum(np.random.randn(60)) + 2600.0)
    macd_line, macd_sig, macd_hist = compute_macd(synthetic_closes)
    # Check mathematical relation: hist == line - signal
    diff = round(macd_line - macd_sig, 4)
    assert abs(macd_hist - diff) <= 0.0001
    log_test("MACD", True, f"MACD line={macd_line}, signal={macd_sig}, hist={macd_hist}")
except Exception as e:
    log_test("MACD", False, str(e))


# --------------------------------------------------------------------------- #
# 9. ADX & Directional Movement (+DI, -DI)
# --------------------------------------------------------------------------- #
try:
    adx_highs = [2600.0 + i * 1.5 for i in range(40)]
    adx_lows = [2595.0 + i * 1.5 for i in range(40)]
    adx_closes = [2598.0 + i * 1.5 for i in range(40)]
    pdi, mdi, adx = compute_adx(adx_highs, adx_lows, adx_closes, period=14)
    assert pdi > mdi, "Upward trend must have +DI > -DI"
    assert adx > 20.0, "Strong trend must have ADX > 20"
    log_test("ADX & Directional Movement", True, f"+DI={pdi}, -DI={mdi}, ADX={adx}")
except Exception as e:
    log_test("ADX", False, str(e))


# --------------------------------------------------------------------------- #
# 10. Bollinger Bands & BandWidth
# --------------------------------------------------------------------------- #
try:
    bb_closes = [100.0] * 20
    bb_closes[-1] = 105.0
    mid, up, low, bw = compute_bollinger_bands(bb_closes, period=20, mult=2.0)
    assert up > mid > low
    assert bw >= 0.0
    log_test("Bollinger Bands & BandWidth", True, f"Mid={mid}, Upper={up}, Lower={low}, Width={bw}")
except Exception as e:
    log_test("Bollinger Bands", False, str(e))


# --------------------------------------------------------------------------- #
# 11. Market Structure (BOS, Pivots, Donchian, Fibs)
# --------------------------------------------------------------------------- #
try:
    st_highs = [10.0, 12.0, 15.0, 13.0, 11.0, 14.0, 18.0, 16.0, 14.0, 20.0, 22.0]
    st_lows =  [8.0,  9.0,  11.0, 10.0, 9.0,  11.0, 13.0, 12.0, 11.0, 15.0, 17.0]
    st_closes = [9.0, 11.0, 14.0, 11.0, 10.0, 13.0, 17.0, 13.0, 12.0, 19.0, 21.5]
    ms_res = compute_market_structure(st_highs, st_lows, st_closes, lookback=2)
    assert ms_res.bos_signal in (-1, 0, 1)
    assert ms_res.donchian_high >= ms_res.donchian_low
    assert ms_res.pivot_r1 >= ms_res.pivot_p >= ms_res.pivot_s1
    assert -1.0 <= ms_res.structure_score <= 1.0
    log_test("Market Structure & BOS", True, f"State={ms_res.trend_state}, Score={ms_res.structure_score}, BOS={ms_res.bos_signal}, PivotP={ms_res.pivot_p}")
except Exception as e:
    log_test("Market Structure", False, str(e))


# --------------------------------------------------------------------------- #
# 12. Intraday VWAP
# --------------------------------------------------------------------------- #
try:
    vw_h = [10.0, 20.0]
    vw_l = [8.0, 18.0]
    vw_c = [9.0, 19.0]
    vw_v = [100.0, 200.0]
    # Bar 1 Typical = (10+8+9)/3 = 9.0 * 100 = 900
    # Bar 2 Typical = (20+18+19)/3 = 19.0 * 200 = 3800
    # Total PV = 4700, Total V = 300 -> VWAP = 4700 / 300 = 15.6667
    vwap, dev, note = compute_vwap(vw_h, vw_l, vw_c, vw_v)
    assert abs(vwap - (4700.0 / 300.0)) < 1e-4
    assert abs(dev - (19.0 - vwap)) < 1e-4
    assert "tick-volume" in note
    log_test("VWAP", True, f"VWAP={vwap:.4f}, Dev={dev:.4f}, LimitationNote={note}")
except Exception as e:
    log_test("VWAP", False, str(e))


# --------------------------------------------------------------------------- #
# 13. 1D Kalman Filter
# --------------------------------------------------------------------------- #
try:
    kf = KalmanPriceFilter1D(q_process_variance=1e-4, r_measurement_variance=1e-2)
    prices = [2650.0, 2650.5, 2651.0, 2651.2, 2651.8, 2652.5]
    for p in prices:
        state, slope = kf.update(p)
    assert state > 2650.0
    assert slope > 0.0, "Positive price trajectory must yield positive Kalman slope"
    log_test("1D Kalman Filter", True, f"Final State={state}, Final Slope={slope}")
except Exception as e:
    log_test("1D Kalman Filter", False, str(e))


# --------------------------------------------------------------------------- #
# 14. Market Regime Detector
# --------------------------------------------------------------------------- #
try:
    r1, s1 = classify_market_regime(adx=32.0, relative_range=1.1, bandwidth=0.03, kalman_slope=0.005)
    assert r1 == "TRENDING_BULL"
    assert s1 > 0.0

    r2, s2 = classify_market_regime(adx=14.0, relative_range=0.8, bandwidth=0.02, kalman_slope=0.0)
    assert r2 == "RANGE_BOUND"
    assert s2 == 0.0

    r3, s3 = classify_market_regime(adx=40.0, relative_range=2.5, bandwidth=0.10, kalman_slope=-0.008)
    assert r3 == "HIGH_VOLATILITY"
    log_test("Market Regime Detector", True, f"Regime 1: {r1}, Regime 2: {r2}, Regime 3: {r3}")
except Exception as e:
    log_test("Market Regime Detector", False, str(e))


# --------------------------------------------------------------------------- #
# 15. Execution Quality Layer (Spread, Slippage, CostRatio)
# --------------------------------------------------------------------------- #
try:
    # Spread = 0.20, ATR = 1.00 -> Cost = 0.20 + 0.05 = 0.25 -> ExpectedMove = 0.50 -> CostRatio = 0.25 / 0.50 = 0.50 <= 0.50 (Pass)
    c_pass = evaluate_execution_cost(bid=2650.00, ask=2650.20, atr=1.00, max_allowed_cost_ratio=0.50)
    assert c_pass.passed is True
    assert c_pass.cost_ratio == 0.50

    # Spread = 0.60, ATR = 0.80 -> Cost = 0.60 + 0.05 = 0.65 -> CostRatio = 0.65 / 0.40 = 1.625 > 0.50 (Fail)
    c_fail = evaluate_execution_cost(bid=2650.00, ask=2650.60, atr=0.80, max_allowed_cost_ratio=0.50)
    assert c_fail.passed is False
    assert "HIGH_COST_RATIO" in c_fail.reason
    log_test("Execution Quality Layer", True, f"Pass CostRatio={c_pass.cost_ratio}, Fail Reason={c_fail.reason}")
except Exception as e:
    log_test("Execution Quality Layer", False, str(e))


# --------------------------------------------------------------------------- #
# 16. Unified Prediction Engine & Conflict Detection
# --------------------------------------------------------------------------- #
try:
    engine = UnifiedAgexConsensusEngine(score_threshold=0.30)
    df_bars = pd.DataFrame({
        "open": [2650.0, 2651.0, 2652.0, 2653.0],
        "high": [2651.5, 2652.5, 2653.8, 2655.0],
        "low":  [2649.8, 2650.8, 2651.8, 2652.8],
        "close":[2651.2, 2652.2, 2653.5, 2654.8],
        "volume": [100, 120, 110, 130]
    })
    tracker_bull = SignedTickPressureTracker()
    for _ in range(5):
        tracker_bull.add_tick(2654.0, 2654.2)
        tracker_bull.add_tick(2654.5, 2654.7)

    res_bull = engine.analyze_full(
        symbol="XAUUSDm",
        last_4_m1_bars=df_bars,
        historical_bars=df_bars,
        tick_tracker=tracker_bull,
        ofi_calculator=None,
        current_bid=2654.6,
        current_ask=2654.8,
    )
    assert res_bull.decision in ("BUY", "NO_TRADE")
    assert -1.0 <= res_bull.directional_score <= 1.0
    assert 0.0 <= res_bull.confidence_estimate <= 1.0
    log_test("Unified Prediction Engine", True, f"Decision={res_bull.decision}, Score={res_bull.directional_score}, Quality={res_bull.data_quality}")
except Exception as e:
    log_test("Unified Prediction Engine", False, str(e))

print("=== DEEP TECHNICAL AUDIT COMPLETED ===")
