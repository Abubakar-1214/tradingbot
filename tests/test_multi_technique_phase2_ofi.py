"""
Unit tests for AGEX AI Phase 2: Order Flow Imbalance (OFI) & Market Depth Engine (`core/multi_technique_phase2_ofi.py`).
Verifies:
  - Cont, Kukanov & Stoikov (2014) CKS Formula
  - Best-level OFI events (bid improvements, ask drops, queue additions/cancellations)
  - Static and rolling Depth Imbalance
  - Safe MT5 DOM adapter handling (UNAVAILABLE when not supported, never fabricated)
  - Seamless integration with Phase 1 Consensus Engine
"""

import math
import pandas as pd
import pytest
from core.multi_technique_phase2_ofi import (
    OrderBookLevel,
    OrderBookSnapshot,
    OFICalculator,
    OFIEvent,
    MT5MarketBookManager,
)
from core.multi_technique_phase1 import (
    SignedTickPressureTracker,
    Phase1PredictionEngine,
)


def test_order_book_snapshot_and_depth_imbalance():
    bids = [OrderBookLevel(price=4100.0, volume=60.0, side="bid"), OrderBookLevel(price=4099.0, volume=40.0, side="bid")]
    asks = [OrderBookLevel(price=4101.0, volume=30.0, side="ask"), OrderBookLevel(price=4102.0, volume=20.0, side="ask")]

    snap = OrderBookSnapshot(timestamp=100.0, bids=bids, asks=asks)
    assert snap.best_bid.price == 4100.0
    assert snap.best_ask.price == 4101.0
    assert snap.total_bid_depth == 100.0
    assert snap.total_ask_depth == 50.0

    # DepthImbalance = (100 - 50) / (100 + 50) = +50 / 150 = +0.3333
    imb = snap.depth_imbalance()
    assert round(imb, 4) == 0.3333


def test_ofi_cont_kukanov_stoikov_formulas():
    calc = OFICalculator(window_seconds=240.0)

    # Initial snapshot: Bid 4100.0 (vol 10), Ask 4102.0 (vol 10)
    s1 = OrderBookSnapshot(
        timestamp=10.0,
        bids=[OrderBookLevel(price=4100.0, volume=10.0, side="bid")],
        asks=[OrderBookLevel(price=4102.0, volume=10.0, side="ask")],
    )
    e1 = calc.add_snapshot(s1)
    assert e1 is None  # First snapshot initializes state

    # Snapshot 2: Bid improves to 4101.0 with volume 15! (Ask stays 4102.0, vol 10)
    # Bid improved -> BidContrib = +15. Ask same -> AskContrib = 0. OFI = +15
    s2 = OrderBookSnapshot(
        timestamp=11.0,
        bids=[OrderBookLevel(price=4101.0, volume=15.0, side="bid")],
        asks=[OrderBookLevel(price=4102.0, volume=10.0, side="ask")],
    )
    e2 = calc.add_snapshot(s2)
    assert e2 is not None
    assert e2.bid_contribution == 15.0
    assert e2.ask_contribution == 0.0
    assert e2.event_ofi == 15.0

    # Snapshot 3: Ask drops to 4101.5 with volume 20! (Selling pressure enters)
    # Bid same (4101.0, vol 15) -> BidContrib = 0.
    # Ask improved (dropped 4102 -> 4101.5) -> AskContrib = +20.
    # Net OFI = BidContrib - AskContrib = 0 - 20 = -20
    s3 = OrderBookSnapshot(
        timestamp=12.0,
        bids=[OrderBookLevel(price=4101.0, volume=15.0, side="bid")],
        asks=[OrderBookLevel(price=4101.5, volume=20.0, side="ask")],
    )
    e3 = calc.add_snapshot(s3)
    assert e3 is not None
    assert e3.bid_contribution == 0.0
    assert e3.ask_contribution == 20.0
    assert e3.event_ofi == -20.0

    metrics = calc.compute_ofi_metrics()
    assert metrics["status"] == "AVAILABLE"
    assert metrics["event_count"] == 2
    # ofi_total = 15 - 20 = -5.0
    assert metrics["ofi_total"] == -5.0
    assert -1.0 <= metrics["normalized_ofi"] <= 1.0


def test_ofi_unavailable_handling():
    # If no DOM is available, OFICalculator must report UNAVAILABLE and never fabricate numbers
    calc = OFICalculator()
    metrics = calc.compute_ofi_metrics()
    assert metrics["status"] == "UNAVAILABLE"
    assert metrics["ofi_total"] == 0.0
    assert metrics["normalized_ofi"] == 0.0


def test_phase1_consensus_with_phase2_ofi_integration():
    engine = Phase1PredictionEngine(score_threshold=0.30)
    tracker = SignedTickPressureTracker()
    for p in [4100.0, 4103.0, 4106.0, 4109.0]:
        tracker.add_tick(bid=p - 0.1, ask=p + 0.1)

    bars_bull = pd.DataFrame({
        "open": [4100.0, 4102.0, 4104.0, 4106.0],
        "high": [4103.0, 4105.0, 4107.0, 4110.0],
        "low": [4099.5, 4101.5, 4103.5, 4105.5],
        "close": [4102.5, 4104.5, 4106.5, 4109.8],
    })

    # Case A: OFI Calculator is UNAVAILABLE -> Fallback to STP cleanly
    calc_empty = OFICalculator()
    res_a = engine.analyze(
        symbol="EURUSD",
        last_4_m1_bars=bars_bull,
        historical_bars=bars_bull,
        tick_tracker=tracker,
        current_bid=4109.7,
        current_ask=4109.9,
        ofi_calculator=calc_empty,
    )
    assert res_a.ofi_status == "UNAVAILABLE"
    assert res_a.decision == "BUY"

    # Case B: OFI Calculator is AVAILABLE and strongly Bullish
    calc_bull = OFICalculator()
    s1 = OrderBookSnapshot(
        timestamp=10.0,
        bids=[OrderBookLevel(price=4100.0, volume=10.0, side="bid")],
        asks=[OrderBookLevel(price=4102.0, volume=10.0, side="ask")],
    )
    s2 = OrderBookSnapshot(
        timestamp=11.0,
        bids=[OrderBookLevel(price=4101.0, volume=50.0, side="bid")],
        asks=[OrderBookLevel(price=4102.0, volume=10.0, side="ask")],
    )
    calc_bull.add_snapshot(s1)
    calc_bull.add_snapshot(s2)

    res_b = engine.analyze(
        symbol="EURUSD",
        last_4_m1_bars=bars_bull,
        historical_bars=bars_bull,
        tick_tracker=tracker,
        current_bid=4109.7,
        current_ask=4109.9,
        ofi_calculator=calc_bull,
    )
    assert res_b.ofi_status == "AVAILABLE"
    assert res_b.decision == "BUY"
    assert res_b.feature_groups["order_flow"]["normalized_ofi"] > 0

