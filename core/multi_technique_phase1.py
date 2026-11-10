"""
AGEX AI — Phase 1 Multi-Technique Prediction Engine.

Implements the institutional mathematical formulas specified in
`Multi-Technique Prediction.md` (Sections 3, 4, 5, 6, 8, 9, 10, 24, 25, 26, 30):
  1. Signed Tick Pressure (STP) — Market microstructure quote direction
  2. Candle Body Strength & SignedBody — Intrabar range dominance
  3. Close Location Value (CLV) — Normalised intrabar settlement [-1, +1]
  4. Momentum Consistency (MC) — Directional alignment across returns [-1, +1]
  5. Wick Rejection Ratios — Upper and lower exhaustion detection
  6. Volatility & Relative Range (Wilder's ATR, RelativeRange)
  7. Execution Cost Filter (Spread, Slippage, CostRatio)
  8. Phase 1 Consensus Decision Engine with Structured JSON Output
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. TECHNIQUE 1: SIGNED TICK PRESSURE (Section 3)
# --------------------------------------------------------------------------- #
@dataclass
class TickObservation:
    timestamp: float
    bid: float
    ask: float
    mid: float
    direction: int  # +1 (up), -1 (down), 0 (unchanged)


class SignedTickPressureTracker:
    """
    Maintains a rolling window of ticks (e.g. 4 minutes) and calculates
    institutional Signed Tick Pressure (STP):
        Mid_i = (Bid_i + Ask_i) / 2
        d_i   = sign(Mid_i - Mid_{i-1})
        STP   = Sum(d_i) / N
    """

    def __init__(self, window_seconds: float = 240.0):
        self.window_seconds = float(window_seconds)
        self.ticks: List[TickObservation] = []
        self._last_mid: Optional[float] = None
        self._last_ts: float = 0.0

    def add_tick(self, bid: float, ask: float, timestamp: Optional[float] = None) -> Optional[TickObservation]:
        """Ingests a quote tick and updates signed tick direction."""
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            return None

        ts = float(timestamp if timestamp is not None else time.time())
        if ts < self._last_ts:
            ts = self._last_ts + 0.0001
        self._last_ts = ts

        mid = (bid + ask) / 2.0
        direction = 0
        if self._last_mid is not None:
            diff = mid - self._last_mid
            if diff > 1e-6:
                direction = 1
            elif diff < -1e-6:
                direction = -1
            else:
                direction = 0

        self._last_mid = mid
        obs = TickObservation(timestamp=ts, bid=round(bid, 5), ask=round(ask, 5), mid=round(mid, 5), direction=direction)
        self.ticks.append(obs)

        # Evict ticks older than the window
        cutoff = ts - self.window_seconds
        while self.ticks and self.ticks[0].timestamp < cutoff:
            self.ticks.pop(0)

        return obs

    def clear(self) -> None:
        self.ticks.clear()
        self._last_mid = None
        self._last_ts = 0.0

    def compute_stp(self) -> Dict[str, Any]:
        """
        Computes STP and direction breakdown.
        Returns:
            stp: float in [-1.0, +1.0]
            up_count: int
            down_count: int
            zero_count: int
            total_observations: int
        """
        n = len(self.ticks)
        if n < 2:
            return {
                "stp": 0.0,
                "up_count": 0,
                "down_count": 0,
                "zero_count": 0,
                "total_observations": n,
                "status": "INSUFFICIENT_DATA",
            }

        up = sum(1 for t in self.ticks if t.direction == 1)
        down = sum(1 for t in self.ticks if t.direction == -1)
        zero = sum(1 for t in self.ticks if t.direction == 0)
        net_direction_sum = up - down
        stp = net_direction_sum / float(n)

        return {
            "stp": round(stp, 4),
            "up_count": up,
            "down_count": down,
            "zero_count": zero,
            "total_observations": n,
            "status": "VALID",
        }


# --------------------------------------------------------------------------- #
# 2. TECHNIQUES 2, 3, 7: CANDLE METRICS (Sections 4, 5, 9)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CandleMetrics:
    open: float
    high: float
    low: float
    close: float
    range: float
    body: float
    body_strength: float         # abs(C - O) / Range
    signed_body: float           # (C - O) / Range
    clv: float                   # (2*C - H - L) / (H - L) in [-1.0, +1.0]
    upper_wick: float
    lower_wick: float
    upper_rejection_ratio: float # UpperWick / Range
    lower_rejection_ratio: float # LowerWick / Range


def compute_candle_metrics(open_p: float, high_p: float, low_p: float, close_p: float) -> CandleMetrics:
    """Computes BodyStrength, SignedBody, CLV, and Wick Rejections for a candle or aggregate."""
    c_range = max(high_p - low_p, 0.0001)
    body = abs(close_p - open_p)
    body_strength = min(1.0, max(0.0, body / c_range))
    signed_body = min(1.0, max(-1.0, (close_p - open_p) / c_range))

    # Close Location Value (CLV): Formula (2*Close - High - Low) / (High - Low)
    clv = min(1.0, max(-1.0, (2.0 * close_p - high_p - low_p) / c_range))

    upper_wick = max(0.0, high_p - max(open_p, close_p))
    lower_wick = max(0.0, min(open_p, close_p) - low_p)

    upper_rejection = min(1.0, max(0.0, upper_wick / c_range))
    lower_rejection = min(1.0, max(0.0, lower_wick / c_range))

    return CandleMetrics(
        open=round(open_p, 4),
        high=round(high_p, 4),
        low=round(low_p, 4),
        close=round(close_p, 4),
        range=round(c_range, 4),
        body=round(body, 4),
        body_strength=round(body_strength, 4),
        signed_body=round(signed_body, 4),
        clv=round(clv, 4),
        upper_wick=round(upper_wick, 4),
        lower_wick=round(lower_wick, 4),
        upper_rejection_ratio=round(upper_rejection, 4),
        lower_rejection_ratio=round(lower_rejection, 4),
    )


# --------------------------------------------------------------------------- #
# 3. TECHNIQUE 4: MOMENTUM CONSISTENCY (Section 6)
# --------------------------------------------------------------------------- #
def compute_momentum_consistency(close_prices: List[float]) -> Dict[str, Any]:
    """
    Computes directional momentum consistency over recent returns:
        r_j = ln(Close_j / Close_{j-1})
        MC  = Sum(sign(r_j)) / K
    """
    if len(close_prices) < 2:
        return {"mc": 0.0, "status": "INSUFFICIENT_DATA", "returns": []}

    returns = []
    signs = []
    for i in range(1, len(close_prices)):
        prev = close_prices[i - 1]
        curr = close_prices[i]
        if prev > 0.0 and curr > 0.0:
            r = math.log(curr / prev)
            returns.append(r)
            if r > 1e-6:
                signs.append(1)
            elif r < -1e-6:
                signs.append(-1)
            else:
                signs.append(0)

    if not signs:
        return {"mc": 0.0, "status": "INSUFFICIENT_DATA", "returns": []}

    mc = sum(signs) / float(len(signs))
    return {
        "mc": round(min(1.0, max(-1.0, mc)), 4),
        "status": "VALID",
        "return_count": len(signs),
        "positive_count": sum(1 for s in signs if s > 0),
        "negative_count": sum(1 for s in signs if s < 0),
    }


# --------------------------------------------------------------------------- #
# 4. TECHNIQUE 6: VOLATILITY & RELATIVE RANGE (Section 8)
# --------------------------------------------------------------------------- #
def compute_atr_and_relative_range(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> Dict[str, Any]:
    """
    Calculates True Range, Wilder's Smoothed ATR, and Relative Range.
    """
    n = len(closes)
    if n < 2:
        return {
            "atr": 0.50,
            "relative_range": 1.0,
            "current_range": 0.50,
            "average_range": 0.50,
            "status": "FALLBACK_INSUFFICIENT_BARS",
        }

    # True Range sequence
    tr_list = []
    ranges = []
    for i in range(1, n):
        h = highs[i]
        l = lows[i]
        prev_c = closes[i - 1]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        tr_list.append(tr)
        ranges.append(h - l)

    if not tr_list:
        return {"atr": 0.50, "relative_range": 1.0, "status": "NO_TR"}

    # Wilder's Smoothing for ATR
    k = min(period, len(tr_list))
    atr = sum(tr_list[:k]) / float(k)
    for i in range(k, len(tr_list)):
        atr = ((atr * (k - 1)) + tr_list[i]) / float(k)

    # Average range of previous completed candles
    prev_ranges = ranges[:-1] if len(ranges) > 1 else ranges
    avg_prev_range = sum(prev_ranges[-period:]) / float(len(prev_ranges[-period:])) if prev_ranges else 1.0
    current_range = ranges[-1] if ranges else 1.0
    rel_range = current_range / avg_prev_range if avg_prev_range > 1e-6 else 1.0

    return {
        "atr": round(atr, 4),
        "current_range": round(current_range, 4),
        "average_range": round(avg_prev_range, 4),
        "relative_range": round(rel_range, 3),
        "status": "VALID",
    }


# --------------------------------------------------------------------------- #
# 5. TECHNIQUE 8: SPREAD AND EXECUTION COST FILTER (Section 10)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecutionCostAssessment:
    spread: float
    commission: float
    estimated_slippage: float
    estimated_trading_cost: float
    expected_move: float
    cost_ratio: float
    passed: bool
    reason: str


def evaluate_execution_cost(
    bid: float,
    ask: float,
    atr: float,
    commission_pts: float = 0.0,
    estimated_slippage_pts: float = 0.05,
    movement_factor: float = 0.50,
    max_allowed_cost_ratio: float = 0.40,
) -> ExecutionCostAssessment:
    """
    Evaluates whether execution costs consume too much of the expected movement:
        EstimatedTradingCost = Spread + Commission + EstimatedSlippage
        ExpectedMove = ATR * MovementFactor
        CostRatio = EstimatedTradingCost / ExpectedMove
    """
    spread = max(0.0, ask - bid)
    trading_cost = spread + commission_pts + estimated_slippage_pts
    expected_move = max(0.05, atr * movement_factor)
    cost_ratio = trading_cost / expected_move

    passed = cost_ratio <= max_allowed_cost_ratio
    reason = "COST_OK" if passed else f"HIGH_COST_RATIO_{cost_ratio:.2f}_EXCEEDS_{max_allowed_cost_ratio:.2f}"

    return ExecutionCostAssessment(
        spread=round(spread, 4),
        commission=round(commission_pts, 4),
        estimated_slippage=round(estimated_slippage_pts, 4),
        estimated_trading_cost=round(trading_cost, 4),
        expected_move=round(expected_move, 4),
        cost_ratio=round(cost_ratio, 3),
        passed=passed,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# 6. CENTRALIZED PHASE 1 CONSENSUS ENGINE (Sections 24, 26, 30)
# --------------------------------------------------------------------------- #
@dataclass
class Phase1PredictionResult:
    symbol: str
    analysis_window: str
    prediction_horizon: str
    decision: str  # BUY | SELL | NO_TRADE
    directional_score: float  # [-1.0, +1.0]
    confidence_estimate: float  # [0.0, 1.0]
    probabilities: Dict[str, Optional[float]]
    feature_groups: Dict[str, Any]
    ofi_status: str
    data_quality: str
    expected_move: float
    current_spread: float
    estimated_trading_cost: float
    reasons: List[str]
    risk_status: str
    signal_timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Serializes strictly according to Section 30 JSON schema."""
        return {
            "symbol": self.symbol,
            "analysis_window": self.analysis_window,
            "prediction_horizon": self.prediction_horizon,
            "decision": self.decision,
            "directional_score": self.directional_score,
            "confidence_estimate": self.confidence_estimate,
            "probabilities": self.probabilities,
            "feature_groups": self.feature_groups,
            "ofi_status": self.ofi_status,
            "data_quality": self.data_quality,
            "expected_move": self.expected_move,
            "current_spread": self.current_spread,
            "estimated_trading_cost": self.estimated_trading_cost,
            "reasons": self.reasons,
            "risk_status": self.risk_status,
            "signal_timestamp": self.signal_timestamp,
        }


class Phase1PredictionEngine:
    """
    Centralized Phase 1 Prediction Engine for AGEX AI.
    Combines:
      - Signed Tick Pressure (Group 1)
      - Candle Pressure & Metrics (Group 2)
      - Momentum Consistency (Group 3)
      - Volatility & Range Context (Group 4)
      - Execution Cost Filter (Group 7)
    Produces calibrated consensus direction: BUY, SELL, or NO_TRADE.
    """

    def __init__(
        self,
        weight_order_flow: float = 0.30,
        weight_candle_pressure: float = 0.40,
        weight_momentum: float = 0.30,
        score_threshold: float = 0.35,
        max_cost_ratio: float = 0.40,
    ):
        self.w_of = float(weight_order_flow)
        self.w_cp = float(weight_candle_pressure)
        self.w_mo = float(weight_momentum)
        self.threshold = float(score_threshold)
        self.max_cost_ratio = float(max_cost_ratio)

    def analyze(
        self,
        symbol: str,
        last_4_m1_bars: pd.DataFrame,
        historical_bars: Optional[pd.DataFrame],
        tick_tracker: SignedTickPressureTracker,
        current_bid: float,
        current_ask: float,
        ofi_calculator: Optional[Any] = None,
    ) -> Phase1PredictionResult:
        now_ts = time.time()
        reasons: List[str] = []

        # 1. Data Quality Check
        if last_4_m1_bars is None or len(last_4_m1_bars) < 4:
            return Phase1PredictionResult(
                symbol=symbol,
                analysis_window="4 completed minutes",
                prediction_horizon="next 1 minute",
                decision="NO_TRADE",
                directional_score=0.0,
                confidence_estimate=0.0,
                probabilities={"buy": None, "sell": None, "no_trade": 1.0},
                feature_groups={},
                ofi_status="UNAVAILABLE",
                data_quality="INVALID",
                expected_move=0.0,
                current_spread=max(0.0, current_ask - current_bid),
                estimated_trading_cost=0.0,
                reasons=["INSUFFICIENT_M1_BARS"],
                risk_status="BLOCKED",
                signal_timestamp=now_ts,
            )

        # 2. Group 1: Order Flow (Signed Tick Pressure + OFI if available)
        stp_res = tick_tracker.compute_stp()
        order_flow_score = float(stp_res["stp"])

        ofi_status = "UNAVAILABLE"
        ofi_metrics = {}
        if ofi_calculator is not None and hasattr(ofi_calculator, "compute_ofi_metrics"):
            ofi_metrics = ofi_calculator.compute_ofi_metrics()
            ofi_status = str(ofi_metrics.get("status", "UNAVAILABLE"))
            if ofi_status == "AVAILABLE":
                norm_ofi = float(ofi_metrics.get("normalized_ofi", 0.0))
                # Combine STP and Normalized OFI with equal weighting
                order_flow_score = min(1.0, max(-1.0, 0.50 * float(stp_res["stp"]) + 0.50 * norm_ofi))

        # 3. Group 2: Candle Pressure & Aggregate Metrics
        agg_open = float(last_4_m1_bars["open"].iloc[0])
        agg_close = float(last_4_m1_bars["close"].iloc[-1])
        agg_high = float(last_4_m1_bars["high"].max())
        agg_low = float(last_4_m1_bars["low"].min())

        agg_metrics = compute_candle_metrics(agg_open, agg_high, agg_low, agg_close)
        # Candle Score combines SignedBody (40%), CLV (35%), and Rejection Bias (25%)
        rejection_bias = agg_metrics.lower_rejection_ratio - agg_metrics.upper_rejection_ratio
        candle_score = (
            0.40 * agg_metrics.signed_body
            + 0.35 * agg_metrics.clv
            + 0.25 * rejection_bias
        )
        candle_score = min(1.0, max(-1.0, candle_score))

        # 4. Group 3: Momentum Consistency across 4 M1 bars
        # Prepend last bar before the 4 bars if available in historical_bars for 4 clean returns
        all_closes = []
        if historical_bars is not None and len(historical_bars) >= 5:
            all_closes = [float(c) for c in historical_bars["close"].iloc[-5:]]
        else:
            all_closes = [float(c) for c in last_4_m1_bars["close"]]

        mc_res = compute_momentum_consistency(all_closes)
        momentum_score = float(mc_res["mc"])

        # 5. Group 4: Volatility & ATR
        eval_bars = historical_bars if historical_bars is not None and len(historical_bars) >= 15 else last_4_m1_bars
        highs = [float(h) for h in eval_bars["high"]]
        lows = [float(l) for l in eval_bars["low"]]
        closes = [float(c) for c in eval_bars["close"]]
        vol_res = compute_atr_and_relative_range(highs, lows, closes, period=14)
        atr_val = float(vol_res["atr"])
        rel_range = float(vol_res["relative_range"])

        # 6. Group 7: Execution Cost Assessment
        cost_eval = evaluate_execution_cost(
            bid=current_bid,
            ask=current_ask,
            atr=atr_val,
            max_allowed_cost_ratio=self.max_cost_ratio,
        )

        # 7. Consensus Directional Score (Section 24.B: Missing data redistribution)
        valid_weights = {}
        if stp_res["status"] == "VALID":
            valid_weights["of"] = self.w_of
        valid_weights["cp"] = self.w_cp
        if mc_res["status"] == "VALID":
            valid_weights["mo"] = self.w_mo

        total_valid = sum(valid_weights.values())
        if total_valid <= 0:
            final_score = 0.0
        else:
            final_score = 0.0
            if "of" in valid_weights:
                final_score += (valid_weights["of"] / total_valid) * order_flow_score
            if "cp" in valid_weights:
                final_score += (valid_weights["cp"] / total_valid) * candle_score
            if "mo" in valid_weights:
                final_score += (valid_weights["mo"] / total_valid) * momentum_score

        final_score = min(1.0, max(-1.0, round(final_score, 4)))
        confidence = round(abs(final_score), 3)

        # Decision Evaluation
        decision = "NO_TRADE"
        risk_status = "PASS"

        # Check guards:
        if not cost_eval.passed:
            decision = "NO_TRADE"
            risk_status = "BLOCKED"
            reasons.append(cost_eval.reason)

        elif rel_range < 0.25:
            decision = "NO_TRADE"
            reasons.append(f"DEAD_MARKET_VOLATILITY_TOO_LOW_{rel_range:.2f}")

        # Choppiness / Indecision check (e.g. body strength < 15% with high equal wicks)
        elif agg_metrics.body_strength < 0.15 and agg_metrics.upper_rejection_ratio >= 0.25 and agg_metrics.lower_rejection_ratio >= 0.25:
            decision = "NO_TRADE"
            reasons.append("AGGREGATE_INDECISION_CHOP")

        # Consensus BUY: FinalScore >= threshold, CandleScore > 0, OrderFlow >= -0.10
        elif final_score >= self.threshold and candle_score > 0.05 and order_flow_score >= -0.10:
            decision = "BUY"
            reasons.append(f"CONSENSUS_BULLISH_SCORE_{final_score:+.2f}_STP_{order_flow_score:+.2f}_CLV_{agg_metrics.clv:+.2f}")

        # Consensus SELL: FinalScore <= -threshold, CandleScore < 0, OrderFlow <= +0.10
        elif final_score <= -self.threshold and candle_score < -0.05 and order_flow_score <= 0.10:
            decision = "SELL"
            reasons.append(f"CONSENSUS_BEARISH_SCORE_{final_score:+.2f}_STP_{order_flow_score:+.2f}_CLV_{agg_metrics.clv:+.2f}")

        else:
            decision = "NO_TRADE"
            reasons.append(f"WEAK_OR_CONFLICTING_EVIDENCE_SCORE_{final_score:+.2f}")

        # Feature groups output dictionary
        feature_groups = {
            "order_flow": {
                "stp": float(stp_res["stp"]),
                "order_flow_consensus": order_flow_score,
                "up_ticks": stp_res["up_count"],
                "down_ticks": stp_res["down_count"],
                "total_ticks": stp_res["total_observations"],
                "ofi_status": ofi_status,
                "normalized_ofi": float(ofi_metrics.get("normalized_ofi", 0.0)),
                "depth_imbalance": float(ofi_metrics.get("depth_imbalance", 0.0)),
            },
            "candle_pressure": {
                "signed_body": agg_metrics.signed_body,
                "body_strength": agg_metrics.body_strength,
                "clv": agg_metrics.clv,
                "upper_rejection": agg_metrics.upper_rejection_ratio,
                "lower_rejection": agg_metrics.lower_rejection_ratio,
            },
            "momentum": {
                "momentum_consistency": momentum_score,
                "return_count": mc_res.get("return_count", 0),
            },
            "volatility": {
                "atr": atr_val,
                "relative_range": rel_range,
                "average_range": vol_res["average_range"],
            },
            "market_structure": None,  # Phase 3
            "market_regime": None,     # Phase 3
            "execution_quality": {
                "spread": cost_eval.spread,
                "cost_ratio": cost_eval.cost_ratio,
                "passed": cost_eval.passed,
            },
        }

        # Probabilities
        if decision == "BUY":
            probs = {"buy": round(0.50 + (confidence * 0.40), 2), "sell": round(0.50 - (confidence * 0.40), 2), "no_trade": 0.0}
        elif decision == "SELL":
            probs = {"buy": round(0.50 - (confidence * 0.40), 2), "sell": round(0.50 + (confidence * 0.40), 2), "no_trade": 0.0}
        else:
            probs = {"buy": 0.0, "sell": 0.0, "no_trade": 1.0}

        return Phase1PredictionResult(
            symbol=symbol,
            analysis_window="4 completed minutes",
            prediction_horizon="next 1 minute",
            decision=decision,
            directional_score=final_score,
            confidence_estimate=confidence,
            probabilities=probs,
            feature_groups=feature_groups,
            ofi_status=ofi_status,
            data_quality="GOOD" if stp_res["status"] == "VALID" else "DEGRADED",
            expected_move=cost_eval.expected_move,
            current_spread=cost_eval.spread,
            estimated_trading_cost=cost_eval.estimated_trading_cost,
            reasons=reasons,
            risk_status=risk_status,
            signal_timestamp=now_ts,
        )
