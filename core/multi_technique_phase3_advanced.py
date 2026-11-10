"""
AGEX AI — Phase 3: Advanced Market Structure, Regimes & Full 7-Group Consensus Engine.

Implements all remaining institutional specifications from `Multi-Technique Prediction.md`:
  - Technique 11: Market Structure (HH, HL, LH, LL, Bullish/Bearish BOS) (Section 13)
  - Technique 12: EMA / SMA Trend Filters & Slope (Section 14)
  - Technique 14: MACD Line, Signal & Histogram (Section 16)
  - Technique 15: Average Directional Index (ADX) & Directional Movement (+DI, -DI) (Section 17)
  - Technique 16: Bollinger Bands & BandWidth (Section 18)
  - Technique 17: 1D Kalman Filter (Noise reduction & trend slope) (Section 19)
  - Technique 18: Market Regime Detector (Trending, Range-Bound, High-Vol) (Section 20)
  - Technique 19: Classic Pivot Points (P, R1, S1, R2, S2) (Section 21)
  - Technique 20: Fibonacci Retracement Levels (Section 22)
  - Technique 21: Donchian Channels & Breakout Context (Section 23)
  - Unified 7-Group Consensus Decision Engine (Section 24, 30)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. TECHNIQUE 11, 19, 20, 21: MARKET STRUCTURE & PRICE LEVELS
# --------------------------------------------------------------------------- #
@dataclass
class MarketStructureResult:
    trend_state: str  # "BULLISH_STRUCTURE", "BEARISH_STRUCTURE", "RANGE"
    swing_high: float
    swing_low: float
    structure_score: float  # [-1.0, +1.0]
    bos_signal: int  # +1 Bullish BOS, -1 Bearish BOS, 0 None
    pivot_p: float
    pivot_r1: float
    pivot_s1: float
    donchian_high: float
    donchian_low: float
    donchian_mid: float
    fib_50: float
    fib_618: float


def compute_market_structure(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    lookback: int = 3,
) -> MarketStructureResult:
    """
    Computes Swing Points, Break of Structure (BOS), Classic Pivots, Donchian Channel, and Fibs.
    """
    n = len(closes)
    if n < 5:
        cur_p = closes[-1] if closes else 1.0
        return MarketStructureResult(
            trend_state="RANGE",
            swing_high=cur_p + 1.0,
            swing_low=cur_p - 1.0,
            structure_score=0.0,
            bos_signal=0,
            pivot_p=cur_p,
            pivot_r1=cur_p + 0.5,
            pivot_s1=cur_p - 0.5,
            donchian_high=cur_p + 1.0,
            donchian_low=cur_p - 1.0,
            donchian_mid=cur_p,
            fib_50=cur_p,
            fib_618=cur_p,
        )

    # 1. Identify confirmed Swing Highs and Swing Lows (using lag to prevent lookahead)
    swing_highs = []
    swing_lows = []
    for i in range(lookback, n - 1):
        is_sh = all(highs[i] >= highs[i - j] for j in range(1, lookback + 1)) and (highs[i] >= highs[i + 1])
        is_sl = all(lows[i] <= lows[i - j] for j in range(1, lookback + 1)) and (lows[i] <= lows[i + 1])
        if is_sh:
            swing_highs.append(highs[i])
        if is_sl:
            swing_lows.append(lows[i])

    prior_highs = highs[:-1] if len(highs) > 1 else highs
    prior_lows = lows[:-1] if len(lows) > 1 else lows
    recent_sh = swing_highs[-1] if swing_highs else max(prior_highs[-10:] if len(prior_highs) >= 10 else prior_highs)
    recent_sl = swing_lows[-1] if swing_lows else min(prior_lows[-10:] if len(prior_lows) >= 10 else prior_lows)

    cur_c = closes[-1]

    # Break of Structure (BOS)
    bos_signal = 0
    if cur_c > recent_sh:
        bos_signal = 1  # Bullish BOS
    elif cur_c < recent_sl:
        bos_signal = -1 # Bearish BOS

    # Structure Score: +1.0 for Bullish BOS, -1.0 for Bearish BOS, or normalized location between SL and SH
    struct_range = max(recent_sh - recent_sl, 0.001)
    pos_in_range = ((cur_c - recent_sl) / struct_range) * 2.0 - 1.0  # mapped to [-1.0, +1.0]

    if bos_signal == 1:
        structure_score = 1.0
        trend_state = "BULLISH_STRUCTURE"
    elif bos_signal == -1:
        structure_score = -1.0
        trend_state = "BEARISH_STRUCTURE"
    else:
        structure_score = min(0.8, max(-0.8, pos_in_range))
        trend_state = "RANGE" if abs(pos_in_range) < 0.3 else ("BULLISH_STRUCTURE" if pos_in_range > 0 else "BEARISH_STRUCTURE")

    # Classic Pivot Levels (Section 21)
    prev_h = max(highs[-20:])
    prev_l = min(lows[-20:])
    prev_c = closes[-2]
    pivot_p = (prev_h + prev_l + prev_c) / 3.0
    pivot_r1 = 2.0 * pivot_p - prev_l
    pivot_s1 = 2.0 * pivot_p - prev_h

    # Donchian Channel (Section 23)
    k_donchian = min(20, n)
    donchian_h = max(highs[-k_donchian:])
    donchian_l = min(lows[-k_donchian:])
    donchian_m = (donchian_h + donchian_l) / 2.0

    # Fibonacci Retracement (Section 22)
    fib_range = max(recent_sh - recent_sl, 0.001)
    fib_50 = recent_sh - (fib_range * 0.500)
    fib_618 = recent_sh - (fib_range * 0.618)

    return MarketStructureResult(
        trend_state=trend_state,
        swing_high=round(recent_sh, 4),
        swing_low=round(recent_sl, 4),
        structure_score=round(structure_score, 4),
        bos_signal=bos_signal,
        pivot_p=round(pivot_p, 4),
        pivot_r1=round(pivot_r1, 4),
        pivot_s1=round(pivot_s1, 4),
        donchian_high=round(donchian_h, 4),
        donchian_low=round(donchian_l, 4),
        donchian_mid=round(donchian_m, 4),
        fib_50=round(fib_50, 4),
        fib_618=round(fib_618, 4),
    )


# --------------------------------------------------------------------------- #
# 2. TECHNIQUES 12, 14, 15, 16: ADVANCED MOMENTUM & INDICATORS
# --------------------------------------------------------------------------- #
def compute_macd(closes: List[float], fast: int = 12, slow: int = 26, signal_period: int = 9) -> Tuple[float, float, float]:
    """Computes MACD line, Signal line, and Histogram."""
    if len(closes) < slow + signal_period:
        return 0.0, 0.0, 0.0

    s = pd.Series(closes)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - signal_line

    return round(float(macd_line.iloc[-1]), 4), round(float(signal_line.iloc[-1]), 4), round(float(hist.iloc[-1]), 4)


def compute_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[float, float, float]:
    """Computes +DI, -DI, and ADX."""
    n = len(closes)
    if n < period * 2:
        return 20.0, 20.0, 20.0

    tr_list = []
    plus_dm = []
    minus_dm = []
    for i in range(1, n):
        h = highs[i]
        l = lows[i]
        prev_h = highs[i - 1]
        prev_l = lows[i - 1]
        prev_c = closes[i - 1]

        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        tr_list.append(tr)

        up_move = h - prev_h
        down_move = prev_l - l

        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)

        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)

    # Wilder smoothing
    sm_tr = sum(tr_list[:period])
    sm_plus = sum(plus_dm[:period])
    sm_minus = sum(minus_dm[:period])

    dx_list = []
    for i in range(period, len(tr_list)):
        sm_tr = sm_tr - (sm_tr / period) + tr_list[i]
        sm_plus = sm_plus - (sm_plus / period) + plus_dm[i]
        sm_minus = sm_minus - (sm_minus / period) + minus_dm[i]

        pdi = (100.0 * sm_plus / sm_tr) if sm_tr > 0 else 0.0
        mdi = (100.0 * sm_minus / sm_tr) if sm_tr > 0 else 0.0
        denom = pdi + mdi
        dx = (100.0 * abs(pdi - mdi) / denom) if denom > 0 else 0.0
        dx_list.append(dx)

    adx = sum(dx_list[-period:]) / float(len(dx_list[-period:])) if dx_list else 20.0
    last_pdi = (100.0 * sm_plus / sm_tr) if sm_tr > 0 else 20.0
    last_mdi = (100.0 * sm_minus / sm_tr) if sm_tr > 0 else 20.0

    return round(last_pdi, 2), round(last_mdi, 2), round(adx, 2)


def compute_bollinger_bands(closes: List[float], period: int = 20, mult: float = 2.0) -> Tuple[float, float, float, float]:
    """Computes MiddleBand, UpperBand, LowerBand, and BandWidth."""
    if len(closes) < period:
        cur_c = closes[-1] if closes else 1.0
        return cur_c, cur_c + 1.0, cur_c - 1.0, 0.05

    sub = closes[-period:]
    mid = sum(sub) / float(period)
    variance = sum((x - mid) ** 2 for x in sub) / float(period)
    std = math.sqrt(variance)

    upper = mid + (mult * std)
    lower = mid - (mult * std)
    bandwidth = (upper - lower) / mid if mid > 0 else 0.0

    return round(mid, 4), round(upper, 4), round(lower, 4), round(bandwidth, 4)


# --------------------------------------------------------------------------- #
# 3. TECHNIQUES 17, 18: 1D KALMAN FILTER & MARKET REGIME DETECTOR
# --------------------------------------------------------------------------- #
class KalmanPriceFilter1D:
    """
    1D Kalman Filter (Section 19):
      Filters high-frequency price noise and calculates the true underlying trend slope.
    """

    def __init__(self, q_process_variance: float = 1e-4, r_measurement_variance: float = 1e-2):
        self.q = float(q_process_variance)
        self.r = float(r_measurement_variance)
        self.state: Optional[float] = None
        self.covariance: float = 1.0
        self.previous_state: Optional[float] = None

    def update(self, price: float) -> Tuple[float, float]:
        """
        Updates filter with incoming price.
        Returns:
            (filtered_state, slope_velocity)
        """
        if self.state is None:
            self.state = price
            self.covariance = 1.0
            self.previous_state = price
            return price, 0.0

        # Predict
        p_pred = self.covariance + self.q

        # Kalman Gain
        k_gain = p_pred / (p_pred + self.r)

        # Update State
        self.previous_state = self.state
        self.state = self.state + k_gain * (price - self.state)
        self.covariance = (1.0 - k_gain) * p_pred

        slope = self.state - self.previous_state
        return round(self.state, 4), round(slope, 5)


def classify_market_regime(
    adx: float,
    relative_range: float,
    bandwidth: float,
    kalman_slope: float,
) -> Tuple[str, float]:
    """
    Classifies Market Regime (Section 20):
      - TRENDING_BULL / TRENDING_BEAR (ADX > 25, clear Kalman slope)
      - HIGH_VOLATILITY (RelativeRange > 1.8, BandWidth expanding)
      - RANGE_BOUND / CHOP (ADX < 20, RelativeRange low)
    Returns:
        (regime_name, regime_directional_score [-1.0, +1.0])
    """
    if relative_range > 2.0 or bandwidth > 0.08:
        regime = "HIGH_VOLATILITY"
        score = 0.50 if kalman_slope > 0 else -0.50
    elif adx >= 25.0:
        if kalman_slope > 1e-4:
            regime = "TRENDING_BULL"
            score = min(1.0, kalman_slope * 100.0)
        elif kalman_slope < -1e-4:
            regime = "TRENDING_BEAR"
            score = max(-1.0, kalman_slope * 100.0)
        else:
            regime = "TRENDING_CONSOLIDATING"
            score = 0.0
    else:
        regime = "RANGE_BOUND"
        score = 0.0

    return regime, round(score, 3)


# --------------------------------------------------------------------------- #
# 4. FULL 7-GROUP UNIFIED AGEX CONSENSUS ENGINE (Section 24, 30)
# --------------------------------------------------------------------------- #
@dataclass
class TechniqueOutputContract:
    """
    Standard Module Output Contract specified in Section 12 of
    `messege_fir_Antigravity ai.md`. Every technique reports this uniform schema.
    """
    technique_name: str
    layer: str  # OrderFlow, CandlePressure, Momentum, Volatility, MarketStructure, Statistical, ExecutionQuality
    timestamp: Optional[float]
    value: Any
    normalized_score: Optional[float]  # bounded [-1.0, +1.0]
    direction: str  # BULLISH, BEARISH, NEUTRAL
    valid: bool
    data_quality: str  # GOOD, DEGRADED, INVALID
    confidence: Optional[float]
    reason: str
    limitations: List[str]


def compute_vwap(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
) -> Tuple[float, float, str]:
    """
    Computes Intraday VWAP (Section 6.7).
    Returns (vwap_value, deviation_from_vwap, data_note).
    Explicit limitation: Uses MT5 tick volume, not centralized exchange trade volume.
    """
    if not highs or not volumes or len(highs) != len(volumes):
        cur_c = closes[-1] if closes else 0.0
        return cur_c, 0.0, "INSUFFICIENT_VOLUME_DATA"

    cum_pv = 0.0
    cum_v = 0.0
    for h, l, c, v in zip(highs, lows, closes, volumes):
        typical_price = (h + l + c) / 3.0
        vol = max(1.0, float(v))
        cum_pv += typical_price * vol
        cum_v += vol

    vwap = cum_pv / cum_v if cum_v > 0 else closes[-1]
    last_c = closes[-1]
    dev = last_c - vwap
    return round(vwap, 4), round(dev, 4), "Forex/CFD tick-volume proxy (not exchange centralized order volume)"


@dataclass
class FullConsensusResult:
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
        """Serializes exactly to Section 30 JSON output specification."""
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


class UnifiedAgexConsensusEngine:
    """
    Centralized 7-Group Institutional Prediction Engine for AGEX AI.
    Integrates all 21 techniques from `Multi-Technique Prediction.md`.
    """

    def __init__(
        self,
        weight_order_flow: float = 0.25,
        weight_candle_pressure: float = 0.25,
        weight_momentum: float = 0.20,
        weight_structure: float = 0.20,
        weight_regime: float = 0.10,
        score_threshold: float = 0.32,
        max_cost_ratio: float = 0.55,
        enable_exhaustion_guard: bool = True,
    ):
        self.w_of = float(weight_order_flow)
        self.w_cp = float(weight_candle_pressure)
        self.w_mo = float(weight_momentum)
        self.w_st = float(weight_structure)
        self.w_re = float(weight_regime)
        self.threshold = float(score_threshold)
        self.max_cost_ratio = float(max_cost_ratio)
        self.enable_exhaustion_guard = bool(enable_exhaustion_guard)
        self.kalman = KalmanPriceFilter1D()

    def analyze_full(
        self,
        symbol: str,
        last_4_m1_bars: pd.DataFrame,
        historical_bars: Optional[pd.DataFrame],
        tick_tracker: Any,
        ofi_calculator: Optional[Any],
        current_bid: float,
        current_ask: float,
    ) -> FullConsensusResult:
        from core.multi_technique_phase1 import (
            compute_candle_metrics,
            compute_momentum_consistency,
            compute_atr_and_relative_range,
            evaluate_execution_cost,
        )

        now_ts = time.time()
        reasons: List[str] = []

        if last_4_m1_bars is None or len(last_4_m1_bars) < 3:
            return FullConsensusResult(
                symbol=symbol,
                analysis_window=f"{len(last_4_m1_bars) if last_4_m1_bars is not None else 0} completed minutes",
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

        # ------------------------------------------------------------------- #
        # Group 1: Order Flow (STP + OFI)
        # ------------------------------------------------------------------- #
        stp_res = tick_tracker.compute_stp() if tick_tracker else {"stp": 0.0, "status": "NO_TRACKER"}
        raw_stp = float(stp_res.get("stp", 0.0))

        ofi_status = "UNAVAILABLE"
        ofi_metrics = {}
        if ofi_calculator is not None and hasattr(ofi_calculator, "compute_ofi_metrics"):
            ofi_metrics = ofi_calculator.compute_ofi_metrics()
            ofi_status = str(ofi_metrics.get("status", "UNAVAILABLE"))

        if ofi_status == "AVAILABLE":
            norm_ofi = float(ofi_metrics.get("normalized_ofi", 0.0))
            order_flow_score = min(1.0, max(-1.0, 0.50 * raw_stp + 0.50 * norm_ofi))
        else:
            order_flow_score = raw_stp

        # ------------------------------------------------------------------- #
        # Group 2: Candle Pressure
        # ------------------------------------------------------------------- #
        agg_open = float(last_4_m1_bars["open"].iloc[0])
        agg_close = float(last_4_m1_bars["close"].iloc[-1])
        agg_high = float(last_4_m1_bars["high"].max())
        agg_low = float(last_4_m1_bars["low"].min())
        agg_metrics = compute_candle_metrics(agg_open, agg_high, agg_low, agg_close)

        rejection_bias = agg_metrics.lower_rejection_ratio - agg_metrics.upper_rejection_ratio
        candle_score = min(1.0, max(-1.0, 0.40 * agg_metrics.signed_body + 0.35 * agg_metrics.clv + 0.25 * rejection_bias))

        # ------------------------------------------------------------------- #
        # Group 3: Momentum (MC, MACD - RSI Removed)
        # ------------------------------------------------------------------- #
        eval_bars = historical_bars if historical_bars is not None and len(historical_bars) >= 15 else last_4_m1_bars
        highs = [float(h) for h in eval_bars["high"]]
        lows = [float(l) for l in eval_bars["low"]]
        closes = [float(c) for c in eval_bars["close"]]

        all_4_closes = [float(c) for c in last_4_m1_bars["close"]]
        mc_res = compute_momentum_consistency(all_4_closes)
        mc_score = float(mc_res["mc"])

        macd_line, macd_sig, macd_hist = compute_macd(closes)
        macd_score = 1.0 if macd_hist > 0 else (-1.0 if macd_hist < 0 else 0.0)

        # Redistributed momentum weights: 70% MC, 30% MACD (sum = 1.00)
        momentum_score = min(1.0, max(-1.0, 0.70 * mc_score + 0.30 * macd_score))

        # ------------------------------------------------------------------- #
        # Group 4: Volatility & Execution Quality
        # ------------------------------------------------------------------- #
        vol_res = compute_atr_and_relative_range(highs, lows, closes, period=14)
        atr_val = float(vol_res["atr"])
        rel_range = float(vol_res["relative_range"])

        bb_mid, bb_upper, bb_lower, bb_width = compute_bollinger_bands(closes, period=20)

        cost_eval = evaluate_execution_cost(
            bid=current_bid,
            ask=current_ask,
            atr=atr_val,
            max_allowed_cost_ratio=self.max_cost_ratio,
        )

        # ------------------------------------------------------------------- #
        # Group 5: Market Structure (BOS, Pivots, Donchian, Fibs, VWAP)
        # ------------------------------------------------------------------- #
        struct_res = compute_market_structure(highs, lows, closes)
        structure_score = struct_res.structure_score

        vol_series = eval_bars.get("tick_volume", eval_bars.get("volume", [10.0] * len(eval_bars)))
        volumes = [float(v) for v in vol_series]
        vwap_val, vwap_dev, vwap_note = compute_vwap(highs, lows, closes, volumes)

        # ------------------------------------------------------------------- #
        # Group 6: Market Regime & Kalman Filter
        # ------------------------------------------------------------------- #
        pdi, mdi, adx_val = compute_adx(highs, lows, closes)
        kalman_price, kalman_slope = self.kalman.update(closes[-1])
        regime_name, regime_score = classify_market_regime(adx_val, rel_range, bb_width, kalman_slope)

        # ------------------------------------------------------------------- #
        # Step 3: Conflict Detection & Disagreements (Section 9.3)
        # ------------------------------------------------------------------- #
        disagreements: List[str] = []
        # Check OFI vs Candle Pressure
        if ofi_status == "AVAILABLE":
            norm_ofi = float(ofi_metrics.get("normalized_ofi", 0.0))
            if norm_ofi > 0.25 and candle_score < -0.20:
                disagreements.append("CONFLICT_OFI_BULLISH_VS_CANDLE_BEARISH")
            elif norm_ofi < -0.25 and candle_score > 0.20:
                disagreements.append("CONFLICT_OFI_BEARISH_VS_CANDLE_BULLISH")

        # Check Momentum vs Market Structure
        if momentum_score > 0.25 and structure_score < -0.20:
            disagreements.append("CONFLICT_MOMENTUM_BULLISH_VS_STRUCTURE_BEARISH")
        elif momentum_score < -0.25 and structure_score > 0.20:
            disagreements.append("CONFLICT_MOMENTUM_BEARISH_VS_STRUCTURE_BULLISH")

        # Check Tick Pressure vs Candle CLV / Closing Pressure
        if raw_stp > 0.30 and agg_metrics.clv < -0.30:
            disagreements.append("CONFLICT_TICK_PRESSURE_BULLISH_VS_CLV_BEARISH")
        elif raw_stp < -0.30 and agg_metrics.clv > 0.30:
            disagreements.append("CONFLICT_TICK_PRESSURE_BEARISH_VS_CLV_BULLISH")

        # ------------------------------------------------------------------- #
        # Weighted Consensus Aggregation
        # ------------------------------------------------------------------- #
        active_weights = {
            "of": self.w_of if stp_res.get("status") == "VALID" or ofi_status == "AVAILABLE" else 0.0,
            "cp": self.w_cp,
            "mo": self.w_mo,
            "st": self.w_st,
            "re": self.w_re,
        }
        total_w = sum(active_weights.values())
        if total_w <= 0:
            final_score = 0.0
        else:
            final_score = (
                (active_weights["of"] * order_flow_score)
                + (active_weights["cp"] * candle_score)
                + (active_weights["mo"] * momentum_score)
                + (active_weights["st"] * structure_score)
                + (active_weights["re"] * regime_score)
            ) / total_w

        final_score = min(1.0, max(-1.0, round(final_score, 4)))
        confidence = round(abs(final_score), 3)

        # ------------------------------------------------------------------- #
        # Step 3.5: Overextension / Resistance & Support Exhaustion Guard
        # Prevents buying at the top of large green candles (chasing tops) or
        # selling at the bottom of extended red runs (selling into support floors).
        # ------------------------------------------------------------------- #
        cur_mid = (current_bid + current_ask) / 2.0
        agg_total_range = max(agg_high - agg_low, 0.001)

        # 1. Bullish Overextension / Resistance Ceiling:
        # Traps happen when price hits resistance AND shows upper rejection (sellers pushing back)
        # OR when price is attempting to buy directly into unconfirmed resistance with upper wick.
        is_bull_exhausted = False
        if self.enable_exhaustion_guard:
            hit_resistance = (cur_mid >= struct_res.swing_high or cur_mid >= struct_res.pivot_r1)
            is_bull_exhausted = (
                (hit_resistance and agg_metrics.upper_rejection_ratio >= 0.20)
                or (struct_res.bos_signal == 0 and cur_mid >= (struct_res.swing_high - 0.20) and agg_metrics.upper_rejection_ratio >= 0.15)
                or (agg_metrics.body_strength >= 0.60 and agg_metrics.upper_rejection_ratio >= 0.25)
            )

        # 2. Bearish Overextension / Support Floor:
        # Traps happen when price hits support AND shows lower rejection (buyers pushing back)
        # OR when price is attempting to sell directly into unconfirmed support with lower wick.
        is_bear_exhausted = False
        if self.enable_exhaustion_guard:
            hit_support = (cur_mid <= struct_res.swing_low or cur_mid <= struct_res.pivot_s1)
            is_bear_exhausted = (
                (hit_support and agg_metrics.lower_rejection_ratio >= 0.20)
                or (struct_res.bos_signal == 0 and cur_mid <= (struct_res.swing_low + 0.20) and agg_metrics.lower_rejection_ratio >= 0.15)
                or (agg_metrics.body_strength >= 0.60 and agg_metrics.lower_rejection_ratio >= 0.25)
            )

        # Final Decision & Guard Rails
        decision = "NO_TRADE"
        risk_status = "PASS"

        if not cost_eval.passed:
            decision = "NO_TRADE"
            risk_status = "BLOCKED"
            reasons.append(cost_eval.reason)

        elif rel_range < 0.20:
            decision = "NO_TRADE"
            reasons.append(f"DEAD_MARKET_LOW_VOLATILITY_{rel_range:.2f}")

        elif agg_metrics.body_strength < 0.15 and agg_metrics.upper_rejection_ratio >= 0.25 and agg_metrics.lower_rejection_ratio >= 0.25:
            decision = "NO_TRADE"
            reasons.append("AGGREGATE_INDECISION_CHOP")

        elif disagreements:
            decision = "NO_TRADE"
            reasons.extend(disagreements)
            reasons.append("BLOCKED_DUE_TO_INTER_LAYER_CONFLICT")

        elif final_score >= self.threshold and candle_score > 0:
            if is_bull_exhausted:
                decision = "NO_TRADE"
                reasons.append("EXHAUSTION_GUARD_BULLISH_OVEREXTENDED_AT_RESISTANCE_WAIT_PULLBACK")
            else:
                decision = "BUY"
                reasons.append(f"CONSENSUS_BULLISH_SCORE_{final_score:+.2f}_REGIME_{regime_name}")

        elif final_score <= -self.threshold and candle_score < 0:
            if is_bear_exhausted:
                decision = "NO_TRADE"
                reasons.append("EXHAUSTION_GUARD_BEARISH_OVEREXTENDED_AT_SUPPORT_WAIT_PULLBACK")
            else:
                decision = "SELL"
                reasons.append(f"CONSENSUS_BEARISH_SCORE_{final_score:+.2f}_REGIME_{regime_name}")

        else:
            decision = "NO_TRADE"
            reasons.append(f"CONFLICTING_OR_SUBTHRESHOLD_SCORE_{final_score:+.2f}")

        # Section 30 JSON Schema feature_groups dictionary
        feature_groups = {
            "order_flow": {
                "stp": raw_stp,
                "order_flow_score": order_flow_score,
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
                "candle_score": candle_score,
            },
            "momentum": {
                "momentum_consistency": mc_score,
                "macd_histogram": macd_hist,
                "momentum_score": momentum_score,
            },
            "volatility": {
                "atr": atr_val,
                "relative_range": rel_range,
                "bollinger_bandwidth": bb_width,
            },
            "market_structure": {
                "trend_state": struct_res.trend_state,
                "bos_signal": struct_res.bos_signal,
                "swing_high": struct_res.swing_high,
                "swing_low": struct_res.swing_low,
                "pivot_p": struct_res.pivot_p,
                "donchian_mid": struct_res.donchian_mid,
                "vwap": vwap_val,
                "vwap_deviation": vwap_dev,
                "vwap_note": vwap_note,
                "structure_score": structure_score,
                "bull_exhausted": is_bull_exhausted,
                "bear_exhausted": is_bear_exhausted,
            },
            "market_regime": {
                "regime": regime_name,
                "adx": adx_val,
                "kalman_slope": kalman_slope,
                "regime_score": regime_score,
            },
            "execution_quality": {
                "spread": cost_eval.spread,
                "cost_ratio": cost_eval.cost_ratio,
                "passed": cost_eval.passed,
            },
            "disagreements": disagreements,
        }

        # Probabilities
        if decision == "BUY":
            probs = {"buy": round(0.50 + (confidence * 0.40), 2), "sell": round(0.50 - (confidence * 0.40), 2), "no_trade": 0.0}
        elif decision == "SELL":
            probs = {"buy": round(0.50 - (confidence * 0.40), 2), "sell": round(0.50 + (confidence * 0.40), 2), "no_trade": 0.0}
        else:
            probs = {"buy": 0.0, "sell": 0.0, "no_trade": 1.0}

        return FullConsensusResult(
            symbol=symbol,
            analysis_window=f"{len(last_4_m1_bars)} completed minutes",
            prediction_horizon="next 1 minute",
            decision=decision,
            directional_score=final_score,
            confidence_estimate=confidence,
            probabilities=probs,
            feature_groups=feature_groups,
            ofi_status=ofi_status,
            data_quality="GOOD" if stp_res.get("status") == "VALID" else "DEGRADED",
            expected_move=cost_eval.expected_move,
            current_spread=cost_eval.spread,
            estimated_trading_cost=cost_eval.estimated_trading_cost,
            reasons=reasons,
            risk_status=risk_status,
            signal_timestamp=now_ts,
        )
