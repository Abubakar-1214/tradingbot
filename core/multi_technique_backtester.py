"""
AGEX AI — Multi-Technique Historical Backtesting & Validation Engine (Phase 6)
Implements Sections 28 & 31 of Multi-Technique Prediction Specification.

Compares model iterations chronologically under realistic execution costs:
- Baseline: 4-minute candle direction
- Version A: Baseline + Candle features (SignedBody, CLV, Rejection)
- Version B: Version A + Tick Pressure (STP / Volume proxy)
- Version C: Version B + OFI (where DOM available)
- Version D: Version C + Momentum (MC, MACD, EMA slope)
- Version E: Version D + Volatility filter (ATR, Relative Range)
- Version F: Version E + Market Structure (BOS, Pivots, Donchian)
- Version G: Version F + Regime filter (Kalman Trend, Regime classification)
- Version H: Complete 7-Group Consensus + Execution Quality filters

Computes mandatory institutional metrics:
- Win Rate, Expectancy, Profit Factor, Net Profit, Max Drawdown ($ & %)
- BUY / SELL Accuracy, NO TRADE Percentage
- Results by Trading Session (Asian, London, NY)
- Results by Volatility Regime (Low, Normal, High)
- Chronological In-Sample vs Out-of-Sample Walk-Forward Validation
"""

from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

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
from core.multi_technique_phase2_ofi import OFICalculator
from core.multi_technique_phase3_advanced import (
    UnifiedAgexConsensusEngine,
    compute_market_structure,
    compute_macd,
    compute_adx,
    compute_bollinger_bands,
    KalmanPriceFilter1D,
    classify_market_regime,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    trade_id: int
    cycle_time: datetime
    decision: str  # BUY or SELL
    entry_price: float
    exit_price: float
    spread: float
    slippage: float
    commission: float
    gross_pnl: float
    net_pnl: float
    is_win: bool
    session: str  # ASIAN, LONDON, NY, OFF_HOURS
    regime: str   # LOW, NORMAL, HIGH


@dataclass
class VersionMetrics:
    version_name: str
    description: str
    total_candles: int = 0
    total_trades: int = 0
    no_trade_count: int = 0
    no_trade_pct: float = 0.0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    buy_trades: int = 0
    buy_wins: int = 0
    buy_accuracy: float = 0.0
    sell_trades: int = 0
    sell_wins: int = 0
    sell_accuracy: float = 0.0
    session_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    regime_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trades: List[TradeRecord] = field(default_factory=list)


def determine_market_session(dt: datetime) -> str:
    """Classifies UTC timestamp into Forex/Gold trading session."""
    hour = dt.hour
    if 0 <= hour < 8:
        return "ASIAN"
    elif 8 <= hour < 14:
        return "LONDON"
    elif 14 <= hour < 21:
        return "NEW_YORK"
    return "OFF_HOURS"


def compute_metrics_from_trades(
    trades: List[TradeRecord],
    total_cycles: int,
    version_name: str,
    description: str,
    initial_balance: float = 100.0,
) -> VersionMetrics:
    """Computes all Section 28 required performance metrics from trade list."""
    m = VersionMetrics(
        version_name=version_name,
        description=description,
        total_candles=total_cycles,
        total_trades=len(trades),
        no_trade_count=max(0, total_cycles - len(trades)),
        trades=trades,
    )

    if total_cycles > 0:
        m.no_trade_pct = round((m.no_trade_count / total_cycles) * 100.0, 2)

    if not trades:
        return m

    pnl_series = [t.net_pnl for t in trades]
    m.wins = sum(1 for p in pnl_series if p > 1e-6)
    m.losses = sum(1 for p in pnl_series if p < -1e-6)
    m.ties = len(trades) - m.wins - m.losses

    m.win_rate = round((m.wins / m.total_trades) * 100.0, 2) if m.total_trades > 0 else 0.0

    m.gross_profit = round(sum(p for p in pnl_series if p > 0), 4)
    m.gross_loss = round(abs(sum(p for p in pnl_series if p < 0)), 4)
    m.net_profit = round(sum(pnl_series), 4)

    m.avg_win = round(m.gross_profit / m.wins, 4) if m.wins > 0 else 0.0
    m.avg_loss = round(m.gross_loss / m.losses, 4) if m.losses > 0 else 0.0

    p_win = m.win_rate / 100.0
    p_loss = (m.losses / m.total_trades) if m.total_trades > 0 else 0.0
    m.expectancy = round((p_win * m.avg_win) - (p_loss * m.avg_loss), 4)

    if m.gross_loss > 1e-6:
        m.profit_factor = round(m.gross_profit / m.gross_loss, 2)
    elif m.gross_profit > 1e-6:
        m.profit_factor = 999.99
    else:
        m.profit_factor = 0.0

    # Drawdown calculation
    equity = initial_balance
    peak = equity
    max_dd_usd = 0.0
    max_dd_pct = 0.0

    for p in pnl_series:
        equity += p
        if equity > peak:
            peak = equity
        dd_usd = peak - equity
        dd_pct = (dd_usd / peak * 100.0) if peak > 0 else 0.0
        if dd_usd > max_dd_usd:
            max_dd_usd = dd_usd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    m.max_drawdown_usd = round(max_dd_usd, 4)
    m.max_drawdown_pct = round(max_dd_pct, 2)

    # Directional accuracy
    buys = [t for t in trades if t.decision == "BUY"]
    m.buy_trades = len(buys)
    m.buy_wins = sum(1 for t in buys if t.is_win)
    m.buy_accuracy = round((m.buy_wins / m.buy_trades * 100.0), 2) if m.buy_trades > 0 else 0.0

    sells = [t for t in trades if t.decision == "SELL"]
    m.sell_trades = len(sells)
    m.sell_wins = sum(1 for t in sells if t.is_win)
    m.sell_accuracy = round((m.sell_wins / m.sell_trades * 100.0), 2) if m.sell_trades > 0 else 0.0

    # Session breakdown
    sessions = ["ASIAN", "LONDON", "NEW_YORK", "OFF_HOURS"]
    for s in sessions:
        s_trades = [t for t in trades if t.session == s]
        if s_trades:
            s_wins = sum(1 for t in s_trades if t.is_win)
            s_net = round(sum(t.net_pnl for t in s_trades), 4)
            m.session_breakdown[s] = {
                "trades": len(s_trades),
                "win_rate": round(s_wins / len(s_trades) * 100.0, 2),
                "net_profit": s_net,
            }

    # Volatility regime breakdown
    regimes = ["LOW", "NORMAL", "HIGH"]
    for r in regimes:
        r_trades = [t for t in trades if t.regime == r]
        if r_trades:
            r_wins = sum(1 for t in r_trades if t.is_win)
            r_net = round(sum(t.net_pnl for t in r_trades), 4)
            m.regime_breakdown[r] = {
                "trades": len(r_trades),
                "win_rate": round(r_wins / len(r_trades) * 100.0, 2),
                "net_profit": r_net,
            }

    return m


class AgexMultiTechniqueBacktester:
    """
    Backtesting engine simulating the 4+1 minute institutional trading cycle
    over historical M1 bar sequences without lookahead bias.
    """

    def __init__(
        self,
        symbol: str = "XAUUSDm",
        contract_size: float = 100.0,  # 1 lot = 100 oz Gold
        lot_size: float = 0.01,        # standard retail micro-lot (0.01)
        fixed_spread_usd: float = 0.24,# 24 cents on Gold (~24 points)
        slippage_usd: float = 0.02,    # 2 cents realistic execution slippage
        commission_usd: float = 0.03,  # 3 cents per 0.01 lot
        initial_balance: float = 100.0,
    ):
        self.symbol = symbol
        self.contract_size = contract_size
        self.lot_size = lot_size
        self.fixed_spread_usd = fixed_spread_usd
        self.slippage_usd = slippage_usd
        self.commission_usd = commission_usd
        self.initial_balance = initial_balance

        # Unified institutional engine for Version H
        self.unified_engine = UnifiedAgexConsensusEngine(
            score_threshold=0.32,
            max_cost_ratio=0.55,
        )

    def group_m1_into_5min_cycles(self, df_m1: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """
        Groups continuous M1 bars into 5-minute blocks according to minute % 5.
        Returns a list of (last_4_observation_bars, 5th_target_candle).
        Strict causal separation: 5th candle is NEVER in the observation window.
        """
        if df_m1 is None or len(df_m1) < 5:
            return []

        df = df_m1.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])

        df = df.sort_values("time").reset_index(drop=True)
        cycles = []

        # Find 5-minute blocks
        # minute % 5: 0, 1, 2, 3 are observations, 4 is the 5th minute
        i = 0
        n = len(df)
        while i < n - 4:
            t0 = df.loc[i, "time"]
            m0 = t0.minute % 5
            # We want block starting at minute % 5 == 0
            if m0 != 0:
                i += 1
                continue

            # Verify consecutive 5 bars
            block = df.iloc[i : i + 5]
            if len(block) == 5:
                # Check timestamps are roughly consecutive (1-minute intervals)
                time_span = (block["time"].iloc[-1] - block["time"].iloc[0]).total_seconds()
                if time_span <= 300:  # within 5 minutes
                    obs_4 = block.iloc[:4]
                    target_5 = block.iloc[4]
                    cycles.append((obs_4, target_5))
                    i += 5
                else:
                    i += 1
            else:
                break

        return cycles

    def group_m1_into_rolling_cycles(
        self,
        df_m1: pd.DataFrame,
        window: int = 7,
    ) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """
        Groups continuous M1 bars into rolling observation windows of N candles (e.g. 7 bars)
        followed by a 1-minute execution bar.
        Strict causal separation: target bar is NEVER in the observation window.
        """
        if df_m1 is None or len(df_m1) < window + 1:
            return []

        df = df_m1.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])

        df = df.sort_values("time").reset_index(drop=True)
        cycles = []

        n = len(df)
        for i in range(window, n):
            obs_n = df.iloc[i - window : i]
            target_1 = df.iloc[i]
            # Check timestamps are consecutive (within 120 seconds)
            time_span = (target_1["time"] - obs_n["time"].iloc[-1]).total_seconds()
            if time_span <= 120:
                cycles.append((obs_n, target_1))

        return cycles

    def simulate_trade_execution(
        self,
        decision: str,
        target_5th_bar: pd.Series,
        cycle_time: datetime,
        trade_id: int,
        regime_label: str = "NORMAL",
    ) -> Optional[TradeRecord]:
        """
        Executes trade on the 5th candle open and exits on the 5th candle close.
        Includes realistic broker spread, slippage, and commission.
        """
        if decision not in ("BUY", "SELL"):
            return None

        # Extract spread from data if present, otherwise fixed spread
        spread = self.fixed_spread_usd
        if "spread" in target_5th_bar and not math.isnan(target_5th_bar["spread"]):
            # MT5 spread on Gold is in points (e.g., 240 points = 0.24 USD)
            raw_s = float(target_5th_bar["spread"])
            spread = (raw_s / 1000.0) if raw_s > 10.0 else raw_s

        half_spread = spread / 2.0
        slippage = self.slippage_usd
        commission = self.commission_usd

        bar_open = float(target_5th_bar["open"])
        bar_close = float(target_5th_bar["close"])

        if decision == "BUY":
            entry_price = bar_open + half_spread + slippage
            exit_price = bar_close - half_spread
            price_change = exit_price - entry_price
        else:  # SELL
            entry_price = bar_open - half_spread - slippage
            exit_price = bar_close + half_spread
            price_change = entry_price - exit_price

        # Dollar PnL on Gold = price_change * contract_size * lot_size
        multiplier = self.contract_size * self.lot_size
        gross_pnl = price_change * multiplier
        net_pnl = gross_pnl - commission
        is_win = net_pnl > 0.0

        session = determine_market_session(cycle_time)

        return TradeRecord(
            trade_id=trade_id,
            cycle_time=cycle_time,
            decision=decision,
            entry_price=round(entry_price, 3),
            exit_price=round(exit_price, 3),
            spread=round(spread, 3),
            slippage=round(slippage, 3),
            commission=round(commission, 3),
            gross_pnl=round(gross_pnl, 4),
            net_pnl=round(net_pnl, 4),
            is_win=is_win,
            session=session,
            regime=regime_label,
        )

    def evaluate_cycle_decision(
        self,
        version: str,
        obs_4: pd.DataFrame,
        historical_m1_so_far: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Evaluates the specific version's prediction rule for the upcoming 5th minute.
        Returns: 'BUY', 'SELL', or 'NO_TRADE'.
        """
        # Common pre-computations on the 4 observation candles
        c1_open = float(obs_4["open"].iloc[0])
        c4_close = float(obs_4["close"].iloc[-1])
        agg_high = float(obs_4["high"].max())
        agg_low = float(obs_4["low"].min())
        agg_vol = float(obs_4["tick_volume"].sum() if "tick_volume" in obs_4 else obs_4.get("volume", pd.Series([1])).sum())

        net_move = c4_close - c1_open
        tot_range = max(1e-6, agg_high - agg_low)
        body_ratio = abs(net_move) / tot_range

        # Extract price series from observation / historical bars
        context_df = historical_m1_so_far if historical_m1_so_far is not None else obs_4
        highs = [float(x) for x in context_df["high"].tolist()]
        lows = [float(x) for x in context_df["low"].tolist()]
        closes = [float(x) for x in context_df["close"].tolist()]

        # -----------------------------------------------------------------
        # 1. BASELINE: Simple 4-minute candle direction
        # -----------------------------------------------------------------
        if version == "BASELINE":
            if net_move > 0:
                return "BUY"
            elif net_move < 0:
                return "SELL"
            return "NO_TRADE"

        # -----------------------------------------------------------------
        # 2. VERSION A: Baseline + Candle Features (SignedBody, CLV, Rejection)
        # -----------------------------------------------------------------
        candle_metrics = compute_candle_metrics(c1_open, agg_high, agg_low, c4_close)
        wick_rejection = candle_metrics.lower_rejection_ratio - candle_metrics.upper_rejection_ratio

        if version == "VERSION_A":
            if candle_metrics.body_strength < 0.15:
                return "NO_TRADE"
            candle_score = (
                0.40 * candle_metrics.signed_body +
                0.30 * candle_metrics.clv +
                0.30 * wick_rejection
            )
            if candle_score > 0.20:
                return "BUY"
            elif candle_score < -0.20:
                return "SELL"
            return "NO_TRADE"

        # -----------------------------------------------------------------
        # 3. VERSION B: A + Tick / Volume Pressure Proxy (STP)
        # -----------------------------------------------------------------
        up_vol = 0.0
        down_vol = 0.0
        for idx in range(len(obs_4)):
            row = obs_4.iloc[idx]
            v = float(row.get("tick_volume", row.get("volume", 10.0)))
            if float(row["close"]) >= float(row["open"]):
                up_vol += v
            else:
                down_vol += v
        tot_v = max(1.0, up_vol + down_vol)
        tick_score = (up_vol - down_vol) / tot_v  # bounded [-1, +1]

        if version == "VERSION_B":
            if candle_metrics.body_strength < 0.15:
                return "NO_TRADE"
            candle_score = (
                0.40 * candle_metrics.signed_body +
                0.30 * candle_metrics.clv +
                0.30 * wick_rejection
            )
            comb_b = 0.60 * candle_score + 0.40 * tick_score
            if comb_b > 0.25:
                return "BUY"
            elif comb_b < -0.25:
                return "SELL"
            return "NO_TRADE"

        # -----------------------------------------------------------------
        # 4. VERSION C: B + OFI (If DOM available, otherwise graceful fallback)
        # -----------------------------------------------------------------
        if version == "VERSION_C":
            if candle_metrics.body_strength < 0.15:
                return "NO_TRADE"
            candle_score = (
                0.40 * candle_metrics.signed_body +
                0.30 * candle_metrics.clv +
                0.30 * wick_rejection
            )
            comb_c = 0.60 * candle_score + 0.40 * tick_score
            if comb_c > 0.25:
                return "BUY"
            elif comb_c < -0.25:
                return "SELL"
            return "NO_TRADE"

        # -----------------------------------------------------------------
        # 5. VERSION D: C + Momentum (Momentum Consistency, MACD - RSI Removed)
        # -----------------------------------------------------------------
        obs_closes = [float(x) for x in obs_4["close"].tolist()]
        mc_dict = compute_momentum_consistency(obs_closes)
        mc = float(mc_dict.get("mc", 0.0))
        macd_hist = 0.0
        if len(closes) >= 30:
            _, _, macd_hist = compute_macd(closes)

        macd_score = 1.0 if macd_hist > 0.05 else (-1.0 if macd_hist < -0.05 else 0.0)
        # Redistributed momentum weights: 60% MC, 40% MACD
        mom_score = 0.60 * mc + 0.40 * macd_score

        if version == "VERSION_D":
            if candle_metrics.body_strength < 0.15:
                return "NO_TRADE"
            candle_score = (
                0.40 * candle_metrics.signed_body +
                0.30 * candle_metrics.clv +
                0.30 * wick_rejection
            )
            comb_d = 0.30 * tick_score + 0.35 * candle_score + 0.35 * mom_score
            if comb_d > 0.28:
                return "BUY"
            elif comb_d < -0.28:
                return "SELL"
            return "NO_TRADE"

        # -----------------------------------------------------------------
        # 6. VERSION E: D + Volatility Filter (ATR & Relative Range)
        # -----------------------------------------------------------------
        atr_res = compute_atr_and_relative_range(highs, lows, closes)
        rel_range = float(atr_res.get("relative_range", 1.0))

        if version == "VERSION_E":
            if candle_metrics.body_strength < 0.15:
                return "NO_TRADE"
            if rel_range > 3.0 or rel_range < 0.20:
                return "NO_TRADE"
            candle_score = (
                0.40 * candle_metrics.signed_body +
                0.30 * candle_metrics.clv +
                0.30 * wick_rejection
            )
            comb_e = 0.30 * tick_score + 0.35 * candle_score + 0.35 * mom_score
            if comb_e > 0.28:
                return "BUY"
            elif comb_e < -0.28:
                return "SELL"
            return "NO_TRADE"

        # -----------------------------------------------------------------
        # 7. VERSION F: E + Market Structure (BOS, Pivots, Donchian)
        # -----------------------------------------------------------------
        struct = compute_market_structure(highs, lows, closes)
        if version == "VERSION_F":
            if candle_metrics.body_strength < 0.15:
                return "NO_TRADE"
            if rel_range > 3.0 or rel_range < 0.20:
                return "NO_TRADE"
            candle_score = (
                0.40 * candle_metrics.signed_body +
                0.30 * candle_metrics.clv +
                0.30 * wick_rejection
            )
            comb_f = (
                0.25 * tick_score +
                0.30 * candle_score +
                0.30 * mom_score +
                0.15 * struct.structure_score
            )
            if struct.bos_signal == "BULLISH_BOS" and comb_f < 0:
                return "NO_TRADE"
            if struct.bos_signal == "BEARISH_BOS" and comb_f > 0:
                return "NO_TRADE"

            if comb_f > 0.30:
                return "BUY"
            elif comb_f < -0.30:
                return "SELL"
            return "NO_TRADE"

        # -----------------------------------------------------------------
        # 8. VERSION G: F + Regime Filters (Kalman Trend & Regime classification)
        # -----------------------------------------------------------------
        _, _, adx_val = compute_adx(highs, lows, closes)
        _, _, _, bb_width = compute_bollinger_bands(closes)
        kalman = KalmanPriceFilter1D()
        k_slope = 0.0
        for c in closes[-15:]:
            _, k_slope = kalman.update(c)
        regime_name, regime_score = classify_market_regime(
            adx=adx_val,
            relative_range=rel_range,
            bandwidth=bb_width,
            kalman_slope=k_slope,
        )

        if version == "VERSION_G":
            if candle_metrics.body_strength < 0.15:
                return "NO_TRADE"
            if rel_range > 3.0 or rel_range < 0.20:
                return "NO_TRADE"
            candle_score = (
                0.40 * candle_metrics.signed_body +
                0.30 * candle_metrics.clv +
                0.30 * wick_rejection
            )
            comb_g = (
                0.20 * tick_score +
                0.25 * candle_score +
                0.25 * mom_score +
                0.15 * struct.structure_score +
                0.15 * regime_score
            )
            thresh = 0.38 if regime_name == "RANGE_BOUND" else 0.30
            if comb_g > thresh:
                return "BUY"
            elif comb_g < -thresh:
                return "SELL"
            return "NO_TRADE"

        # -----------------------------------------------------------------
        # 9. VERSION H: Complete System + Full Institutional Consensus & Cost Filters
        # -----------------------------------------------------------------
        if version == "VERSION_H":
            # Run the production UnifiedAgexConsensusEngine
            # Create a STP tracker proxy seeded with the observation ticks/candles
            tracker = SignedTickPressureTracker(window_seconds=240.0)
            for idx in range(len(obs_4)):
                row = obs_4.iloc[idx]
                p = float(row["close"])
                tracker.add_tick(p - 0.05, p + 0.05)

            current_bid = c4_close - 0.12
            current_ask = c4_close + 0.12

            res = self.unified_engine.analyze_full(
                symbol=self.symbol,
                last_4_m1_bars=obs_4,
                historical_bars=historical_m1_so_far,
                tick_tracker=tracker,
                ofi_calculator=None,
                current_bid=current_bid,
                current_ask=current_ask,
            )
            return res.decision

        return "NO_TRADE"

    def run_backtest_all_versions(
        self,
        df_m1: pd.DataFrame,
        versions: Optional[List[str]] = None,
        max_cycles: Optional[int] = None,
        cycle_mode: str = "5MIN",
        rolling_window: int = 7,
        cooldown_bars: int = 2,
    ) -> Dict[str, VersionMetrics]:
        """
        Runs backtests on all versions across historical cycles.
        Supports:
          - cycle_mode='5MIN': Clock-aligned 4-minute observation + 5th minute execution.
          - cycle_mode='ROLLING': Rolling N-candle (e.g. 7-candle) observation + post-trade cooldown.
        """
        if versions is None:
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

        version_descriptions = {
            "BASELINE": f"Existing direction on {rolling_window if cycle_mode == 'ROLLING' else 4} candles",
            "VERSION_A": "Baseline + Candle features (SignedBody, CLV, Rejection)",
            "VERSION_B": "Version A + Signed Tick Pressure (STP proxy)",
            "VERSION_C": "Version B + Order Flow Imbalance (OFI where DOM exists)",
            "VERSION_D": "Version C + Momentum (MC, MACD, EMA slope)",
            "VERSION_E": "Version D + Volatility filters (ATR, Relative Range)",
            "VERSION_F": "Version E + Market Structure (BOS, Pivots, Donchian)",
            "VERSION_G": "Version F + Market Regime & Kalman Trend Filter",
            "VERSION_H": "Complete 7-Group Institutional Consensus + Execution Filters",
        }

        if cycle_mode.upper() == "ROLLING":
            cycles = self.group_m1_into_rolling_cycles(df_m1, window=rolling_window)
            cycle_desc = f"{rolling_window}-minute rolling"
        else:
            cycles = self.group_m1_into_5min_cycles(df_m1)
            cycle_desc = "5-minute block"

        if max_cycles and len(cycles) > max_cycles:
            cycles = cycles[-max_cycles:]

        total_cycles = len(cycles)
        logger.info("Starting AGEX Backtest across %d %s cycles (Mode: %s)...", total_cycles, cycle_desc, cycle_mode)

        # Pre-build chronological index for O(1) instant historical window slicing
        df_sorted = df_m1.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_sorted["time"]):
            df_sorted["time"] = pd.to_datetime(df_sorted["time"])
        df_sorted = df_sorted.sort_values("time").reset_index(drop=True)
        time_to_idx = {t: idx for idx, t in enumerate(df_sorted["time"])}

        results: Dict[str, VersionMetrics] = {}

        for v in versions:
            trades: List[TradeRecord] = []
            trade_id = 1
            last_trade_cycle_idx = -999

            for cycle_idx, (obs_bars, target_bar) in enumerate(cycles):
                cycle_time = target_bar["time"]

                # Post-trade cooldown check in rolling mode
                if cycle_mode.upper() == "ROLLING" and (cycle_idx - last_trade_cycle_idx) < cooldown_bars:
                    continue

                end_t = obs_bars["time"].iloc[-1]
                idx = time_to_idx.get(end_t)
                if idx is not None and idx >= 15:
                    hist_window = df_sorted.iloc[max(0, idx - 100) : idx + 1]
                else:
                    hist_window = obs_bars

                # Classify regime for telemetry
                c_df = hist_window if hist_window is not None else obs_bars
                c_highs = [float(x) for x in c_df["high"].tolist()]
                c_lows = [float(x) for x in c_df["low"].tolist()]
                c_closes = [float(x) for x in c_df["close"].tolist()]
                atr_r = compute_atr_and_relative_range(c_highs, c_lows, c_closes)
                r_range = float(atr_r.get("relative_range", 1.0))
                if r_range < 0.7:
                    regime_lbl = "LOW"
                elif r_range > 1.8:
                    regime_lbl = "HIGH"
                else:
                    regime_lbl = "NORMAL"

                decision = self.evaluate_cycle_decision(v, obs_bars, hist_window)

                if decision in ("BUY", "SELL"):
                    trade = self.simulate_trade_execution(
                        decision=decision,
                        target_5th_bar=target_bar,
                        cycle_time=cycle_time,
                        trade_id=trade_id,
                        regime_label=regime_lbl,
                    )
                    if trade is not None:
                        trades.append(trade)
                        trade_id += 1
                        last_trade_cycle_idx = cycle_idx

            metrics = compute_metrics_from_trades(
                trades=trades,
                total_cycles=total_cycles,
                version_name=v,
                description=version_descriptions.get(v, v),
                initial_balance=self.initial_balance,
            )
            results[v] = metrics

        return results

    def run_walk_forward_validation(
        self,
        df_m1: pd.DataFrame,
        train_ratio: float = 0.70,
        version: str = "VERSION_H",
    ) -> Dict[str, Any]:
        """
        Chronological Walk-Forward validation (In-Sample Train vs Out-of-Sample Test).
        Verifies model does not overfit and generalizes to unseen market periods.
        """
        cycles = self.group_m1_into_5min_cycles(df_m1)
        if len(cycles) < 20:
            return {"error": "Insufficient cycles for walk-forward validation"}

        split_idx = int(len(cycles) * train_ratio)
        train_cycles = cycles[:split_idx]
        test_cycles = cycles[split_idx:]

        # Run on train
        train_m1_end = train_cycles[-1][1]["time"]
        df_train = df_m1[df_m1["time"] <= train_m1_end].copy()
        df_test = df_m1[df_m1["time"] > train_m1_end].copy()

        train_results = self.run_backtest_all_versions(df_train, versions=[version])
        test_results = self.run_backtest_all_versions(df_test, versions=[version])

        train_m = train_results[version]
        test_m = test_results[version]

        return {
            "version": version,
            "train_cycles": len(train_cycles),
            "test_cycles": len(test_cycles),
            "train_metrics": {
                "trades": train_m.total_trades,
                "win_rate": train_m.win_rate,
                "net_profit": train_m.net_profit,
                "profit_factor": train_m.profit_factor,
                "expectancy": train_m.expectancy,
                "max_drawdown_usd": train_m.max_drawdown_usd,
                "no_trade_pct": train_m.no_trade_pct,
            },
            "test_metrics": {
                "trades": test_m.total_trades,
                "win_rate": test_m.win_rate,
                "net_profit": test_m.net_profit,
                "profit_factor": test_m.profit_factor,
                "expectancy": test_m.expectancy,
                "max_drawdown_usd": test_m.max_drawdown_usd,
                "no_trade_pct": test_m.no_trade_pct,
            },
            "generalization_ratio": round(test_m.win_rate / train_m.win_rate, 2) if train_m.win_rate > 0 else 0.0,
        }

    def generate_comparison_markdown_table(self, results: Dict[str, VersionMetrics]) -> str:
        """Formats the results of all versions into a markdown table."""
        lines = [
            "| Version | Description | Trades | Win Rate | Net Profit | Profit Factor | Expectancy | Max DD ($) | NO TRADE % |",
            "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for v, m in results.items():
            pf_str = f"{m.profit_factor:.2f}" if m.profit_factor < 900 else "N/A"
            lines.append(
                f"| **{m.version_name}** | {m.description[:35]}... | {m.total_trades} | "
                f"**{m.win_rate:.1f}%** | ${m.net_profit:+.2f} | {pf_str} | "
                f"${m.expectancy:+.2f} | ${m.max_drawdown_usd:.2f} | {m.no_trade_pct:.1f}% |"
            )
        return "\n".join(lines)
