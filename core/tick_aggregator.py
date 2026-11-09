"""
AGEX AI — Tick-Based Aggregation & Microstructure Feature Engine.

Implements the institutional 23-formula specification defined in
`Tick-Based Aggregation.md` for AGEX AI:
  Stage 1: Incoming Tick Preprocessing (MidPrice, Spread, Validation)
  Stage 2: Tick-Based Candle Construction (K-Tick OHLCV)
  Stage 3: Completed Candle Features (Range, Body, Direction, Wicks, Duration, TickRate, Spreads)
  Stage 4: Rolling Market Intelligence (AvgRange, AvgTickRate, Volatility, Acceleration, NormalizedRange)
  Stage 5: High-Precision Price Action & Signal Classification
"""

from dataclasses import dataclass, field
import math
import time
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AgexTick:
    """Stage 1: Processed and validated tick."""
    timestamp: float
    bid: float
    ask: float
    mid: float
    spread: float


@dataclass
class AgexTickCandle:
    """Stage 2 & 3: Completed K-tick candlestick with rich microstructure features."""
    index: int
    start_timestamp: float
    end_timestamp: float
    tick_count: int
    open: float
    high: float
    low: float
    close: float
    range: float
    body: float
    direction: int  # +1 Bullish, -1 Bearish, 0 Neutral
    upper_wick: float
    lower_wick: float
    duration_seconds: float
    tick_rate: float
    average_spread: float
    maximum_spread: float
    body_position_pct: float
    pattern: str = "UNKNOWN"
    action: int = 0  # 1 BUY, 2 SELL, 0 WAIT


@dataclass
class AgexRollingFeatures:
    """Stage 4: Rolling dynamic market intelligence features."""
    candle_index: int
    rolling_average_range: float
    rolling_average_tick_rate: float
    rolling_volatility: float
    tick_rate_change: float
    normalized_range: float


class AgexTickAggregator:
    """
    Core Tick-Based Aggregation Engine for AGEX AI.
    Groups raw ticks into discrete candles based on fixed tick counts (K)
    and computes institutional price action and volatility features.
    """

    def __init__(self, ticks_per_candle: int = 100, rolling_window_size: int = 20):
        self.k = max(2, int(ticks_per_candle))
        self.m = max(2, int(rolling_window_size))

        # Current in-progress candle accumulators
        self._current_ticks: List[AgexTick] = []
        self._last_tick_time: float = 0.0

        # Completed history
        self.completed_candles: List[AgexTickCandle] = []
        self.rolling_features: List[AgexRollingFeatures] = []
        self.total_accepted_ticks: int = 0

    # ----------------------------------------------------------------------- #
    # STAGE 1: Incoming Tick Preprocessing (Formulas 1, 2)
    # ----------------------------------------------------------------------- #
    def process_raw_tick(
        self, bid: float, ask: float, timestamp: Optional[float] = None
    ) -> Optional[AgexTick]:
        """
        Validates raw tick and computes Mid Price and Spread.
        Formula 1: MidPrice = (Bid + Ask) / 2
        Formula 2: Spread = Ask - Bid
        """
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            return None  # Invalid quote

        ts = float(timestamp if timestamp is not None else time.time())

        # Duplicate or backwards tick check (Engineering Rule 7 & 8)
        if ts < self._last_tick_time:
            ts = self._last_tick_time + 0.0001  # Safe monotonic adjustment

        mid = (bid + ask) / 2.0
        spread = ask - bid
        self._last_tick_time = ts

        return AgexTick(
            timestamp=ts,
            bid=round(bid, 5),
            ask=round(ask, 5),
            mid=round(mid, 5),
            spread=round(spread, 5),
        )

    # ----------------------------------------------------------------------- #
    # STAGE 2 & 3: Tick Aggregation & Candle Features (Formulas 3-18)
    # ----------------------------------------------------------------------- #
    def add_tick(
        self, bid: float, ask: float, timestamp: Optional[float] = None
    ) -> Optional[AgexTickCandle]:
        """
        Ingests a tick and returns a completed AgexTickCandle if TickCount >= K.
        """
        tick = self.process_raw_tick(bid, ask, timestamp)
        if tick is None:
            return None

        self._current_ticks.append(tick)
        self.total_accepted_ticks += 1

        # Formula 4: CandleComplete = (TickCount >= K)
        if len(self._current_ticks) >= self.k:
            candle = self._finalize_current_candle()
            self._current_ticks.clear()
            return candle

        return None

    def _finalize_current_candle(self) -> AgexTickCandle:
        """Computes all Stage 2 & Stage 3 formulas on the completed K-tick block."""
        ticks = self._current_ticks
        k = len(ticks)

        # Formula 6, 7, 8, 9: OHLC on Mid Prices
        open_p = ticks[0].mid
        close_p = ticks[-1].mid
        high_p = max(t.mid for t in ticks)
        low_p = min(t.mid for t in ticks)

        # Formula 10: Range = H - L
        c_range = max(high_p - low_p, 0.001)

        # Formula 11: Body = |C - O|
        body = abs(close_p - open_p)

        # Formula 12: Direction
        if close_p > open_p:
            direction = 1
        elif close_p < open_p:
            direction = -1
        else:
            direction = 0

        # Formula 13: Upper Wick = H - max(O, C)
        upper_wick = high_p - max(open_p, close_p)

        # Formula 14: Lower Wick = min(O, C) - L
        lower_wick = min(open_p, close_p) - low_p

        # Formula 15: Duration = EndTimestamp - StartTimestamp
        start_ts = ticks[0].timestamp
        end_ts = ticks[-1].timestamp
        duration = max(end_ts - start_ts, 0.0001)

        # Formula 16: TickRate = TickCount / DurationSeconds
        tick_rate = k / duration if duration > 0.0 else 0.0

        # Formula 17: AverageSpread = (sum Spread(i)) / K
        avg_spread = sum(t.spread for t in ticks) / k

        # Formula 18: MaximumSpread = max(Spread(1), ..., Spread(K))
        max_spread = max(t.spread for t in ticks)

        # AGEX Video Principle: Body position relative to High and Low
        body_center = (open_p + close_p) / 2.0
        body_position_pct = ((body_center - low_p) / c_range) * 100.0

        # Price Action Pattern Classification
        body_pct = (body / c_range) * 100.0
        upper_pct = (upper_wick / c_range) * 100.0
        lower_pct = (lower_wick / c_range) * 100.0

        pattern, action = self._classify_candle(
            direction, body_pct, upper_pct, lower_pct, lower_wick, upper_wick, body, c_range
        )

        candle_idx = len(self.completed_candles) + 1
        candle = AgexTickCandle(
            index=candle_idx,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            tick_count=k,
            open=round(open_p, 3),
            high=round(high_p, 3),
            low=round(low_p, 3),
            close=round(close_p, 3),
            range=round(c_range, 3),
            body=round(body, 3),
            direction=direction,
            upper_wick=round(upper_wick, 3),
            lower_wick=round(lower_wick, 3),
            duration_seconds=round(duration, 3),
            tick_rate=round(tick_rate, 2),
            average_spread=round(avg_spread, 3),
            maximum_spread=round(max_spread, 3),
            body_position_pct=round(body_position_pct, 1),
            pattern=pattern,
            action=action,
        )

        self.completed_candles.append(candle)
        self._compute_rolling_features(candle)

        return candle

    # ----------------------------------------------------------------------- #
    # STAGE 4: Rolling Market Intelligence (Formulas 19-23)
    # ----------------------------------------------------------------------- #
    def _compute_rolling_features(self, current_candle: AgexTickCandle) -> AgexRollingFeatures:
        """
        Computes Formulas 19-23 over the rolling M-window of completed candles.
        Prevents look-ahead leakage (Engineering Rule 9).
        """
        history = self.completed_candles
        m = min(self.m, len(history))

        recent = history[-m:]

        # Formula 19: Rolling Average Range = (1 / M) * sum Range(t-j)
        avg_range = sum(c.range for c in recent) / m

        # Formula 20: Rolling Average Tick Rate = (1 / M) * sum TickRate(t-j)
        avg_tick_rate = sum(c.tick_rate for c in recent) / m

        # Formula 21: Rolling Volatility = std_dev(ln(Close(t) / Close(t-1)))
        volatility = 0.0
        if len(recent) >= 2:
            log_returns = []
            for i in range(1, len(recent)):
                prev_c = recent[i - 1].close
                cur_c = recent[i].close
                if prev_c > 0 and cur_c > 0:
                    log_returns.append(math.log(cur_c / prev_c))
            if len(log_returns) >= 2:
                mean_ret = sum(log_returns) / len(log_returns)
                var = sum((r - mean_ret) ** 2 for r in log_returns) / (len(log_returns) - 1)
                volatility = math.sqrt(max(0.0, var))

        # Formula 22: Tick Rate Change = (TickRate(t) - TickRate(t-1)) / TickRate(t-1)
        tick_rate_change = 0.0
        if len(history) >= 2:
            prev_tr = history[-2].tick_rate
            if prev_tr > 0.0:
                tick_rate_change = (current_candle.tick_rate - prev_tr) / prev_tr

        # Formula 23: Normalized Candle Range = Range(t) / AverageRange(t)
        norm_range = (current_candle.range / avg_range) if avg_range > 0.0 else 1.0

        features = AgexRollingFeatures(
            candle_index=current_candle.index,
            rolling_average_range=round(avg_range, 3),
            rolling_average_tick_rate=round(avg_tick_rate, 2),
            rolling_volatility=round(volatility, 6),
            tick_rate_change=round(tick_rate_change, 3),
            normalized_range=round(norm_range, 2),
        )

        self.rolling_features.append(features)
        return features

    # ----------------------------------------------------------------------- #
    # Pattern & Direction Classifier (Video Price Action Rules)
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _classify_candle(
        direction: int,
        body_pct: float,
        upper_pct: float,
        lower_pct: float,
        lower_wick: float,
        upper_wick: float,
        body: float,
        c_range: float,
    ) -> Tuple[str, int]:
        """Classifies candle into high-probability institutional price action patterns."""
        # 1. Indecision (Doji / Spinning Top / Equal wicks fight with tiny body < 20%)
        if (body_pct < 20.0 and upper_pct >= 25.0 and lower_pct >= 25.0) or (
            body_pct < 10.0 and abs(upper_wick - lower_wick) / c_range < 0.25
        ):
            return "INDECISION_CHOP", 0

        # 2. Bullish Rejection (Hammer / Pin Bar / Buyers violently reject Low)
        if lower_pct >= 35.0 and lower_wick >= 1.3 * max(body, 0.05) and upper_pct <= 25.0:
            return "BULLISH_HAMMER_REJECTION", 1

        # 3. Bullish Momentum / Expansion (Green Marubozu / Solid Bull body)
        if direction == 1 and body_pct >= 35.0 and upper_pct <= 35.0:
            return "BULLISH_EXPANSION", 1

        # 4. Bearish Rejection (Shooting Star / Inverted Pin Bar / Sellers violently reject High)
        if upper_pct >= 35.0 and upper_wick >= 1.3 * max(body, 0.05) and lower_pct <= 25.0:
            return "BEARISH_SHOOTING_STAR_REJECTION", 2

        # 5. Bearish Momentum / Expansion (Red Marubozu / Solid Bear body)
        if direction == -1 and body_pct >= 35.0 and lower_pct <= 35.0:
            return "BEARISH_EXPANSION", 2

        # 6. Directional Follow (Body >= 20%)
        if direction == 1 and body_pct >= 20.0:
            return "BULLISH_MODERATE", 1
        elif direction == -1 and body_pct >= 20.0:
            return "BEARISH_MODERATE", 2

        return "NEUTRAL_WAIT", 0

    # ----------------------------------------------------------------------- #
    # Query Helpers
    # ----------------------------------------------------------------------- #
    def current_progress(self) -> Dict[str, object]:
        """Returns the in-progress candle state."""
        count = len(self._current_ticks)
        if count == 0:
            return {"ticks": 0, "pct": 0.0, "status": "WAITING_FOR_TICKS"}

        first = self._current_ticks[0]
        last = self._current_ticks[-1]
        high = max(t.mid for t in self._current_ticks)
        low = min(t.mid for t in self._current_ticks)

        return {
            "ticks": count,
            "target_k": self.k,
            "progress_pct": round((count / self.k) * 100.0, 1),
            "open": first.mid,
            "current_price": last.mid,
            "high": high,
            "low": low,
            "current_spread": last.spread,
        }

    def latest_candle(self) -> Optional[AgexTickCandle]:
        return self.completed_candles[-1] if self.completed_candles else None

    def latest_features(self) -> Optional[AgexRollingFeatures]:
        return self.rolling_features[-1] if self.rolling_features else None

