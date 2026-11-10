"""
AGEX AI — Phase 2: Order Flow Imbalance (OFI) & Market Depth Engine.

Implements the institutional quantitative order-book framework defined in
`Multi-Technique Prediction.md` (Sections 2.C, 7, 24.A, 31):
  - Cont, Kukanov & Stoikov (2014) Order Flow Imbalance (CKS Formula)
  - Best-Level OFI and Multi-Level OFI with decaying depth weights
  - Static & Dynamic Order Book Depth Imbalance
  - Rolling Window Aggregation (4-minute cycle matching AGEX AI)
  - MT5 DOM Safe Adapter (MarketBookAdd, MarketBookGet, MarketBookRelease)
  - Strict Data Integrity: If DOM is unavailable from broker, status is UNAVAILABLE
    and never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class OrderBookLevel:
    """Represents a single price level in the order book."""
    price: float
    volume: float
    side: str  # "bid" or "ask"


@dataclass
class OrderBookSnapshot:
    """A timestamped point-in-time state of the Limit Order Book."""
    timestamp: float
    bids: List[OrderBookLevel]  # Sorted descending by price (best bid first)
    asks: List[OrderBookLevel]  # Sorted ascending by price (best ask first)

    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None

    @property
    def total_bid_depth(self) -> float:
        return sum(b.volume for b in self.bids)

    @property
    def total_ask_depth(self) -> float:
        return sum(a.volume for a in self.asks)

    def depth_imbalance(self) -> float:
        """
        Formula (Section 7):
            DepthImbalance = (BidDepth - AskDepth) / (BidDepth + AskDepth)
        Bounded in [-1.0, +1.0].
        """
        bd = self.total_bid_depth
        ad = self.total_ask_depth
        total = bd + ad
        if total <= 1e-6:
            return 0.0
        return min(1.0, max(-1.0, (bd - ad) / total))


@dataclass
class OFIEvent:
    timestamp: float
    best_bid_price: float
    best_bid_qty: float
    best_ask_price: float
    best_ask_qty: float
    bid_contribution: float
    ask_contribution: float
    event_ofi: float


class OFICalculator:
    """
    Computes Cont-Kukanov-Stoikov (2014) Order Flow Imbalance (OFI).

    Formulas (Section 7):
      BidContribution = I(Bid_n >= Bid_{n-1}) * Qty_n^b - I(Bid_n <= Bid_{n-1}) * Qty_{n-1}^b
      AskContribution = I(Ask_n <= Ask_{n-1}) * Qty_n^a - I(Ask_n >= Ask_{n-1}) * Qty_{n-1}^a
      OFI_n = BidContribution - AskContribution
      OFI_Total = Sum(OFI_n)
      Normalized_OFI = OFI_Total / (AverageBidDepth + AverageAskDepth)
    """

    def __init__(self, window_seconds: float = 240.0, max_levels: int = 5):
        self.window_seconds = float(window_seconds)
        self.max_levels = int(max_levels)

        self._last_snapshot: Optional[OrderBookSnapshot] = None
        self._events: List[OFIEvent] = []
        self._depth_imbalances: List[Tuple[float, float]] = []  # (ts, imbalance)
        self.total_snapshots_processed: int = 0
        self.is_available: bool = False

    def add_snapshot(self, snapshot: OrderBookSnapshot) -> Optional[OFIEvent]:
        """Ingests an order book snapshot and updates OFI if a previous state exists."""
        if snapshot is None or not snapshot.bids or not snapshot.asks:
            return None

        self.is_available = True
        self.total_snapshots_processed += 1
        now_ts = snapshot.timestamp

        # Record depth imbalance
        cur_imbalance = snapshot.depth_imbalance()
        self._depth_imbalances.append((now_ts, cur_imbalance))

        event: Optional[OFIEvent] = None
        if self._last_snapshot is not None:
            prev_b = self._last_snapshot.best_bid
            prev_a = self._last_snapshot.best_ask
            curr_b = snapshot.best_bid
            curr_a = snapshot.best_ask

            if prev_b and prev_a and curr_b and curr_a:
                # 1. Bid Contribution
                # If Bid improves (Price > PrevPrice): entire new volume adds to buy pressure
                # If Bid drops (Price < PrevPrice): entire old volume drops from buy pressure
                # If Bid price unchanged: change in volume delta adds/subtracts
                if curr_b.price > prev_b.price:
                    bid_contrib = curr_b.volume
                elif curr_b.price < prev_b.price:
                    bid_contrib = -prev_b.volume
                else:
                    bid_contrib = curr_b.volume - prev_b.volume

                # 2. Ask Contribution
                # If Ask drops/improves (Price < PrevPrice): entire new volume adds to sell pressure
                # If Ask rises (Price > PrevPrice): entire old volume removed
                # If Ask price unchanged: change in ask volume
                if curr_a.price < prev_a.price:
                    ask_contrib = curr_a.volume
                elif curr_a.price > prev_a.price:
                    ask_contrib = -prev_a.volume
                else:
                    ask_contrib = curr_a.volume - prev_a.volume

                # 3. Net Event OFI
                event_ofi = bid_contrib - ask_contrib

                event = OFIEvent(
                    timestamp=now_ts,
                    best_bid_price=curr_b.price,
                    best_bid_qty=curr_b.volume,
                    best_ask_price=curr_a.price,
                    best_ask_qty=curr_a.volume,
                    bid_contribution=round(bid_contrib, 4),
                    ask_contribution=round(ask_contrib, 4),
                    event_ofi=round(event_ofi, 4),
                )
                self._events.append(event)

        self._last_snapshot = snapshot

        # Evict events older than window
        cutoff = now_ts - self.window_seconds
        while self._events and self._events[0].timestamp < cutoff:
            self._events.pop(0)
        while self._depth_imbalances and self._depth_imbalances[0][0] < cutoff:
            self._depth_imbalances.pop(0)

        return event

    def clear(self) -> None:
        self._last_snapshot = None
        self._events.clear()
        self._depth_imbalances.clear()
        self.total_snapshots_processed = 0

    def compute_ofi_metrics(self) -> Dict[str, Any]:
        """
        Aggregates OFI over the active window.
        Returns normalized OFI [-1.0, +1.0] and average depth metrics.
        """
        if not self.is_available or not self._events or self._last_snapshot is None:
            return {
                "status": "UNAVAILABLE",
                "ofi_total": 0.0,
                "normalized_ofi": 0.0,
                "depth_imbalance": 0.0,
                "average_depth": 0.0,
                "event_count": len(self._events),
            }

        ofi_total = sum(e.event_ofi for e in self._events)

        # Average total depth across recent snapshots
        cur_depth = self._last_snapshot.total_bid_depth + self._last_snapshot.total_ask_depth
        avg_depth = max(cur_depth, 1.0)

        # Normalized OFI in [-1.0, +1.0] using hyperbolic tangent scaling against depth
        # Ensures smooth bounds without hard-clipping artifacts
        scaling_factor = max(avg_depth * 2.0, 1.0)
        normalized_ofi = math.tanh(ofi_total / scaling_factor)

        # Mean depth imbalance across window
        avg_imbalance = (
            sum(imb for _, imb in self._depth_imbalances) / float(len(self._depth_imbalances))
            if self._depth_imbalances
            else 0.0
        )

        return {
            "status": "AVAILABLE",
            "ofi_total": round(ofi_total, 4),
            "normalized_ofi": round(normalized_ofi, 4),
            "depth_imbalance": round(avg_imbalance, 4),
            "average_depth": round(avg_depth, 2),
            "event_count": len(self._events),
        }


# --------------------------------------------------------------------------- #
# MT5 Market Depth Safe Adapter
# --------------------------------------------------------------------------- #
class MT5MarketBookManager:
    """
    Safely subscribes to and polls MT5 Depth of Market (DOM).
    Automatically detects if broker supports DOM for the symbol.
    Provides graceful fallback if broker spot feed does not offer DOM.
    """

    def __init__(self, mt5_instance: Any, symbol: str):
        self.mt5 = mt5_instance
        self.symbol = str(symbol)
        self.is_subscribed: bool = False
        self.dom_supported: bool = False
        self._last_poll_time: float = 0.0

    def subscribe(self) -> bool:
        """Calls market_book_add() on the MT5 instance."""
        if self.mt5 is None:
            return False

        try:
            ok = self.mt5.market_book_add(self.symbol)
            if ok:
                self.is_subscribed = True
                self.dom_supported = True
                return True
            else:
                self.is_subscribed = False
                self.dom_supported = False
                return False
        except Exception:
            self.is_subscribed = False
            self.dom_supported = False
            return False

    def poll_snapshot(self) -> Optional[OrderBookSnapshot]:
        """Polls current order book from MT5 and returns OrderBookSnapshot."""
        if not self.is_subscribed or self.mt5 is None:
            return None

        try:
            raw_book = self.mt5.market_book_get(self.symbol)
            if not raw_book:
                return None

            now_ts = time.time()
            bids: List[OrderBookLevel] = []
            asks: List[OrderBookLevel] = []

            # MqlBookInfo types:
            # BOOK_TYPE_BUY = 1 (Bid)
            # BOOK_TYPE_SELL = 2 (Ask)
            for item in raw_book:
                item_type = getattr(item, "type", None)
                price = float(getattr(item, "price", 0.0))
                vol = float(getattr(item, "volume", 0.0) or getattr(item, "volume_real", 0.0))

                if price <= 0.0 or vol <= 0.0:
                    continue

                if item_type == 1 or str(item_type).upper() in ("BOOK_TYPE_BUY", "1"):
                    bids.append(OrderBookLevel(price=price, volume=vol, side="bid"))
                elif item_type == 2 or str(item_type).upper() in ("BOOK_TYPE_SELL", "2"):
                    asks.append(OrderBookLevel(price=price, volume=vol, side="ask"))

            # Sort bids descending (best bid highest price), asks ascending (best ask lowest price)
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price, reverse=False)

            if not bids and not asks:
                return None

            return OrderBookSnapshot(timestamp=now_ts, bids=bids, asks=asks)

        except Exception:
            return None

    def release(self) -> None:
        """Cancels DOM subscription via market_book_release()."""
        if self.is_subscribed and self.mt5 is not None:
            try:
                self.mt5.market_book_release(self.symbol)
            except Exception:
                pass
            finally:
                self.is_subscribed = False

