"""
live/broker.py — broker abstraction layer shared by MockBroker and Mt5Broker.

Centralizes:
  * MT5 trade retcode constants + human-readable names (verified against the
    MQL5 reference: REQUOTE=10004, DONE=10009, DONE_PARTIAL=10010, PLACED=10008,
    REJECT=10006, MARKET_CLOSED=10018, ...)
  * typed order / position / account structures
  * the :class:`BaseBroker` ABC every execution path must implement

The live system only ever talks to a broker through this interface, which
makes every order path testable with MockBroker (sandbox/demo) and deployable
with Mt5Broker (real MT5 terminal).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# MT5 trade retcodes (MQL5 TRADE_RETCODE_*; the Python package exposes the
# same integer values on the result objects)
# --------------------------------------------------------------------------- #
TRADE_RETCODE_REQUOTE = 10004        # Requote
TRADE_RETCODE_REJECT = 10006         # Request rejected
TRADE_RETCODE_CANCEL = 10007         # Request canceled by trader
TRADE_RETCODE_PLACED = 10008         # Order placed
TRADE_RETCODE_DONE = 10009           # Request completed
TRADE_RETCODE_DONE_PARTIAL = 10010   # Request completed partially
TRADE_RETCODE_ERROR = 10014          # Request rejected by server
TRADE_RETCODE_TIMEOUT = 10015        # Request canceled by timeout
TRADE_RETCODE_INVALID = 10016        # Invalid request (volume/price/stops)
TRADE_RETCODE_TRADE_DISABLED = 10017  # Trade disabled
TRADE_RETCODE_MARKET_CLOSED = 10018  # Market closed
TRADE_RETCODE_NO_MONEY = 10019       # Not enough money
TRADE_RETCODE_PRICE_CHANGED = 10020  # Price changed
TRADE_RETCODE_PRICE_OFF = 10021      # No prices

RETCODE_NAMES: Dict[int, str] = {
    10004: "REQUOTE",
    10006: "REJECT",
    10007: "CANCEL",
    10008: "PLACED",
    10009: "DONE",
    10010: "DONE_PARTIAL",
    10014: "ERROR",
    10015: "TIMEOUT",
    10016: "INVALID",
    10017: "TRADE_DISABLED",
    10018: "MARKET_CLOSED",
    10019: "NO_MONEY",
    10020: "PRICE_CHANGED",
    10021: "PRICE_OFF",
}

# Retcodes that mean the order was accepted by the broker.
SUCCESS_RETCODES = frozenset({TRADE_RETCODE_DONE, TRADE_RETCODE_DONE_PARTIAL, TRADE_RETCODE_PLACED})

# Retcodes that are transient and safe to retry with a fresh price.
TRANSIENT_RETCODES = frozenset({TRADE_RETCODE_REQUOTE, TRADE_RETCODE_PRICE_CHANGED, TRADE_RETCODE_PRICE_OFF})

# --------------------------------------------------------------------------- #
# MT5 request enums (MQL5 fixed values — used directly in request dicts)
# --------------------------------------------------------------------------- #
TRADE_ACTION_DEAL = 1
TRADE_ACTION_SLTP = 3
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TIME_GTC = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_FOK = 2
POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1


# --------------------------------------------------------------------------- #
# Typed structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OrderRequest:
    """A single order intent.

    ``token`` is the idempotency key.  It MUST be deterministic for a given
    (symbol, side, decision bar, signal) so that a crash-and-restart between
    order_send and confirmation never double-submits the same market order.
    """

    symbol: str
    side: str                          # "buy" | "sell"
    volume: float
    sl: Optional[float] = None         # stop-loss price (mandatory on opens)
    tp: Optional[float] = None         # take-profit price (mandatory on opens)
    comment: str = ""
    magic: int = 0
    token: Optional[str] = None        # idempotency key
    reduce_only: bool = False          # flatten a position (close semantics)


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    retcode: int
    retcode_name: str
    ticket: Optional[int] = None
    volume_filled: float = 0.0
    fill_price: Optional[float] = None
    message: str = ""

    @classmethod
    def success(cls, ticket: Optional[int], volume_filled: float,
                fill_price: Optional[float], message: str = "", retcode: int = TRADE_RETCODE_DONE) -> "OrderResult":
        return cls(ok=True, retcode=retcode, retcode_name=RETCODE_NAMES.get(retcode, "OK"),
                   ticket=ticket, volume_filled=volume_filled, fill_price=fill_price, message=message)

    @classmethod
    def failure(cls, retcode: int, message: str, retcode_name: Optional[str] = None) -> "OrderResult":
        return cls(ok=False, retcode=retcode,
                   retcode_name=retcode_name or RETCODE_NAMES.get(retcode, "UNKNOWN"),
                   message=message)


@dataclass(frozen=True)
class PositionInfo:
    ticket: int
    symbol: str
    side: str                          # "buy" | "sell"
    volume: float
    open_price: float
    sl: Optional[float]
    tp: Optional[float]
    magic: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "side": self.side,
            "volume": self.volume,
            "open_price": self.open_price,
            "sl": self.sl,
            "tp": self.tp,
            "magic": self.magic,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "PositionInfo":
        return cls(
            ticket=int(d.get("ticket", 0)),
            symbol=str(d.get("symbol", "")),
            side=str(d.get("side", "buy")),
            volume=float(d.get("volume", 0.0)),
            open_price=float(d.get("open_price", 0.0)),
            sl=None if d.get("sl") is None else float(d["sl"]),
            tp=None if d.get("tp") is None else float(d["tp"]),
            magic=int(d.get("magic", 0)),
        )


@dataclass(frozen=True)
class AccountInfo:
    balance: float
    equity: float
    margin_free: float


@dataclass(frozen=True)
class ReconResult:
    """Outcome of a broker-vs-local startup reconciliation."""

    ok: bool
    broker_positions: List[PositionInfo]
    local_positions: List[PositionInfo]
    missing_in_broker: List[PositionInfo]   # local wants, broker has not
    unknown_in_broker: List[PositionInfo]   # broker has, local does not know
    mismatched: List[tuple]                 # (local, broker) volume/price drift

    def describe(self) -> str:
        parts = []
        if self.missing_in_broker:
            parts.append(f"missing_in_broker={len(self.missing_in_broker)}")
        if self.unknown_in_broker:
            parts.append(f"unknown_in_broker={len(self.unknown_in_broker)}")
        if self.mismatched:
            parts.append(f"mismatched={len(self.mismatched)}")
        return "clean" if self.ok else ("DRIFT: " + ", ".join(parts))


class BrokerError(Exception):
    """Raised when a broker operation fails outside the order-retcode path."""

    def __init__(self, message: str, retcode: Optional[int] = None,
                 retcode_name: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.retcode = retcode
        self.retcode_name = retcode_name or (RETCODE_NAMES.get(retcode, "?") if retcode else None)


# --------------------------------------------------------------------------- #
# Base broker interface
# --------------------------------------------------------------------------- #
class BaseBroker(ABC):
    """Interface every broker implementation (mock or real MT5) must provide."""

    @abstractmethod
    def connect(self) -> None:
        """Connect / initialize the broker session."""

    @abstractmethod
    def disconnect(self) -> None:
        """Shut the broker session down."""

    @abstractmethod
    def account_info(self) -> AccountInfo:
        """Current balance / equity / free margin."""

    @abstractmethod
    def get_tick(self, symbol: str) -> Dict[str, float]:
        """Latest bid / ask / last for a symbol."""

    @abstractmethod
    def get_positions(self, symbol: Optional[str] = None,
                      magic: Optional[int] = None) -> List[PositionInfo]:
        """Open positions, optionally filtered."""

    @abstractmethod
    def submit_order(self, req: OrderRequest) -> OrderResult:
        """Submit a market order (entry or reduce-only flatten)."""

    @abstractmethod
    def close_position(self, ticket: int, volume: Optional[float] = None) -> OrderResult:
        """Close (all or part of) an open position by broker ticket."""

    @abstractmethod
    def modify_sl_tp(self, ticket: int, sl: Optional[float],
                     tp: Optional[float]) -> OrderResult:
        """Update the SL/TP of an open position."""

    def reconcile(self, local_positions: List[PositionInfo],
                  symbol: Optional[str] = None,
                  magic: Optional[int] = None) -> ReconResult:
        """Compare broker positions against locally-persisted state.

        This is the startup-reconciliation primitive required by P0-6: on
        process start the live loop calls this and refuses/halts when drift
        is detected (missing, unknown, or mismatched positions).
        """
        broker = self.get_positions(symbol=symbol, magic=magic)
        local = list(local_positions)

        broker_by_ticket = {b.ticket: b for b in broker}
        missing: List[PositionInfo] = []
        unknown: List[PositionInfo] = list(broker)
        mismatched: List[tuple] = []

        for p in local:
            # Primary key is the broker ticket (persisted from a prior session).
            if p.ticket > 0 and p.ticket in broker_by_ticket:
                b = broker_by_ticket[p.ticket]
                unknown = [u for u in unknown if u.ticket != b.ticket]
                vol_drift = abs(b.volume - p.volume) > 1e-9
                price_drift = (p.open_price > 0 and b.open_price > 0 and
                               abs(b.open_price - p.open_price) / p.open_price > 0.01)
                if vol_drift or price_drift:
                    mismatched.append((p, b))
                continue
            # Local ticket unknown (0/None): fuzzy-match against the remaining
            # broker positions by side + volume.  Exactly one match consumes
            # it; anything else is a local-only position (missing in broker).
            matches = [b for b in unknown if b.side == p.side]
            exact = [b for b in matches if abs(b.volume - p.volume) <= 1e-9]
            if len(exact) == 1:
                unknown = [u for u in unknown if u.ticket != exact[0].ticket]
            else:
                missing.append(p)

        ok = not missing and not unknown and not mismatched
        return ReconResult(
            ok=ok,
            broker_positions=broker,
            local_positions=local,
            missing_in_broker=missing,
            unknown_in_broker=unknown,
            mismatched=mismatched,
        )
