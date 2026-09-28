"""
live/mock_broker.py — deterministic simulated broker for demo/sandbox/testing.

The MockBroker implements the SAME :class:`BaseBroker` interface as the real
MT5 broker so every order path (entry, SL/TP-triggered exit, manual close,
modify) can be exercised without a terminal — and so the sandbox demo mode
trades against realistic simulated fills instead of fabricated numbers.

Fill semantics (deterministic, seeded):
  * buys fill at ask + adverse slippage, sells at bid - adverse slippage
    (bid/ask derived from a mid price and the configured round-trip spread);
  * SL/TP orders are honored by ``advance(bar)``: for each bar the broker
    checks whether the stop or the take-profit level is breached.  When both
    levels would be breached inside one bar the STOP fills first
    (conservative — the same assumption backtesting.py makes).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from live.broker import (
    AccountInfo,
    BaseBroker,
    BrokerError,
    OrderRequest,
    OrderResult,
    PositionInfo,
    TRADE_RETCODE_DONE,
)

logger = logging.getLogger(__name__)


class MockBroker(BaseBroker):
    """Deterministic in-memory broker.  Optional JSON state persistence."""

    def __init__(
        self,
        symbol: str = "XAUUSD",
        balance: float = 10_000.0,
        spread: float = 0.00025,      # round-trip spread fraction (as CostConfig)
        commission: float = 0.00003,  # per side relative
        slippage: float = 0.00005,    # mean |adverse slippage| per fill
        seed: int = 42,
        state_path: Optional[Path] = None,
    ):
        self.symbol = symbol
        self._balance = float(balance)
        self.spread = float(spread)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self._rng = np.random.default_rng(int(seed))
        self.seed = int(seed)
        self.state_path = Path(state_path) if state_path else None

        self._positions: Dict[int, PositionInfo] = {}
        self._next_ticket: int = 1
        self._mid: float = 2000.0      # starting mid (tests set it explicitly)
        self._closed: List[dict] = []  # fill history
        self._order_log: List[str] = []
        self._connected: bool = False

        if self.state_path is not None:
            self._load_state()

    # ------------------------------------------------------------------ #
    # BaseBroker interface
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        self._connected = True
        logger.info("MockBroker connected (seed=%s, spread=%s)", self.seed, self.spread)

    def disconnect(self) -> None:
        self._connected = False
        self._save_state()

    def account_info(self) -> AccountInfo:
        equity = self._balance + self._floating_pnl()
        return AccountInfo(balance=self._balance, equity=equity,
                           margin_free=max(0.0, equity))

    def get_tick(self, symbol: str) -> Dict[str, float]:
        self._require_symbol(symbol)
        half = self.spread / 2.0
        # Spread is a FRACTION of price (CostConfig.spread = 2.5bp round-trip):
        # bid = mid * (1 - spread/2), ask = mid * (1 + spread/2), so on
        # XAUUSD @ $2000 the spread is $0.50, not $0.00025.
        return {"bid": self._mid * (1 - half), "ask": self._mid * (1 + half),
                "last": self._mid, "spread": self.spread}

    def get_positions(self, symbol: Optional[str] = None,
                      magic: Optional[int] = None) -> List[PositionInfo]:
        out = []
        for p in self._positions.values():
            if symbol and p.symbol != symbol:
                continue
            if magic is not None and p.magic != magic:
                continue
            out.append(p)
        return out

    def submit_order(self, req: OrderRequest) -> OrderResult:
        if not self._connected:
            raise BrokerError("MockBroker not connected — call connect() first")
        self._order_log.append(
            f"submit {req.side} vol={req.volume} token={req.token or '-'} "
            f"sl={req.sl} tp={req.tp}"
        )
        tick = self.get_tick(req.symbol)
        slip = self._rng.normal(loc=0.0, scale=self.slippage)  # signed, mean |.| = slippage
        if req.side == "buy":
            fill = tick["ask"] + abs(slip)
        elif req.side == "sell":
            fill = tick["bid"] - abs(slip)
        else:
            raise BrokerError(f"unknown side {req.side!r}")

        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions[ticket] = PositionInfo(
            ticket=ticket,
            symbol=req.symbol,
            side=req.side,
            volume=req.volume,
            open_price=float(fill),
            sl=req.sl,
            tp=req.tp,
            magic=req.magic,
        )
        self._order_log.append(f"  -> filled ticket={ticket} at {fill:.4f}")
        self._save_state()
        return OrderResult.success(ticket=ticket, volume_filled=req.volume,
                                   fill_price=float(fill))

    def close_position(self, ticket: int, volume: Optional[float] = None) -> OrderResult:
        pos = self._positions.get(ticket)
        if pos is None:
            return OrderResult.failure(retcode=10016, message=f"no position ticket={ticket}")
        close_vol = volume if volume is not None else pos.volume
        if close_vol > pos.volume + 1e-12:
            return OrderResult.failure(retcode=10016,
                                       message=f"close volume {close_vol} > pos {pos.volume}")
        tick = self.get_tick(pos.symbol)
        slip = self._rng.normal(loc=0.0, scale=self.slippage)
        if pos.side == "buy":
            fill = tick["bid"] - abs(slip)
        else:
            fill = tick["ask"] + abs(slip)

        pnl = (fill - pos.open_price) * close_vol * (1 if pos.side == "buy" else -1)
        comm = close_vol * fill * self.commission
        net = pnl - comm - abs(close_vol * fill * self.commission * 0.0)  # comm on exit side
        self._balance += pnl - comm
        self._closed.append({
            "ticket": ticket, "side": pos.side, "volume": close_vol,
            "entry": pos.open_price, "exit": float(fill), "pnl": float(net),
            "reason": "manual_close",
        })
        if close_vol >= pos.volume - 1e-12:
            del self._positions[ticket]
        else:
            self._positions[ticket] = PositionInfo(
                ticket=pos.ticket, symbol=pos.symbol, side=pos.side,
                volume=pos.volume - close_vol, open_price=pos.open_price,
                sl=pos.sl, tp=pos.tp, magic=pos.magic)
        self._save_state()
        return OrderResult.success(ticket=ticket, volume_filled=close_vol,
                                   fill_price=float(fill))

    def modify_sl_tp(self, ticket: int, sl: Optional[float],
                     tp: Optional[float]) -> OrderResult:
        pos = self._positions.get(ticket)
        if pos is None:
            return OrderResult.failure(retcode=10016, message=f"no position ticket={ticket}")
        self._positions[ticket] = PositionInfo(
            ticket=pos.ticket, symbol=pos.symbol, side=pos.side,
            volume=pos.volume, open_price=pos.open_price,
            sl=pos.sl if sl is None else sl,
            tp=pos.tp if tp is None else tp,
            magic=pos.magic)
        self._save_state()
        return OrderResult.success(ticket=ticket, volume_filled=pos.volume,
                                   fill_price=float(self._mid))

    # ------------------------------------------------------------------ #
    # Mock-only helpers (driving the simulated market)
    # ------------------------------------------------------------------ #
    def set_mid(self, mid: float) -> None:
        """Set the mid price directly (test convenience).

        SL/TP evaluation happens on ``advance(bar)`` where a full high/low/close
        bar is available; ``set_mid`` only moves the quote for the next fills.
        """
        self._mid = float(mid)

    def advance(self, high: float, low: float, close: float) -> List[dict]:
        """Process one price bar against every open position's SL/TP.

        Conservative ordering: when a bar breaches both the stop and the
        take-profit, the STOP fills first (matching backtesting.py).
        """
        if high < low:
            raise ValueError("advance(): high < low")
        fills: List[dict] = []
        # Snapshot keys because _close_position mutates the dict.
        for ticket in list(self._positions.keys()):
            pos = self._positions[ticket]
            if pos.side == "buy":
                stopped = low <= float(pos.sl) if pos.sl is not None else False
                took = high >= float(pos.tp) if pos.tp is not None else False
                if stopped:
                    fills.append(self._stop_fill(ticket, float(pos.sl)))
                elif took:
                    fills.append(self._tp_fill(ticket, float(pos.tp)))
            else:  # short
                stopped = high >= float(pos.sl) if pos.sl is not None else False
                took = low <= float(pos.tp) if pos.tp is not None else False
                if stopped:
                    fills.append(self._stop_fill(ticket, float(pos.sl)))
                elif took:
                    fills.append(self._tp_fill(ticket, float(pos.tp)))
        self._mid = float(close)
        return fills

    def _stop_fill(self, ticket: int, level: float) -> dict:
        pos = self._positions[ticket]
        slip = abs(self._rng.normal(loc=0.0, scale=self.slippage))
        fill = level - slip if pos.side == "buy" else level + slip  # adverse
        return self._close_at(ticket, fill, reason="SL")

    def _tp_fill(self, ticket: int, level: float) -> dict:
        pos = self._positions[ticket]
        slip = abs(self._rng.normal(loc=0.0, scale=self.slippage))
        fill = level + slip if pos.side == "buy" else level - slip  # adverse
        return self._close_at(ticket, fill, reason="TP")

    def _close_at(self, ticket: int, fill: float, reason: str) -> dict:
        pos = self._positions[ticket]
        pnl = (fill - pos.open_price) * pos.volume * (1 if pos.side == "buy" else -1)
        comm = pos.volume * fill * self.commission
        self._balance += pnl - comm
        rec = {
            "ticket": ticket, "side": pos.side, "volume": pos.volume,
            "entry": pos.open_price, "exit": fill, "pnl": float(pnl - comm),
            "reason": reason,
        }
        self._closed.append(rec)
        del self._positions[ticket]
        self._order_log.append(f"  -> {reason} ticket={ticket} at {fill:.4f}")
        self._save_state()
        return rec

    # ------------------------------------------------------------------ #
    # introspection / persistence
    # ------------------------------------------------------------------ #
    def open_position(self, ticket: int) -> Optional[PositionInfo]:
        return self._positions.get(ticket)

    def closed_fills(self) -> List[dict]:
        return list(self._closed)

    def order_log(self) -> List[str]:
        return list(self._order_log)

    def _floating_pnl(self) -> float:
        return 0.0  # mock holds no mark-to-market until advance()

    def _require_symbol(self, symbol: str) -> None:
        if symbol != self.symbol:
            raise BrokerError(f"mock only knows symbol={self.symbol}, got {symbol!r}")

    # ------------------------------------------------------------------ #
    # optional JSON persistence (survives restart for demo mode)
    # ------------------------------------------------------------------ #
    def _state_dict(self) -> dict:
        return {
            "balance": self._balance,
            "mid": self._mid,
            "next_ticket": self._next_ticket,
            "positions": [p.to_dict() for p in self._positions.values()],
            "closed": self._closed[-500:],
        }

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(__import__("json").dumps(self._state_dict()), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError:  # pragma: no cover - best-effort persistence
            logger.warning("MockBroker state save failed", exc_info=True)

    def _load_state(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            data = __import__("json").loads(self.state_path.read_text(encoding="utf-8"))
            self._balance = float(data.get("balance", self._balance))
            self._mid = float(data.get("mid", self._mid))
            self._next_ticket = int(data.get("next_ticket", 1))
            self._positions = {
                int(p["ticket"]): PositionInfo.from_dict(p)
                for p in data.get("positions", [])
            }
            self._closed = list(data.get("closed", []))
        except (ValueError, TypeError, KeyError, OSError):
            logger.warning("MockBroker state load failed; starting fresh", exc_info=True)
            self._positions = {}
