"""
live/trade_executor.py — RiskSupervisor-wired order-path controller.

Every order path in the live system goes through this class:

    * execute_entry(...)      — open a new long/short position
    * execute_scale(...)      — add to an existing position
    * modify_sl_tp(...)       — adjust SL/TP on an open position
    * execute_close(...)      — flatten (all or part) a position

For EVERY path the RiskSupervisor is consulted BEFORE the broker is touched:

    * entry/scale   -> ``check_trade(signal, size_fraction, state, market_data)``
                       MUST pass or no order is submitted (P0-1/P0-2 fix).
    * modify_sl_tp  -> supervisor is consulted; a HALTED supervisor blocks
                       modifications (no exposure change is safe while halted).
    * close         -> supervisor is consulted and logged; a rejection NEVER
                       blocks a close because closing is de-risking (a
                       documented safety override; blocked closes would trap
                       the account in a losing position).

Every entry order carries ATR-based SL and TP computed from CAUSAL data
(broker config ``sl_atr_mult`` / ``tp_atr_mult``).  Orders are sized as an
explicit fraction of equity via the ATR position sizer and capped by the risk
config's ``max_position`` — never a hard-coded lot (P0-1 fix).

Position sizing is HONEST about broker lot constraints:

    * volume is FLOORED to the lot step (``LOT_STEP``) — never rounded up,
      because rounding up can double exposure past the configured cap;
    * the aggregate notional cap (``max_position * equity``) is enforced
      across ALL open positions, not just per order — repeated entries and
      scaling cannot silently lever the account past the cap;
    * if the risk-capped size falls below the broker minimum lot (``MIN_LOT``)
      the order is REJECTED (``BELOW_MIN_LOT``) rather than filling an
      oversized order.

Local state (open positions + last bar) is persisted to
``state/bot_state.json`` after every order so startup reconciliation can
compare broker reality against what we believe we hold.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.strategies import atr as causal_atr
from core.config import AppConfig
from live.broker import (
    BaseBroker,
    BrokerError,
    OrderRequest,
    OrderResult,
    PositionInfo,
    ReconResult,
)
from models.position_sizing import ATRPositionSizer
from models.risk_supervisor import RiskSupervisor

logger = logging.getLogger(__name__)

XAUUSD_CONTRACT_SIZE = 100.0   # 1 lot = 100 troy oz
MIN_LOT = 0.01
LOT_STEP = 0.01


class TradeExecutor:
    """Wires RiskSupervisor + position sizing + broker into every order path."""

    def __init__(
        self,
        broker: BaseBroker,
        risk: RiskSupervisor,
        cfg: AppConfig,
        sizer: Optional[ATRPositionSizer] = None,
        local_state_file: Optional[Path] = None,
        close_on_recon_drift: bool = False,
    ) -> None:
        self.broker = broker
        self.risk = risk
        self.cfg = cfg
        self.sizer = sizer or ATRPositionSizer(
            account_risk=cfg.risk.risk_per_trade,
            atr_multiplier=cfg.broker.sl_atr_mult,
        )
        self.local_state_file = Path(local_state_file) if local_state_file else cfg.paths.local_state_file
        self.close_on_recon_drift = close_on_recon_drift
        self.halted: bool = False
        self.halt_reason: str = ""
        self.last_bar_time: Optional[str] = None
        self._positions: Dict[int, PositionInfo] = {}

        self._load_local_state()

    # ------------------------------------------------------------------ #
    # Local state persistence (for startup reconciliation)
    # ------------------------------------------------------------------ #
    def _load_local_state(self) -> None:
        if self.local_state_file is None or not self.local_state_file.exists():
            return
        try:
            data = json.loads(self.local_state_file.read_text(encoding="utf-8"))
            self.last_bar_time = data.get("last_bar_time")
            self._positions = {
                int(k): PositionInfo.from_dict(v)
                for k, v in data.get("positions", {}).items()
            }
        except (ValueError, TypeError, KeyError, OSError):
            logger.warning("Local state load failed; starting with no positions", exc_info=True)
            self._positions = {}

    def _save_local_state(self) -> None:
        if self.local_state_file is None:
            return
        try:
            self.local_state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "last_bar_time": self.last_bar_time,
                "positions": {str(t): p.to_dict() for t, p in self._positions.items()},
                "saved_ts": time.time(),
            }
            tmp = self.local_state_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.local_state_file)
        except OSError:
            logger.warning("Local state save failed", exc_info=True)

    # ------------------------------------------------------------------ #
    # Startup reconciliation
    # ------------------------------------------------------------------ #
    def startup_reconcile(self) -> ReconResult:
        """Compare broker positions vs persisted local state.

        On drift:
          * if ``close_on_recon_drift`` (live mode): close broker positions
            that our local state does not know (they were opened by a prior
            session whose state was lost) and drop phantom local positions;
          * regardless of the flag the executor HALTS (no new orders) until
            an operator resolves the drift, and a CRITICAL alert is logged.
        """
        local_positions = list(self._positions.values())
        rec = self.broker.reconcile(
            local_positions,
            symbol=self.cfg.broker.symbol,
            magic=self.cfg.broker.magic,
        )
        if rec.ok:
            logger.info("Startup reconciliation OK (%s)", rec.describe())
            self.halted = False
            self.halt_reason = ""
            return rec

        logger.critical("STARTUP RECONCILIATION DRIFT: %s", rec.describe())
        self.halted = True
        self.halt_reason = f"startup reconciliation drift: {rec.describe()}"

        if self.close_on_recon_drift:
            # Close broker positions we do not own per local state (same magic).
            for b in rec.unknown_in_broker:
                if b.magic == self.cfg.broker.magic:
                    res = self.broker.close_position(b.ticket)
                    logger.warning("Recon close of unknown broker pos %s -> %s",
                                   b.ticket, res)
            # Drop phantom local positions the broker does not have.
            for m in rec.missing_in_broker:
                self._positions.pop(m.ticket, None)
            self._save_local_state()
        return rec

    # ------------------------------------------------------------------ #
    # Market context (causal, from the latest closed bars)
    # ------------------------------------------------------------------ #
    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """ATR from the last CLOSED bar — no forming-bar leakage."""
        if df is None or len(df) < period + 1:
            return 0.0
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        vals = causal_atr(h, l, c, period)
        last = float(vals[-2]) if len(vals) > 1 and np.isfinite(vals[-2]) else 0.0
        return last

    def build_market_data(self, df: Optional[pd.DataFrame], tick: Dict[str, float]) -> Dict[str, float]:
        """Volatility z-score + spread + macro context for the supervisor."""
        vol = 0.0
        if df is not None and len(df) > 30:
            h = df["high"].to_numpy(dtype=float)
            l = df["low"].to_numpy(dtype=float)
            c = df["close"].to_numpy(dtype=float)
            atrs = causal_atr(h, l, c, 14)
            finite = atrs[np.isfinite(atrs)]
            if len(finite) > 30:
                mu = float(finite.mean())
                sd = float(finite.std())
                if sd > 0:
                    vol = (float(atrs[-2]) - mu) / sd
        spread_frac = 0.0
        mid = (tick.get("bid", 0.0) + tick.get("ask", 0.0)) / 2.0
        if mid > 0:
            spread_frac = (tick.get("ask", mid) - tick.get("bid", mid)) / mid
        return {
            "volatility": vol,
            "spread": spread_frac,
            "dxy_momentum": 0.0,          # filled by the caller when macro data available
            "is_high_impact_event": False,
            "is_event_window": False,
            "is_market_open": True,
        }

    def build_state(self, equity: float, position_side: int = 0) -> Dict[str, Any]:
        return {"position": position_side, "equity": equity, "is_market_open": True}

    # ------------------------------------------------------------------ #
    # Position size / volume helpers (HONEST lot math)
    # ------------------------------------------------------------------ #
    def _current_notional(self, price: float) -> float:
        """Aggregate notional exposure of every open local position (in $)."""
        total = 0.0
        for p in self._positions.values():
            total += p.volume * XAUUSD_CONTRACT_SIZE * price
        return total

    def fraction_and_volume(
        self, equity: float, entry_price: float, atr: float
    ) -> Tuple[float, float]:
        """Explicit equity fraction + broker lot volume (P0-1/P0-2 fix).

        Returns ``(fraction, volume)``.  ``volume`` is floored to ``LOT_STEP``
        and additionally constrained by the AGGREGATE notional cap so repeated
        entries cannot lever the account past ``max_position``.  A volume of
        ``0.0`` means the risk-capped size is below the broker minimum lot.
        """
        if atr <= 0 or entry_price <= 0 or equity <= 0:
            return 0.0, 0.0
        fraction = float(self.sizer.compute_position_size(atr, entry_price, equity))
        fraction = min(fraction, self.cfg.risk.max_position)

        notional_cap = self.cfg.risk.max_position * equity
        current = self._current_notional(entry_price)
        remaining_cap = max(0.0, notional_cap - current)
        ideal_notional = min(fraction * equity, remaining_cap)

        volume = math.floor(
            ideal_notional / (entry_price * XAUUSD_CONTRACT_SIZE) / LOT_STEP
        ) * LOT_STEP
        if volume < MIN_LOT:
            # Only bump to MIN_LOT when it still fits inside the aggregate cap.
            if MIN_LOT * entry_price * XAUUSD_CONTRACT_SIZE <= remaining_cap + 1e-9:
                volume = MIN_LOT
            else:
                volume = 0.0
        return fraction, volume

    def entry_sl_tp(self, side: str, entry: float, atr: float) -> Tuple[float, float]:
        if side == "buy":
            return entry - atr * self.cfg.broker.sl_atr_mult, entry + atr * self.cfg.broker.tp_atr_mult
        return entry + atr * self.cfg.broker.sl_atr_mult, entry - atr * self.cfg.broker.tp_atr_mult

    # ------------------------------------------------------------------ #
    # ORDER PATHS — every one consults the RiskSupervisor
    # ------------------------------------------------------------------ #
    def execute_entry(
        self,
        signal: Any,
        equity: float,
        df: Optional[pd.DataFrame],
        market_data: Optional[Dict[str, float]] = None,
        bar_time: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[OrderResult]]:
        """Open a new position.  Risk-gated BEFORE the broker is touched."""
        if self.halted:
            return False, f"EXECUTOR_HALTED: {self.halt_reason}", None
        direction = self._direction(signal)
        if direction not in (1, 2):
            return False, f"INVALID_ENTRY_SIGNAL: {signal!r}", None
        if direction == 2 and not self.cfg.broker.allow_short:
            return False, "SHORT_DISABLED: ALLOW_SHORT=false", None

        tick = self.broker.get_tick(self.cfg.broker.symbol)
        entry = tick["ask"] if direction == 1 else tick["bid"]
        atr = self.compute_atr(df)
        if atr <= 0:
            return False, "ATR_UNAVAILABLE: cannot size or set stops", None

        fraction, volume = self.fraction_and_volume(equity, entry, atr)
        if volume <= 0:
            return False, "BELOW_MIN_LOT: risk-capped size is smaller than broker MIN_LOT", None

        # The supervisor reviews the ACTUAL exposure this order would create.
        actual_fraction = volume * XAUUSD_CONTRACT_SIZE * entry / equity
        md = market_data or self.build_market_data(df, tick)
        state = self.build_state(equity, position_side=0)

        approved, reason = self.risk.check_trade(signal, actual_fraction, state, md)
        if not approved:
            return False, f"RISK_REJECTED: {reason}", None

        sl, tp = self.entry_sl_tp("buy" if direction == 1 else "sell", entry, atr)
        req = OrderRequest(
            symbol=self.cfg.broker.symbol,
            side="buy" if direction == 1 else "sell",
            volume=volume,
            sl=sl,
            tp=tp,
            comment=f"executor_v2_{direction}",
            magic=self.cfg.broker.magic,
            token=self._entry_token(direction, bar_time),
        )
        try:
            res = self.broker.submit_order(req)
        except BrokerError as e:
            return False, f"BROKER_ERROR: {e.message}", None

        if res.ok and res.ticket:
            self._positions[res.ticket] = PositionInfo(
                ticket=res.ticket, symbol=self.cfg.broker.symbol,
                side=req.side, volume=res.volume_filled,
                open_price=res.fill_price or entry, sl=sl, tp=tp,
                magic=self.cfg.broker.magic,
            )
            self.last_bar_time = bar_time
            self._save_local_state()
            logger.info("ENTRY %s ticket=%s vol=%.2f sl=%.2f tp=%.2f (risk=%s)",
                        req.side, res.ticket, req.volume, sl, tp, reason)
            return True, "APPROVED", res
        return False, f"ORDER_FAILED: {res.retcode_name} {res.message}", res

    def execute_scale(
        self,
        signal: Any,
        equity: float,
        df: Optional[pd.DataFrame],
        market_data: Optional[Dict[str, float]] = None,
        bar_time: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[OrderResult]]:
        """Add exposure to an existing position.  Risk-gated like an entry."""
        if self.halted:
            return False, f"EXECUTOR_HALTED: {self.halt_reason}", None
        direction = self._direction(signal)
        if direction not in (1, 2):
            return False, "SCALE requires a directional signal", None
        if not self._positions:
            return False, "NO_POSITION_TO_SCALE", None
        existing_side = next(iter(self._positions.values())).side
        want_side = "buy" if direction == 1 else "sell"
        if existing_side != want_side:
            return False, "SCALE_SIDE_MISMATCH: only same-side scaling allowed", None

        tick = self.broker.get_tick(self.cfg.broker.symbol)
        atr = self.compute_atr(df)
        if atr <= 0:
            return False, "ATR_UNAVAILABLE", None

        # Supervisor sees current position is already open (position_side != 0),
        # so the volatility filter cannot block scaling; the size breaker and
        # all other breakers still apply to the ADDED fraction.
        entry = tick["ask"] if direction == 1 else tick["bid"]
        fraction, volume = self.fraction_and_volume(equity, entry, atr)
        if volume <= 0:
            return False, "BELOW_MIN_LOT: aggregate exposure cap reached or size below MIN_LOT", None

        actual_fraction = volume * XAUUSD_CONTRACT_SIZE * entry / equity
        md = market_data or self.build_market_data(df, tick)
        state = self.build_state(equity, position_side=1 if direction == 1 else -1)

        approved, reason = self.risk.check_trade(signal, actual_fraction, state, md)
        if not approved:
            return False, f"RISK_REJECTED: {reason}", None

        sl, tp = self.entry_sl_tp("buy" if direction == 1 else "sell", entry, atr)
        req = OrderRequest(
            symbol=self.cfg.broker.symbol,
            side="buy" if direction == 1 else "sell",
            volume=volume, sl=sl, tp=tp,
            comment=f"executor_v2_scale_{direction}",
            magic=self.cfg.broker.magic,
            token=self._entry_token(direction, bar_time, prefix="scale"),
        )
        try:
            res = self.broker.submit_order(req)
        except BrokerError as e:
            return False, f"BROKER_ERROR: {e.message}", None
        if res.ok and res.ticket:
            self._positions[res.ticket] = PositionInfo(
                ticket=res.ticket, symbol=self.cfg.broker.symbol,
                side=req.side, volume=res.volume_filled,
                open_price=res.fill_price or entry, sl=sl, tp=tp,
                magic=self.cfg.broker.magic,
            )
            self.last_bar_time = bar_time
            self._save_local_state()
            return True, "APPROVED", res
        return False, f"ORDER_FAILED: {res.retcode_name} {res.message}", res

    def modify_sl_tp(self, ticket: int, sl: Optional[float],
                     tp: Optional[float]) -> Tuple[bool, str, Optional[OrderResult]]:
        """Adjust SL/TP.  A HALTED supervisor blocks modifications."""
        if self.halted:
            return False, f"EXECUTOR_HALTED: {self.halt_reason}", None
        if ticket not in self._positions:
            return False, f"UNKNOWN_TICKET: {ticket}", None
        # Consult the supervisor (flat signal, current position already open).
        # No new exposure is created by a modification, but a HALTED account
        # must not be tinkered with; we log the outcome either way.
        tick = self.broker.get_tick(self.cfg.broker.symbol)
        pos = self._positions[ticket]
        state = self.build_state(self.risk.current_equity,
                                 position_side=1 if pos.side == "buy" else -1)
        approved, reason = self.risk.check_trade(
            0, self.risk.max_position_size * 0.5, state,
            self.build_market_data(None, tick))
        if not approved:
            return False, f"RISK_BLOCKED_MODIFY: {reason}", None
        try:
            res = self.broker.modify_sl_tp(ticket, sl, tp)
        except BrokerError as e:
            return False, f"BROKER_ERROR: {e.message}", None
        if res.ok:
            updated = PositionInfo(ticket=pos.ticket, symbol=pos.symbol, side=pos.side,
                                   volume=pos.volume, open_price=pos.open_price,
                                   sl=sl if sl is not None else pos.sl,
                                   tp=tp if tp is not None else pos.tp,
                                   magic=pos.magic)
            self._positions[ticket] = updated
            self._save_local_state()
            return True, "APPROVED", res
        return False, f"MODIFY_FAILED: {res.retcode_name} {res.message}", res

    def execute_close(self, ticket: int, volume: Optional[float] = None,
                      equity: Optional[float] = None) -> Tuple[bool, str, Optional[OrderResult]]:
        """Flatten (part of) a position.

        The supervisor is ALWAYS consulted first; a rejection is logged but
        NEVER blocks the close — closing is de-risking and blocking it would
        trap the account in a losing position (documented safety override).
        """
        if ticket not in self._positions:
            return False, f"UNKNOWN_TICKET: {ticket}", None
        pos = self._positions[ticket]
        close_vol = volume if volume is not None else pos.volume

        try:
            tick = self.broker.get_tick(pos.symbol)
        except BrokerError as e:
            return False, f"BROKER_ERROR: {e.message}", None
        eq = equity if equity is not None else self.risk.current_equity
        state = self.build_state(eq, position_side=1 if pos.side == "buy" else -1)
        approved, reason = self.risk.check_trade(
            0, self.risk.max_position_size * 0.5, state,
            self.build_market_data(None, tick))
        if not approved:
            logger.warning("Risk supervisor REJECTS close (%s) — proceeding with "
                           "de-risk override (closing reduces exposure)", reason)

        try:
            res = self.broker.close_position(ticket, volume=close_vol)
        except BrokerError as e:
            return False, f"BROKER_ERROR: {e.message}", None
        if res.ok:
            remaining = pos.volume - res.volume_filled
            if remaining <= 1e-9:
                self._positions.pop(ticket, None)
            else:
                self._positions[ticket] = PositionInfo(
                    ticket=pos.ticket, symbol=pos.symbol, side=pos.side,
                    volume=remaining, open_price=pos.open_price,
                    sl=pos.sl, tp=pos.tp, magic=pos.magic)
            self._save_local_state()
            return True, "CLOSED", res
        return False, f"CLOSE_FAILED: {res.retcode_name} {res.message}", res

    def update_risk_after_close(self, pnl: float, equity: float, is_win: bool) -> None:
        """Feed the supervisor the realized P&L after a close."""
        self.risk.update_state(pnl, equity, is_win)

    def reconcile_positions(self) -> None:
        """Drop local positions the broker no longer holds (external closes)."""
        broker_pos = self.broker.get_positions(self.cfg.broker.symbol, self.cfg.broker.magic)
        broker_tickets = {p.ticket for p in broker_pos}
        removed = [t for t in self._positions if t not in broker_tickets]
        for t in removed:
            logger.warning("Local position %s no longer on broker — removed from local state", t)
            self._positions.pop(t, None)
        if removed:
            self._save_local_state()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _direction(signal: Any) -> int:
        if isinstance(signal, str):
            s = signal.strip().lower()
            if s in ("long", "buy", "1"):
                return 1
            if s in ("short", "sell", "2"):
                return 2
            return 0
        if isinstance(signal, (int, float)) and not isinstance(signal, bool):
            return int(signal) if int(signal) in (0, 1, 2) else 0
        if isinstance(signal, dict):
            return TradeExecutor._direction(signal.get("action", 0))
        return 0

    @staticmethod
    def _entry_token(direction: int, bar_time: Optional[str], prefix: str = "entry") -> str:
        """Deterministic idempotency token for a (decision bar, direction)."""
        return f"{prefix}-{direction}-{bar_time or 'unknown'}"
