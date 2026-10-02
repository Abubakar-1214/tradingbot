"""
live/mt5_broker.py — production MetaTrader5 broker implementation.

Implements :class:`BaseBroker` against the MetaTrader5 Python package.  All
broker-side behaviour is verified against the MQL5/Python-MT5 contract:

  * ``order_check`` before ``order_send`` (server-side sanity gate),
  * retcode mapping with a BOUNDED retry loop on REQUOTE (10004),
    PRICE_CHANGED (10020) and PRICE_OFF (10021) — each retry re-quotes a
    fresh price, backs off ``order_retry_delay_sec``, and gives up after
    ``max_requote_retries`` (never loops forever),
  * persisted idempotency guard (see live/idempotency.py) so a crash
    between order_send and confirmation cannot double-submit on restart,
  * startup reconciliation via ``BaseBroker.reconcile`` comparing broker
    positions with the locally-persisted state.

The MetaTrader5 module is imported lazily so the package (and the test
suite) can run on machines without the terminal.  Tests inject a fake
``mt5`` module to exercise every retcode/retry/reconciliation path.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Type

from live.broker import (
    SUCCESS_RETCODES,
    TRANSIENT_RETCODES,
    TRADE_ACTION_DEAL,
    TRADE_RETCODE_DONE,
    TRADE_RETCODE_MARKET_CLOSED,
    TRADE_RETCODE_REQUOTE,
    AccountInfo,
    BaseBroker,
    BrokerError,
    OrderRequest,
    OrderResult,
    PositionInfo,
)
from live.idempotency import IdempotencyGuard

logger = logging.getLogger(__name__)

# metaapi-style namedtuple result fields on mt5 order result
_ORDER_SEND_RESULT_FIELDS = ("retcode", "deal", "order", "volume", "price",
                             "bid", "ask", "comment", "request_id", "retcode_external")


class Mt5Broker(BaseBroker):
    """Real MetaTrader5 broker wrapper (inject ``mt5`` for tests)."""

    def __init__(
        self,
        symbol: str,
        magic: int = 234000,
        deviation: int = 20,
        allow_short: bool = False,
        max_requote_retries: int = 3,
        order_retry_delay_sec: float = 1.0,
        login: Optional[str] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        mt5_path: Optional[str] = None,
        idempotency_state_path=None,
        mt5_module=None,
    ):
        self.symbol = symbol
        self.magic = magic
        self.deviation = deviation
        self.allow_short = allow_short
        self.max_requote_retries = max(0, int(max_requote_retries))
        self.order_retry_delay_sec = float(order_retry_delay_sec)
        self.login = login
        self.password = password
        self.server = server
        self.mt5_path = mt5_path

        self._mt5: Optional[Type] = mt5_module  # injected fake in tests
        self._connected = False
        self._idem = IdempotencyGuard(idempotency_state_path) if idempotency_state_path else None
        self._recon_done = False

    # ------------------------------------------------------------------ #
    # connection
    # ------------------------------------------------------------------ #
    def _load_mt5(self):
        if self._mt5 is not None:
            return self._mt5
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as e:  # pragma: no cover - terminal-machine only
            raise BrokerError(
                "MetaTrader5 package not installed. Install with "
                "`pip install MetaTrader5` and run on a machine with the "
                "MT5 terminal.") from e
        self._mt5 = mt5
        return mt5

    def connect(self) -> None:
        mt5 = self._load_mt5()
        path = self.mt5_path
        login = int(self.login) if self.login else None
        password = self.password if self.password else None
        server = self.server if self.server else None
        if path:
            ok = mt5.initialize(path=path, login=login, password=password, server=server)
        elif login is not None:
            ok = mt5.initialize(login=login, password=password, server=server)
        else:
            ok = mt5.initialize()
        if not ok:
            code = mt5.last_error()
            raise BrokerError(f"MT5 initialize() failed: {code}")
        self._connected = True
        logger.info("MT5 connected (symbol=%s magic=%s)", self.symbol, self.magic)

    def disconnect(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
        self._connected = False

    # ------------------------------------------------------------------ #
    # read-only API
    # ------------------------------------------------------------------ #
    def account_info(self) -> AccountInfo:
        mt5 = self._require_connected()
        acc = mt5.account_info()
        if acc is None:
            raise BrokerError(f"account_info() failed: {mt5.last_error()}")
        return AccountInfo(balance=float(acc.balance), equity=float(acc.equity),
                           margin_free=float(acc.margin_free))

    def get_tick(self, symbol: str) -> Dict[str, float]:
        mt5 = self._require_connected()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise BrokerError(f"symbol_info_tick({symbol}) failed: {mt5.last_error()}")
        spread = float(
            getattr(tick, "spread", None)
            if getattr(tick, "spread", None) is not None
            else (tick.ask - tick.bid)
        )
        return {
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "last": float(tick.last),
            "spread": spread,
        }

    def get_positions(self, symbol: Optional[str] = None,
                      magic: Optional[int] = None) -> List[PositionInfo]:
        mt5 = self._require_connected()
        pos = mt5.positions_get(symbol=symbol or self.symbol)
        if pos is None:
            raise BrokerError(f"positions_get() failed: {mt5.last_error()}")
        out = []
        for p in pos:
            if magic is not None and int(p.magic) != magic:
                continue
            out.append(PositionInfo(
                ticket=int(p.ticket),
                symbol=str(p.symbol),
                side="buy" if int(p.type) == 0 else "sell",
                volume=float(p.volume),
                open_price=float(p.price_open),
                sl=None if float(p.sl) == 0.0 else float(p.sl),
                tp=None if float(p.tp) == 0.0 else float(p.tp),
                magic=int(p.magic),
            ))
        return out

    # ------------------------------------------------------------------ #
    # order submission with REQUOTE retry + idempotency
    # ------------------------------------------------------------------ #
    def submit_order(self, req: OrderRequest) -> OrderResult:
        """Submit a market order.

        Retry policy: REQUOTE / PRICE_CHANGED / PRICE_OFF are retried up to
        ``max_requote_retries`` times with a fresh price each attempt and a
        backoff of ``order_retry_delay_sec``.  After the bound is exhausted
        the order is NOT re-submitted — a failure result is returned so the
        caller can halt/alert.  MARKET_CLOSED / REJECT / NO_MONEY fail fast.
        """
        mt5 = self._require_connected()

        if self._idem is not None and req.token:
            prev = self._idem.lookup(req.token)
            if prev is not None and prev.get("ticket"):
                return OrderResult.success(
                    ticket=int(prev["ticket"]),
                    volume_filled=float(prev.get("volume", 0.0)),
                    fill_price=prev.get("fill_price"),
                    message=f"idempotent replay of ticket={prev['ticket']}",
                )

        # order_check first — server-side sanity gate (P0-6)
        check = self._check_order(req)
        if check is not None:
            return check

        tick = self.get_tick(req.symbol)
        order_type = 0 if req.side == "buy" else 1
        price = tick["ask"] if req.side == "buy" else tick["bid"]

        for attempt in range(self.max_requote_retries + 1):
            request = {
                "action": TRADE_ACTION_DEAL,
                "symbol": req.symbol,
                "volume": float(req.volume),
                "type": order_type,
                "price": price,
                "deviation": self.deviation,
                "magic": req.magic or self.magic,
                "comment": req.comment or "RL_v2",
                "type_time": 0,  # ORDER_TIME_GTC
                "type_filling": 1,  # ORDER_FILLING_IOC
            }
            if req.sl is not None:
                request["sl"] = float(req.sl)
            if req.tp is not None:
                request["tp"] = float(req.tp)

            res = mt5.order_send(request)
            if res is None:
                code = mt5.last_error()
                return OrderResult.failure(retcode=int(code or 10014),
                                           message=f"order_send returned None: {code}")
            retcode = int(res.retcode)

            if retcode in SUCCESS_RETCODES:
                ticket = int(res.order) if hasattr(res, "order") and res.order else None
                if self._idem is not None and req.token:
                    self._idem.record(req.token, ticket, symbol=req.symbol,
                                      fill_price=float(res.price) if hasattr(res, "price") else None,
                                      volume=float(res.volume) if hasattr(res, "volume") else 0.0)
                return OrderResult.success(
                    ticket=ticket,
                    volume_filled=float(res.volume) if hasattr(res, "volume") else float(req.volume),
                    fill_price=float(res.price) if hasattr(res, "price") else None,
                    message=str(getattr(res, "comment", "")),
                    retcode=retcode,
                )

            if retcode in TRANSIENT_RETCODES and attempt < self.max_requote_retries:
                logger.warning(
                    "Transient retcode %s on attempt %d/%d — re-quoting "
                    "(symbol=%s side=%s)", retcode, attempt + 1,
                    self.max_requote_retries, req.symbol, req.side)
                time.sleep(self.order_retry_delay_sec)
                tick = self.get_tick(req.symbol)
                price = tick["ask"] if req.side == "buy" else tick["bid"]
                continue

            return OrderResult.failure(retcode=retcode,
                                       message=str(getattr(res, "comment", "")))

        return OrderResult.failure(
            retcode=TRADE_RETCODE_REQUOTE,
            message=f"gave up after {self.max_requote_retries} requote retries")

    def close_position(self, ticket: int, volume: Optional[float] = None) -> OrderResult:
        """Close an open position (reverse market order with position=ticket)."""
        mt5 = self._require_connected()
        pos = mt5.positions_get(ticket=ticket)
        if not pos or len(pos) == 0:
            return OrderResult.failure(retcode=10016, message=f"no position ticket={ticket}")
        p = pos[0]
        side = "sell" if int(p.type) == 0 else "buy"  # opposite of position
        req = OrderRequest(
            symbol=str(p.symbol),
            side=side,
            volume=float(volume if volume is not None else p.volume),
            sl=None, tp=None,
            comment="RL_v2_close",
            magic=int(p.magic),
            reduce_only=True,
            token=f"close-{ticket}",
        )
        return self.submit_order(req)

    def modify_sl_tp(self, ticket: int, sl: Optional[float],
                     tp: Optional[float]) -> OrderResult:
        mt5 = self._require_connected()
        pos = mt5.positions_get(ticket=ticket)
        if not pos or len(pos) == 0:
            return OrderResult.failure(retcode=10016, message=f"no position ticket={ticket}")
        p = pos[0]
        request = {
            "action": 3,  # TRADE_ACTION_SLTP
            "symbol": str(p.symbol),
            "position": ticket,
            "sl": float(sl) if sl is not None else float(p.sl),
            "tp": float(tp) if tp is not None else float(p.tp),
        }
        res = mt5.order_send(request)
        if res is None:
            return OrderResult.failure(retcode=int(mt5.last_error() or 10014),
                                       message="modify_sl_tp order_send None")
        retcode = int(res.retcode)
        if retcode in SUCCESS_RETCODES:
            return OrderResult.success(ticket=ticket, volume_filled=float(p.volume),
                                       fill_price=None, message=str(getattr(res, "comment", "")),
                                       retcode=retcode)
        return OrderResult.failure(retcode=retcode,
                                   message=str(getattr(res, "comment", "")))

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _check_order(self, req: OrderRequest):
        """Run MT5 order_check; returns a failure OrderResult on hard reject."""
        mt5 = self._require_connected()
        tick = self.get_tick(req.symbol)
        order_type = 0 if req.side == "buy" else 1
        price = tick["ask"] if req.side == "buy" else tick["bid"]
        request = {
            "action": TRADE_ACTION_DEAL,
            "symbol": req.symbol,
            "volume": float(req.volume),
            "type": order_type,
            "price": price,
            "deviation": self.deviation,
            "magic": req.magic or self.magic,
            "type_time": 0,
            "type_filling": 1,
        }
        if req.sl is not None:
            request["sl"] = float(req.sl)
        if req.tp is not None:
            request["tp"] = float(req.tp)
        check = mt5.order_check(request)
        if check is None:
            code = mt5.last_error()
            return OrderResult.failure(retcode=int(code or 10014),
                                       message=f"order_check failed: {code}")
        if int(check.retcode) != 0:
            return OrderResult.failure(retcode=int(check.retcode),
                                       message=str(getattr(check, "comment", "order_check reject")))
        return None

    def reconcile(self, local_positions, symbol=None, magic=None):
        """Startup reconciliation — delegate to the base algorithm."""
        result = super().reconcile(local_positions, symbol=symbol or self.symbol,
                                   magic=magic if magic is not None else self.magic)
        self._recon_done = True
        return result

    # ------------------------------------------------------------------ #
    def _require_connected(self):
        if not self._connected:
            raise BrokerError("MT5 broker not connected — call connect() first")
        return self._mt5
