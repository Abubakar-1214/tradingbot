"""
scripts/verify_broker.py — acceptance gate for the rebuilt live broker layer.

Proves (P0-6 fixes):
  1. MockBroker fills are real: buys at ask+slippage, sells at bid-slippage;
     SL/TP are honored on bar advance; stop fills before take-profit when a
     bar breaches both (conservative, matches backtesting.py).
  2. Mt5Broker maps retcodes correctly, retries REQUOTE (10004) with a fresh
     price and backoff up to ``max_requote_retries``, then gives up (bounded —
     never loops forever).
  3. The persisted IdempotencyGuard dedupes a replayed token WITHOUT a second
     order_send (crash-between-send-and-confirm cannot double-submit).
  4. Startup reconciliation detects drift between broker positions and local
     state (missing / unknown / mismatched).
  5. Every filled order in MockBroker carries non-null SL and TP.

The real MetaTrader5 package is never imported: tests inject a deterministic
fake ``mt5`` module so every retcode path is exercised on CI machines.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from live.broker import (
    TRADE_RETCODE_DONE,
    TRADE_RETCODE_MARKET_CLOSED,
    TRADE_RETCODE_REQUOTE,
    OrderRequest,
    PositionInfo,
)
from live.idempotency import IdempotencyGuard
from live.mock_broker import MockBroker
from live.mt5_broker import Mt5Broker

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


# --------------------------------------------------------------------------- #
# Deterministic fake MT5 API (records calls, scriptable retcodes)
# --------------------------------------------------------------------------- #
class FakeMt5:
    """Minimal scriptable stand-in for the MetaTrader5 module."""

    def __init__(self):
        self.calls: list = []
        self.retcodes: list = []          # order_send retcodes consumed in order
        self.check_retcode = 0            # order_check result (0 = OK)
        self.bid = 2000.0
        self.ask = 2000.10
        self.positions = []
        self.last_error_code = 0

    # -- connection -- #
    def initialize(self, *args, **kwargs) -> bool:
        self.calls.append(("initialize", args, kwargs))
        return True

    def shutdown(self) -> None:
        self.calls.append(("shutdown", (), {}))

    def last_error(self) -> int:
        return self.last_error_code

    # -- reads -- #
    def account_info(self) -> SimpleNamespace:
        self.calls.append(("account_info", (), {}))
        return SimpleNamespace(balance=10_000.0, equity=10_000.0, margin_free=9_000.0)

    def symbol_info_tick(self, symbol: str) -> SimpleNamespace:
        self.calls.append(("symbol_info_tick", (symbol,), {}))
        return SimpleNamespace(bid=self.bid, ask=self.ask, last=self.bid,
                               spread=int((self.ask - self.bid) * 10_000))

    def positions_get(self, symbol=None, ticket=None):
        self.calls.append(("positions_get", (symbol, ticket), {}))
        if ticket is not None:
            return [p for p in self.positions if p.ticket == ticket]
        if symbol is not None:
            return [p for p in self.positions if p.symbol == symbol]
        return list(self.positions)

    # -- orders -- #
    def order_check(self, request) -> SimpleNamespace:
        self.calls.append(("order_check", (), {"request": request}))
        return SimpleNamespace(retcode=self.check_retcode, comment="check")

    def order_send(self, request) -> SimpleNamespace:
        self.calls.append(("order_send", (), {"request": request}))
        retcode = self.retcodes.pop(0) if self.retcodes else TRADE_RETCODE_DONE
        # On transient retcodes the broker re-quotes: move the market so the
        # retry genuinely fetches a fresh (different) price.
        if retcode in (10004, 10020, 10021):
            self.ask += 0.50
            self.bid += 0.48
        return SimpleNamespace(
            retcode=retcode,
            order=1001 + len([c for c in self.calls if c[0] == "order_send"]),
            deal=2001,
            volume=request.get("volume", 0.0),
            price=request.get("price", self.ask),
            bid=self.bid,
            ask=self.ask,
            comment="fake",
            request_id="r1",
            retcode_external=0,
        )


def _make_broker(fake: FakeMt5, **kw) -> Mt5Broker:
    tmpdir = Path(tempfile.mkdtemp(prefix="verify_broker_"))
    return Mt5Broker(
        symbol="XAUUSD",
        magic=234000,
        max_requote_retries=3,
        order_retry_delay_sec=0.0,
        idempotency_state_path=tmpdir / "idem.json",
        mt5_module=fake,
        **kw,
    )


def _order(side: str, volume: float = 0.1, token: str = "tok-1",
           sl: float = 1980.0, tp: float = 2030.0) -> OrderRequest:
    return OrderRequest(symbol="XAUUSD", side=side, volume=volume,
                        sl=sl, tp=tp, comment="test", magic=234000, token=token)


def main() -> int:
    print("=" * 70)
    print("verify_broker.py — rebuilt live broker acceptance gate")
    print("=" * 70)

    # --- [1] MockBroker fills + SL/TP honoring ------------------------------ #
    print("\n[1] MockBroker: real fills, SL/TP honored")
    mb = MockBroker(symbol="XAUUSD", balance=10_000.0, seed=42)
    mb.connect()
    mb.set_mid(2000.0)
    tick = mb.get_tick("XAUUSD")
    check("bid < mid < ask (spread modeled)", tick["bid"] < 2000.0 < tick["ask"])
    check("round-trip spread from config", abs((tick["ask"] - tick["bid"]) - 0.00025 * 2000.0) < 0.01)

    req = _order("buy", volume=0.1, sl=1990.0, tp=2020.0)
    res = mb.submit_order(req)
    check("buy order accepted", res.ok and res.ticket is not None, str(res))
    pos = mb.open_position(res.ticket)
    check("SL attached to filled order", pos is not None and pos.sl is not None, str(pos))
    check("TP attached to filled order", pos is not None and pos.tp is not None, str(pos))
    check("buy fills at ask + adverse slippage",
          pos is not None and pos.open_price > 2000.0, f"fill={pos.open_price if pos else None}")

    # SL breach inside a bar -> stop fill
    fills = mb.advance(high=2010.0, low=1985.0, close=1999.0)  # breaches 1990 SL
    check("SL breach triggers stop fill", len(fills) == 1 and fills[0]["reason"] == "SL",
          str(fills))
    check("no position remains after stop", mb.get_positions() == [])
    closed = mb.closed_fills()
    check("SL fill is real PnL", len(closed) == 1 and closed[0]["pnl"] < 0,
          str(closed))

    # TP breach -> tp fill, and stop-when-both-breached fills stop first
    req2 = _order("buy", volume=0.1, sl=1990.0, tp=2020.0, token="tok-2")
    res2 = mb.submit_order(req2)
    mb.set_mid(2010.0)
    # single bar 1985..2025 breaches BOTH -> stop (1990) fills first
    fills2 = mb.advance(high=2025.0, low=1985.0, close=2020.0)
    check("stop fills before take-profit when bar breaches both",
          len(fills2) == 1 and fills2[0]["reason"] == "SL", str(fills2))

    # TP-only breach -> tp fill
    req3 = _order("buy", volume=0.1, sl=1990.0, tp=2015.0, token="tok-3")
    res3 = mb.submit_order(req3)
    mb.set_mid(2010.0)
    fills3 = mb.advance(high=2016.0, low=2008.0, close=2012.0)
    check("TP breach triggers tp fill", len(fills3) == 1 and fills3[0]["reason"] == "TP",
          str(fills3))

    # short-side fills + stops
    req4 = _order("sell", volume=0.1, sl=2010.0, tp=1980.0, token="tok-4")
    mb.set_mid(2005.0)
    res4 = mb.submit_order(req4)
    pos4 = mb.open_position(res4.ticket)
    check("sell fills at bid - slippage",
          pos4 is not None and pos4.open_price < 2005.0, f"fill={pos4.open_price if pos4 else None}")
    fills4 = mb.advance(high=2012.0, low=2000.0, close=2010.0)  # breaches short SL 2010
    check("short SL honored", len(fills4) == 1 and fills4[0]["reason"] == "SL", str(fills4))

    mb.disconnect()

    # --- [2] Mt5Broker retcode mapping + REQUOTE bounded retry -------------- #
    print("\n[2] Mt5Broker: retcodes, REQUOTE retry, give-up")
    fake = FakeMt5()
    b = _make_broker(fake)
    b.connect()
    # order_check passes (0), first send REQUOTE, second DONE
    fake.retcodes = [TRADE_RETCODE_REQUOTE, TRADE_RETCODE_DONE]
    res = b.submit_order(_order("buy", token="tok-a"))
    sends = [c for c in fake.calls if c[0] == "order_send"]
    check("REQUOTE retried then DONE", res.ok and len(sends) == 2, f"ok={res.ok} sends={len(sends)}")
    check("retry re-quoted a fresh price",
          sends[1][2]["request"]["price"] > sends[0][2]["request"]["price"] + 0.4,
          f"attempt1={sends[0][2]['request']['price']} attempt2={sends[1][2]['request']['price']}")
    check("success carries ticket", res.ticket is not None and res.ticket > 0, str(res))

    # MARKET_CLOSED fails fast (no retry)
    fake2 = FakeMt5()
    fake2.retcodes = [TRADE_RETCODE_MARKET_CLOSED]
    b2 = _make_broker(fake2)
    b2.connect()
    res2 = b2.submit_order(_order("buy", token="tok-b"))
    sends2 = [c for c in fake2.calls if c[0] == "order_send"]
    check("MARKET_CLOSED fails fast (no retry)", not res2.ok and len(sends2) == 1
          and res2.retcode_name == "MARKET_CLOSED", str(res2))

    # persistent REQUOTE -> bounded give-up after max_requote_retries
    fake3 = FakeMt5()
    fake3.retcodes = [TRADE_RETCODE_REQUOTE] * 10
    b3 = _make_broker(fake3)
    b3.connect()
    res3 = b3.submit_order(_order("buy", token="tok-c"))
    sends3 = [c for c in fake3.calls if c[0] == "order_send"]
    check("REQUOTE give-up after exactly max_requote_retries+1 attempts",
          not res3.ok and len(sends3) == 4, f"ok={res3.ok} sends={len(sends3)} (expected 4)")
    check("give-up result names REQUOTE", res3.retcode_name == "REQUOTE", str(res3))

    # order_check hard reject -> no order_send at all
    fake4 = FakeMt5()
    fake4.check_retcode = 10018
    b4 = _make_broker(fake4)
    b4.connect()
    res4 = b4.submit_order(_order("buy", token="tok-d"))
    sends4 = [c for c in fake4.calls if c[0] == "order_send"]
    check("order_check reject blocks order_send", not res4.ok and len(sends4) == 0,
          str(res4))

    # --- [3] IdempotencyGuard persisted dedupe ------------------------------ #
    print("\n[3] IdempotencyGuard: persisted dedupe (no double-submit)")
    fake5 = FakeMt5()
    tmp5 = Path(tempfile.mkdtemp(prefix="verify_idem_"))
    b5 = _make_broker(fake5)
    b5._idem = IdempotencyGuard(tmp5 / "idem.json")  # fresh guard, same path used later
    b5.connect()
    fake5.retcodes = [TRADE_RETCODE_DONE]
    r1 = b5.submit_order(_order("buy", token="tok-same", volume=0.2))
    sends5a = len([c for c in fake5.calls if c[0] == "order_send"])
    # Simulate crash + restart: NEW broker instance, SAME token
    b6 = _make_broker(fake5)
    b6._idem = IdempotencyGuard(tmp5 / "idem.json")
    b6.connect()
    fake5.retcodes = []  # no more order_send allowed — replay must short-circuit
    r2 = b6.submit_order(_order("buy", token="tok-same", volume=0.2))
    sends5b = len([c for c in fake5.calls if c[0] == "order_send"])
    check("replay returns recorded ticket", r2.ok and r2.ticket == r1.ticket,
          f"r1={r1.ticket} r2={r2.ticket}")
    check("replay did NOT call order_send again",
          sends5b == sends5a, f"before={sends5a} after={sends5b}")
    check("guard persisted to disk", (tmp5 / "idem.json").exists())

    # --- [4] startup reconciliation ----------------------------------------- #
    print("\n[4] startup reconciliation (broker vs local state)")
    mbR = MockBroker(symbol="XAUUSD", seed=7)
    mbR.connect()
    mbR.set_mid(2000.0)
    r = mbR.submit_order(_order("buy", volume=0.5, sl=1990.0, tp=2030.0, token="tok-rec"))
    broker_pos = mbR.get_positions()
    # Clean: local state matches broker
    rec_ok = mbR.reconcile(local_positions=broker_pos)
    check("clean state reconciles", rec_ok.ok, rec_ok.describe())
    # Missing in broker: local wants a position broker doesn't have
    ghost = PositionInfo(ticket=999, symbol="XAUUSD", side="buy", volume=0.1,
                         open_price=2000.0, sl=1990.0, tp=2030.0, magic=234000)
    rec_miss = mbR.reconcile(local_positions=broker_pos + [ghost])
    check("local-only position detected as missing in broker",
          not rec_miss.ok and len(rec_miss.missing_in_broker) == 1, rec_miss.describe())
    # Unknown in broker: broker has a position local doesn't know
    rec_unk = mbR.reconcile(local_positions=[])
    check("broker-only position detected as unknown",
          not rec_unk.ok and len(rec_unk.unknown_in_broker) == 1, rec_unk.describe())
    # Mismatch: volume drift
    drifted = [PositionInfo(ticket=broker_pos[0].ticket, symbol="XAUUSD",
                            side="buy", volume=broker_pos[0].volume + 0.1,
                            open_price=broker_pos[0].open_price, sl=1990.0, tp=2030.0,
                            magic=234000)]
    rec_drift = mbR.reconcile(local_positions=drifted)
    check("volume drift detected as mismatched",
          not rec_drift.ok and len(rec_drift.mismatched) == 1, rec_drift.describe())

    # --- [5] every filled order in MockBroker carries SL+TP ----------------- #
    print("\n[5] all MockBroker fills carry SL/TP")
    mb5 = MockBroker(symbol="XAUUSD", seed=11)
    mb5.connect()
    for i in range(3):
        mb5.set_mid(2000.0 + i)
        mb5.submit_order(_order("buy", volume=0.1, sl=1990.0, tp=2020.0,
                                token=f"all-{i}"))
    all_pos = mb5.get_positions()
    check("all open mock positions have SL", all(p.sl is not None for p in all_pos))
    check("all open mock positions have TP", all(p.tp is not None for p in all_pos))

    print("\n" + "=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
