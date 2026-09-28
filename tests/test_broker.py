"""Live broker pytest: MockBroker fills/SL-TP, Mt5Broker REQUOTE retry + idempotency + recon.

Uses a deterministic fake ``mt5`` module (never imports the real MetaTrader5).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

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


class FakeMt5:
    """Minimal scriptable stand-in for the MetaTrader5 module."""

    def __init__(self):
        self.calls: list = []
        self.retcodes: list = []
        self.check_retcode = 0
        self.bid = 2000.0
        self.ask = 2000.10
        self.positions = []
        self.last_error_code = 0

    def initialize(self, *args, **kwargs) -> bool:
        self.calls.append(("initialize", args, kwargs))
        return True

    def shutdown(self) -> None:
        self.calls.append(("shutdown", (), {}))

    def last_error(self) -> int:
        return self.last_error_code

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

    def order_check(self, request) -> SimpleNamespace:
        self.calls.append(("order_check", (), {"request": request}))
        return SimpleNamespace(retcode=self.check_retcode, comment="check")

    def order_send(self, request) -> SimpleNamespace:
        self.calls.append(("order_send", (), {"request": request}))
        retcode = self.retcodes.pop(0) if self.retcodes else TRADE_RETCODE_DONE
        if retcode in (10004, 10020, 10021):
            self.ask += 0.50
            self.bid += 0.48
        return SimpleNamespace(
            retcode=retcode,
            order=1001 + len([c for c in self.calls if c[0] == "order_send"]),
            deal=2001,
            volume=request.get("volume", 0.0),
            price=request.get("price", self.ask),
            bid=self.bid, ask=self.ask, comment="fake",
            request_id="r1", retcode_external=0,
        )


def _make_broker(fake: FakeMt5, tmp_path: Path, **kw) -> Mt5Broker:
    return Mt5Broker(
        symbol="XAUUSD", magic=234000,
        max_requote_retries=3, order_retry_delay_sec=0.0,
        idempotency_state_path=tmp_path / "idem.json",
        mt5_module=fake, **kw,
    )


def _order(side: str, volume: float = 0.1, token: str = "tok-1",
           sl: float = 1980.0, tp: float = 2030.0) -> OrderRequest:
    return OrderRequest(symbol="XAUUSD", side=side, volume=volume,
                        sl=sl, tp=tp, comment="test", magic=234000, token=token)


# --------------------------------------------------------------------------- #
# MockBroker
# --------------------------------------------------------------------------- #
def test_mock_broker_spread_and_fill_direction() -> None:
    mb = MockBroker(symbol="XAUUSD", balance=10_000.0, seed=42)
    mb.connect()
    mb.set_mid(2000.0)
    tick = mb.get_tick("XAUUSD")
    assert tick["bid"] < 2000.0 < tick["ask"]
    assert abs((tick["ask"] - tick["bid"]) - 0.00025 * 2000.0) < 0.01
    res = mb.submit_order(_order("buy", volume=0.1, sl=1990.0, tp=2020.0))
    assert res.ok and res.ticket is not None
    pos = mb.open_position(res.ticket)
    assert pos is not None and pos.sl is not None and pos.tp is not None
    assert pos.open_price > 2000.0  # buy fills at ask + adverse slippage


def test_mock_broker_sl_breach_fills_stop() -> None:
    mb = MockBroker(symbol="XAUUSD", balance=10_000.0, seed=42)
    mb.connect()
    mb.set_mid(2000.0)
    mb.submit_order(_order("buy", volume=0.1, sl=1990.0, tp=2020.0))
    fills = mb.advance(high=2010.0, low=1985.0, close=1999.0)  # breaches SL
    assert len(fills) == 1 and fills[0]["reason"] == "SL"
    assert mb.get_positions() == []
    closed = mb.closed_fills()
    assert len(closed) == 1 and closed[0]["pnl"] < 0


def test_mock_broker_stop_before_tp_on_dual_breach() -> None:
    mb = MockBroker(symbol="XAUUSD", balance=10_000.0, seed=42)
    mb.connect()
    mb.set_mid(2000.0)
    mb.submit_order(_order("buy", volume=0.1, sl=1990.0, tp=2020.0))
    mb.set_mid(2010.0)
    fills = mb.advance(high=2025.0, low=1985.0, close=2020.0)  # breaches both
    assert len(fills) == 1 and fills[0]["reason"] == "SL"


def test_mock_broker_short_fill_and_stop() -> None:
    mb = MockBroker(symbol="XAUUSD", balance=10_000.0, seed=42)
    mb.connect()
    mb.set_mid(2005.0)
    res = mb.submit_order(_order("sell", volume=0.1, sl=2010.0, tp=1980.0))
    pos = mb.open_position(res.ticket)
    assert pos is not None and pos.open_price < 2005.0  # sell at bid - slippage
    fills = mb.advance(high=2012.0, low=2000.0, close=2010.0)  # breaches short SL
    assert len(fills) == 1 and fills[0]["reason"] == "SL"


# --------------------------------------------------------------------------- #
# Mt5Broker retcodes / retry
# --------------------------------------------------------------------------- #
def test_requote_retried_then_done(tmp_path) -> None:
    fake = FakeMt5()
    b = _make_broker(fake, tmp_path)
    b.connect()
    fake.retcodes = [TRADE_RETCODE_REQUOTE, TRADE_RETCODE_DONE]
    res = b.submit_order(_order("buy", token="tok-a"))
    sends = [c for c in fake.calls if c[0] == "order_send"]
    assert res.ok and len(sends) == 2
    assert sends[1][2]["request"]["price"] > sends[0][2]["request"]["price"] + 0.4
    assert res.ticket is not None and res.ticket > 0


def test_market_closed_fails_fast(tmp_path) -> None:
    fake = FakeMt5()
    fake.retcodes = [TRADE_RETCODE_MARKET_CLOSED]
    b = _make_broker(fake, tmp_path)
    b.connect()
    res = b.submit_order(_order("buy", token="tok-b"))
    sends = [c for c in fake.calls if c[0] == "order_send"]
    assert not res.ok and len(sends) == 1
    assert res.retcode_name == "MARKET_CLOSED"


def test_requote_give_up_bounded(tmp_path) -> None:
    fake = FakeMt5()
    fake.retcodes = [TRADE_RETCODE_REQUOTE] * 10
    b = _make_broker(fake, tmp_path)
    b.connect()
    res = b.submit_order(_order("buy", token="tok-c"))
    sends = [c for c in fake.calls if c[0] == "order_send"]
    assert not res.ok and len(sends) == 4  # max_requote_retries + 1
    assert res.retcode_name == "REQUOTE"


def test_order_check_reject_blocks_order_send(tmp_path) -> None:
    fake = FakeMt5()
    fake.check_retcode = 10018
    b = _make_broker(fake, tmp_path)
    b.connect()
    res = b.submit_order(_order("buy", token="tok-d"))
    sends = [c for c in fake.calls if c[0] == "order_send"]
    assert not res.ok and len(sends) == 0


# --------------------------------------------------------------------------- #
# IdempotencyGuard
# --------------------------------------------------------------------------- #
def test_idempotency_replay_no_double_submit(tmp_path) -> None:
    fake = FakeMt5()
    fake.retcodes = [TRADE_RETCODE_DONE]
    b1 = _make_broker(fake, tmp_path)
    b1.connect()
    r1 = b1.submit_order(_order("buy", token="tok-same", volume=0.2))
    sends_a = len([c for c in fake.calls if c[0] == "order_send"])

    # Crash + restart: NEW broker, SAME token, no order_send allowed.
    fake.retcodes = []
    b2 = _make_broker(fake, tmp_path)
    b2.connect()
    r2 = b2.submit_order(_order("buy", token="tok-same", volume=0.2))
    sends_b = len([c for c in fake.calls if c[0] == "order_send"])
    assert r2.ok and r2.ticket == r1.ticket
    assert sends_b == sends_a
    assert (tmp_path / "idem.json").exists()


def test_idempotency_guard_persists(tmp_path) -> None:
    guard = IdempotencyGuard(tmp_path / "idem.json")
    guard.record("tok-1", ticket=42, symbol="XAUUSD", fill_price=2000.0, volume=0.1)
    guard2 = IdempotencyGuard(tmp_path / "idem.json")
    rec = guard2.lookup("tok-1")
    assert rec is not None and rec["ticket"] == 42


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #
def test_reconcile_clean(tmp_path) -> None:
    mb = MockBroker(symbol="XAUUSD", seed=7)
    mb.connect()
    mb.set_mid(2000.0)
    r = mb.submit_order(_order("buy", volume=0.5, sl=1990.0, tp=2030.0, token="tok-rec"))
    assert r.ok
    broker_pos = mb.get_positions()
    rec = mb.reconcile(local_positions=broker_pos)
    assert rec.ok


def test_reconcile_detects_missing_and_unknown(tmp_path) -> None:
    mb = MockBroker(symbol="XAUUSD", seed=7)
    mb.connect()
    mb.set_mid(2000.0)
    r = mb.submit_order(_order("buy", volume=0.5, sl=1990.0, tp=2030.0, token="tok-rec"))
    broker_pos = mb.get_positions()
    ghost = PositionInfo(ticket=999, symbol="XAUUSD", side="buy", volume=0.1,
                         open_price=2000.0, sl=1990.0, tp=2030.0, magic=234000)
    rec_miss = mb.reconcile(local_positions=broker_pos + [ghost])
    assert not rec_miss.ok and len(rec_miss.missing_in_broker) == 1
    rec_unk = mb.reconcile(local_positions=[])
    assert not rec_unk.ok and len(rec_unk.unknown_in_broker) == 1


def test_reconcile_detects_volume_drift(tmp_path) -> None:
    mb = MockBroker(symbol="XAUUSD", seed=7)
    mb.connect()
    mb.set_mid(2000.0)
    r = mb.submit_order(_order("buy", volume=0.5, sl=1990.0, tp=2030.0, token="tok-rec"))
    bp = mb.get_positions()[0]
    drifted = [PositionInfo(ticket=bp.ticket, symbol="XAUUSD", side="buy",
                            volume=bp.volume + 0.1, open_price=bp.open_price,
                            sl=1990.0, tp=2030.0, magic=234000)]
    rec = mb.reconcile(local_positions=drifted)
    assert not rec.ok and len(rec.mismatched) == 1


def test_mt5_broker_reconcile_via_fake(tmp_path) -> None:
    fake = FakeMt5()
    fake.positions = [
        SimpleNamespace(ticket=100, symbol="XAUUSD", type=0, volume=0.1,
                        price_open=2000.0, sl=1990.0, tp=2030.0, magic=234000),
    ]
    b = _make_broker(fake, tmp_path)
    b.connect()
    local = [PositionInfo(ticket=100, symbol="XAUUSD", side="buy", volume=0.1,
                          open_price=2000.0, sl=1990.0, tp=2030.0, magic=234000)]
    rec = b.reconcile(local)
    assert rec.ok


def test_mt5_broker_reconcile_detects_drift(tmp_path) -> None:
    fake = FakeMt5()
    fake.positions = []
    b = _make_broker(fake, tmp_path)
    b.connect()
    local = [PositionInfo(ticket=100, symbol="XAUUSD", side="buy", volume=0.1,
                          open_price=2000.0, sl=1990.0, tp=2030.0, magic=234000)]
    rec = b.reconcile(local)
    assert not rec.ok and len(rec.missing_in_broker) == 1
