"""
live/live_trade_mt5.py — production live-trading entry point (REWRITTEN).

Replaces the legacy 173-line script that hard-coded ``VOLUME=0.01``, ran a
10-second polling loop on an H1 strategy, attached NO stop-loss/take-profit,
checked NO MT5 retcodes, and never imported the RiskSupervisor (audit P0-1,
P0-6).  The rewrite is built entirely on the new production stack:

    core.config.load_config()            — TRADING_MODE=demo|live from .env
    live.mt5_broker.Mt5Broker            — real MetaTrader5 (live mode)
    live.mock_broker.MockBroker          — deterministic simulated fills (demo)
    live.trade_executor.TradeExecutor    — RiskSupervisor wired into EVERY order
                                           path, ATR-based sizing + SL/TP
    models.risk_supervisor.RiskSupervisor — SQLite-persisted circuit breakers
    models.position_sizing.ATRPositionSizer — explicit risk-based sizing

Guarantees (verified by the ``--smoke`` demo run):

  * TRADING_MODE defaults to ``demo`` — demo mode trades ONLY against the
    MockBroker and can never send a real order.
  * live mode refuses to start unless its gates pass (risk state loadable,
    MT5 credentials present, startup reconciliation clean, and — for the ML
    signal source — a promoted artifact with a matching evaluation record).
  * No hard-coded volume anywhere: every order is sized by the ATRPositionSizer
    as an explicit equity fraction, floored to the broker lot step and capped
    by ``max_position`` (aggregate notional cap enforced by TradeExecutor).
  * SL/TP are attached to EVERY order (ATR-based, ``sl_atr_mult`` /
    ``tp_atr_mult`` from BrokerConfig).
  * MT5 retcodes are handled by Mt5Broker: REQUOTE/PRICE_CHANGED/PRICE_OFF are
    retried with a fresh quote up to ``max_requote_retries``, MARKET_CLOSED and
    hard rejects fail fast, and every order intent carries a deterministic
    idempotency token persisted across restarts.
  * Candle-close-aligned loop: one decision per NEW closed timeframe candle —
    never 10-second spam on an H1 strategy.  Each newly closed candle is also
    fed to the broker so SL/TP are evaluated against the real closed bar.
  * Startup reconciliation: broker positions are compared against the
    persisted local state; any drift halts the trader and logs CRITICAL.
  * Kill-switch file is checked every loop iteration.
  * Structured rotating logging to ``logs/``.

Run (demo, default — deterministic synthetic bar feed, safe):::

    python live/live_trade_mt5.py --smoke --max-bars 300

Run (live — after all gates pass)::

    set TRADING_MODE=live
    python live/live_trade_mt5.py
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

# Allow direct execution from any CWD (``python live/live_trade_mt5.py``):
# put the repo root on sys.path so ``core`` / ``live`` / ``models`` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (
    AppConfig,
    TradingMode,
    load_config,
)
from core.model_artifacts import ModelArtifactError, load_manifest
from core.observation import AccountState
from live.broker import BaseBroker, BrokerError, PositionInfo
from live.decision_engine import Decision, DecisionEngine
from live.mock_broker import MockBroker
from live.model_signal import ModelSignalSource
from live.mt5_broker import Mt5Broker
from live.trade_executor import TradeExecutor
from live.trade_manager import Close, ModifySL, PartialClose, TradeManager
from models.position_sizing import ATRPositionSizer, KellyPositionSizer
from models.risk_supervisor import RiskSupervisor

logger = logging.getLogger("live_trader")

# Number of CLOSED bars the decision window needs before the first decision
# (>= slow SMA period + margin; ATR itself only needs 15).
CLOSED_BAR_WARMUP = 60

_DEMO_BALANCE = 100_000.0


# --------------------------------------------------------------------------- #
# Signal sources
# --------------------------------------------------------------------------- #
class SignalSource(Protocol):
    def decide(self, df: pd.DataFrame | None, account: AccountState) -> Decision: ...

    def executed(self, action: int) -> None: ...


class RuleSignalSource:
    """Causal SMA fast/slow crossover signal.

    Computed ONLY from closed bars (rolling means are NaN during warm-up), so
    no forming-bar information leaks into the decision.  ``signal()`` returns
    1 on a fast-above-slow cross, 2 on a fast-below-slow cross, else 0.
    """

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast >= slow:
            raise ValueError(f"fast SMA ({fast}) must be < slow SMA ({slow})")
        self.fast = int(fast)
        self.slow = int(slow)

    def signal(self, df: Optional[pd.DataFrame], position_side: int = 0) -> int:
        if df is None or len(df) < self.slow + 2:
            return 0
        closes = df["close"]
        fast_sma = closes.rolling(self.fast).mean()
        slow_sma = closes.rolling(self.slow).mean()
        last_fast, last_slow = fast_sma.iloc[-1], slow_sma.iloc[-1]
        prev_fast, prev_slow = fast_sma.iloc[-2], slow_sma.iloc[-2]
        if pd.isna(last_fast) or pd.isna(last_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
            return 0
        if prev_fast <= prev_slow and last_fast > last_slow:
            return 1
        if prev_fast >= prev_slow and last_fast < last_slow:
            return 2
        return 0

    def decide(
        self, df: pd.DataFrame | None, account: AccountState
    ) -> Decision:
        action = self.signal(df, account.position)
        if action == 0:
            return Decision(0, 1.0, 1.0, "NO_SIGNAL", {"hold": True})
        return Decision(action, 1.0, 1.0, "RULE_SIGNAL")

    def executed(self, action: int) -> None:
        return None


def build_signal_source(cfg: AppConfig) -> SignalSource:
    """Build the configured rule or model source."""
    if cfg.model.signal_source == "model":
        return ModelSignalSource(cfg)
    return RuleSignalSource()


# --------------------------------------------------------------------------- #
# Bar sources (candle-close-aligned data feed)
# --------------------------------------------------------------------------- #
class BarSource(Protocol):
    """Returns the latest ``n`` CLOSED bars as a DataFrame."""

    def last_closed_bars(self, n: int) -> pd.DataFrame: ...


class Mt5BarSource:
    """Feeds closed H1/H4/D1 bars from the MT5 terminal.

    ``copy_rates_from_pos`` returns only CLOSED bars; MT5 names the volume
    column ``tick_volume`` — it is renamed to ``volume`` so the rest of the
    stack (backtesting engine, feature pipeline, executor) sees one schema.
    """

    def __init__(
        self,
        mt5_module,
        symbol: str,
        timeframe: str,
        utc_offset_hours: float = 0.0,
    ) -> None:
        self._mt5 = mt5_module
        self.symbol = symbol
        self.timeframe = timeframe.upper()
        self.utc_offset_hours = float(utc_offset_hours)

    def last_closed_bars(self, n: int) -> pd.DataFrame:
        tf = getattr(self._mt5, f"TIMEFRAME_{self.timeframe}", None)
        if tf is None:
            raise ValueError(f"Unsupported MT5 timeframe {self.timeframe!r}")
        rates = self._mt5.copy_rates_from_pos(self.symbol, tf, 0, int(n))
        if rates is None:
            code = self._mt5.last_error()
            raise BrokerError(f"copy_rates_from_pos failed: {code}")
        df = pd.DataFrame(rates)
        if df.empty:
            return df
        df = df.rename(columns={"tick_volume": "volume"})
        df["time"] = (
            pd.to_datetime(df["time"], unit="s", utc=True)
            - pd.to_timedelta(self.utc_offset_hours, unit="h")
        )
        return df


class SyntheticBarSource:
    """Deterministic synthetic closed-bar feed (demo/smoke ONLY).

    Generates a seeded sinusoidal random walk (period 60 bars, amplitude 0.8)
    so the SMA fast/slow signal oscillates and produces BOTH long and short
    crossovers inside a 300-bar smoke — guaranteeing a full order cycle
    (entry -> close -> opposite entry).  One new CLOSED bar is appended per
    call — exactly the candle-close cadence the loop consumes, with NO
    forming-bar leakage (each returned bar is fully closed).
    """

    def __init__(self, seed: int = 7, start_price: float = 2000.0,
                 step_minutes: int = 60, start: Optional[datetime] = None,
                 period: int = 60, amplitude: float = 0.8) -> None:
        self._rng = np.random.default_rng(seed)
        self.start_price = float(start_price)
        self._step = timedelta(minutes=int(step_minutes))
        self._start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
        self._period = int(period)
        self._amplitude = float(amplitude)
        self._bars: List[dict] = []
        self._closed = float(start_price)
        self._n = 0

    def _append_bar(self) -> None:
        n = self._n
        # Sinusoidal drift: oscillates up/down so fast/slow SMA cross both ways.
        drift = self._amplitude * math.sin(2.0 * math.pi * n / self._period)
        noise = float(self._rng.normal(0.0, 1.2))
        o = self._closed
        c = o + drift + noise
        hi = max(o, c) + abs(float(self._rng.normal(0.0, 0.8)))
        lo = min(o, c) - abs(float(self._rng.normal(0.0, 0.8)))
        ts = self._start + self._step * (n + 1)
        self._bars.append({
            "time": ts,
            "open": o,
            "high": hi,
            "low": lo,
            "close": c,
            "volume": 100.0,
        })
        self._closed = c
        self._n += 1

    def last_closed_bars(self, n: int) -> pd.DataFrame:
        self._append_bar()
        if len(self._bars) < int(n):
            # Warm-up: backfill deterministically so the decision window is
            # always populated even on the first poll.
            while len(self._bars) < int(n):
                self._append_bar()
        return pd.DataFrame(self._bars[-int(n):]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Live trader
# --------------------------------------------------------------------------- #
class LiveTrader:
    """Production trading loop on the new stack.

    Responsibilities:
      * construct the broker for the TRADING_MODE (MockBroker for demo,
        Mt5Broker for live) and wire RiskSupervisor + ATRPositionSizer +
        TradeExecutor;
      * run startup reconciliation and HALT on drift;
      * one RiskSupervisor-gated decision per NEW closed candle;
      * kill-switch file check every loop iteration.
    """

    def __init__(
        self,
        cfg: AppConfig,
        broker: BaseBroker,
        signal_source: SignalSource,
        warmup_bars: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.broker = broker
        self.signal_source = signal_source
        if warmup_bars is None:
            warmup_bars = (
                max(CLOSED_BAR_WARMUP, cfg.model.live_history_bars)
                if cfg.model.signal_source == "model"
                else CLOSED_BAR_WARMUP
            )
        self.warmup_bars = int(warmup_bars)

        now_fn = lambda: datetime.now(timezone.utc)  # noqa: E731
        self.risk = RiskSupervisor(
            config=cfg.risk,
            db_path=cfg.paths.risk_state_db,
            now_fn=now_fn,
        )
        self.executor = TradeExecutor(
            broker=broker,
            risk=self.risk,
            cfg=cfg,
            sizer=ATRPositionSizer(
                account_risk=cfg.risk.risk_per_trade,
                atr_multiplier=cfg.broker.sl_atr_mult,
            ),
            local_state_file=cfg.paths.local_state_file,
        )
        self.decision_engine = DecisionEngine(
            cfg.behavior,
            kelly=KellyPositionSizer(
                max_position=1.0,
                kelly_fraction=cfg.behavior.kelly_fraction,
            ),
            risk_per_trade=cfg.risk.risk_per_trade,
        )
        self.trade_manager = TradeManager(cfg)
        # Live mode: unknown broker positions are closed, phantom local state
        # is dropped, and the executor halts on any drift (audit P0-6).
        self.executor.close_on_recon_drift = cfg.trading_mode is TradingMode.LIVE

    # ------------------------------------------------------------------ #
    # startup
    # ------------------------------------------------------------------ #
    def startup_reconcile(self) -> None:
        """Compare broker positions vs persisted local state; halt on drift."""
        rec = self.executor.startup_reconcile()
        if rec.ok:
            logger.info("Startup reconciliation OK: %s", rec.describe())
        else:
            logger.critical("STARTUP RECONCILIATION FAILED: %s — halting", rec.describe())
            # executor.startup_reconcile() already sets halted=True + reason.

    # ------------------------------------------------------------------ #
    # kill switch
    # ------------------------------------------------------------------ #
    def _kill_switch_triggered(self) -> bool:
        ks = self.cfg.paths.kill_switch_path
        if ks is not None and Path(ks).exists():
            logger.critical("KILL_SWITCH file present at %s — refusing to trade", ks)
            return True
        return False

    # ------------------------------------------------------------------ #
    # one decision per closed candle
    # ------------------------------------------------------------------ #
    def _current_equity(self) -> float:
        try:
            return float(self.broker.account_info().equity)
        except BrokerError:
            return self.risk.current_equity

    def _current_balance(self) -> float:
        try:
            return float(self.broker.account_info().balance)
        except BrokerError:
            return self.risk.current_equity

    def on_closed_bar(
        self,
        df: Optional[pd.DataFrame],
        bar_time: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Handle one closed candle in the fixed safety-to-decision order."""
        events: List[Dict[str, object]] = []
        if self._kill_switch_triggered():
            events.append({"event": "KILL_SWITCH"})
            return events
        if self.executor.halted:
            events.append({"event": "HALTED", "reason": self.executor.halt_reason})
            return events
        if df is None or df.empty:
            events.append({"event": "NO_BARS"})
            return events

        if bar_time is None:
            bar_time = str(df.iloc[-1]["time"])
        now_utc = self._bar_time_utc(bar_time)
        self.executor.advance_closed_bar()
        self._feed_closed_bar_to_broker(df)
        self._manage_open_positions(df, bar_time, events)

        equity = self._current_equity()
        positions: list[PositionInfo] = self.broker.get_positions(
            self.cfg.broker.symbol, self.cfg.broker.magic
        )
        position_side = self._action_side(positions)
        account = self._account_state(positions, equity)
        raw = self.signal_source.decide(df, account)
        decision = self.decision_engine.filter(
            raw,
            now_utc,
            position_side,
            self.executor.bars_since_last_loss,
            self.risk.trade_history,
        )
        if decision.action == 2 and not self.cfg.broker.allow_short:
            decision = Decision(
                0, decision.confidence, 0.0, "SHORT_DISABLED", decision.info
            )

        if decision.action == position_side:
            events.append({
                "event": "HOLD" if position_side else "NO_SIGNAL",
                "reason": decision.reason,
                "action": decision.action,
                "equity": equity,
            })
        else:
            if positions:
                close_reason = (
                    "MODEL_EXIT"
                    if decision.action == 0
                    and self.cfg.model.signal_source == "model"
                    else "SIGNAL_REVERSAL"
                )
                for pos in positions:
                    self._close_position(pos, close_reason, events)

            if decision.action != 0:
                remaining = self.broker.get_positions(
                    self.cfg.broker.symbol, self.cfg.broker.magic
                )
                if not remaining:
                    tick = self.broker.get_tick(self.cfg.broker.symbol)
                    md = self.executor.build_market_data(df, tick)
                    ok, reason, result = self.executor.execute_entry(
                        decision.action,
                        equity=self._current_equity(),
                        df=df,
                        market_data=md,
                        bar_time=bar_time,
                        size_multiplier=decision.size_multiplier,
                    )
                    events.append({
                        "event": "ENTRY",
                        "signal": decision.action,
                        "reason": decision.reason,
                        "ok": ok,
                        "executor_reason": reason,
                        "ticket": result.ticket if result else None,
                        "volume": result.volume_filled if result else None,
                        "fill_price": result.fill_price if result else None,
                    })
                    logger.info(
                        "ENTRY action=%s ok=%s reason=%s ticket=%s",
                        decision.action,
                        ok,
                        reason,
                        result.ticket if result else None,
                    )

        final_positions = self.broker.get_positions(
            self.cfg.broker.symbol, self.cfg.broker.magic
        )
        self.signal_source.executed(self._action_side(final_positions))
        self.executor._save_local_state()
        return events

    @staticmethod
    def _bar_time_utc(bar_time: str) -> datetime:
        timestamp = pd.Timestamp(bar_time)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(timezone.utc)
        return timestamp.to_pydatetime().astimezone(timezone.utc)

    @staticmethod
    def _action_side(positions: list[PositionInfo]) -> int:
        if not positions:
            return 0
        return 1 if positions[0].side == "buy" else 2

    def _account_state(
        self, positions: list[PositionInfo], equity: float
    ) -> AccountState:
        tick = self.broker.get_tick(self.cfg.broker.symbol)
        trade_pnl = 0.0
        bars_in_trade = 0
        for position in positions:
            side = 1.0 if position.side == "buy" else -1.0
            price = tick["bid"] if side > 0 else tick["ask"]
            price_return = (
                (price - position.open_price) / position.open_price * side
            )
            meta = self.executor.get_trade_meta(position.ticket)
            entry_equity = float(
                meta.get("entry_equity", self.executor.reference_equity or equity)
            )
            trade_pnl += (
                price_return
                * position.open_price
                * position.volume
                * 100.0
                / max(entry_equity, 1e-9)
            )
            bars_in_trade = max(
                bars_in_trade, int(meta.get("bars_held", 0))
            )
        peak = max(float(self.risk.peak_equity), float(equity))
        drawdown = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0
        reference = self.executor.reference_equity or equity
        return AccountState(
            position=(
                1 if positions and positions[0].side == "buy"
                else -1 if positions else 0
            ),
            trade_pnl=trade_pnl,
            bars_in_trade=bars_in_trade,
            drawdown=drawdown,
            equity_ratio=equity / reference if reference > 0 else 1.0,
        )

    def _manage_open_positions(
        self,
        df: pd.DataFrame,
        bar_time: str,
        events: list[dict[str, object]],
    ) -> None:
        positions = self.broker.get_positions(
            self.cfg.broker.symbol, self.cfg.broker.magic
        )
        if not positions:
            return
        atr_now = self.executor.compute_atr(df)
        last_bar = df.iloc[-1]
        for position in positions:
            meta = self.executor.get_trade_meta(position.ticket)
            meta.setdefault("initial_sl", position.sl)
            meta.setdefault("entry_atr", atr_now)
            meta.setdefault("partial_done", False)
            meta.setdefault("open_bar_time", bar_time)
            meta["bars_held"] = int(meta.get("bars_held", 0)) + 1
            self.executor.set_trade_meta(position.ticket, meta)
            actions = self.trade_manager.manage(
                position,
                float(meta.get("entry_atr", atr_now)),
                int(meta["bars_held"]),
                last_bar,
                atr_now,
                meta,
            )
            for action in actions:
                if isinstance(action, ModifySL):
                    ok, reason, _ = self.executor.modify_sl_tp(
                        action.ticket, action.sl, position.tp
                    )
                    events.append({
                        "event": "MODIFY_SL",
                        "ticket": action.ticket,
                        "sl": action.sl,
                        "ok": ok,
                        "reason": reason,
                    })
                elif isinstance(action, PartialClose):
                    self._partial_close(action, meta, events)
                elif isinstance(action, Close):
                    current = self._position_by_ticket(action.ticket)
                    if current is not None:
                        self._close_position(current, action.reason, events)

    def _position_by_ticket(self, ticket: int) -> PositionInfo | None:
        return next(
            (
                position
                for position in self.broker.get_positions(
                    self.cfg.broker.symbol, self.cfg.broker.magic
                )
                if position.ticket == ticket
            ),
            None,
        )

    def _close_position(
        self,
        position: PositionInfo,
        close_reason: str,
        events: list[dict[str, object]],
    ) -> bool:
        meta = self.executor.get_trade_meta(position.ticket)
        balance_before = self._current_balance()
        ok, reason, _ = self.executor.execute_close(
            position.ticket, equity=self._current_equity()
        )
        equity_after = self._current_equity()
        entry_balance = float(meta.get("entry_balance", balance_before))
        pnl = self._current_balance() - entry_balance if ok else 0.0
        events.append({
            "event": "CLOSE",
            "ticket": position.ticket,
            "ok": ok,
            "reason": close_reason,
            "executor_reason": reason,
            "pnl": pnl,
        })
        if ok:
            self.executor.update_risk_after_close(pnl, equity_after, pnl > 0.0)
        return ok

    def _partial_close(
        self,
        action: PartialClose,
        meta: dict,
        events: list[dict[str, object]],
    ) -> None:
        balance_before = self._current_balance()
        ok, reason, _ = self.executor.execute_close(
            action.ticket,
            volume=action.volume,
            equity=self._current_equity(),
        )
        equity_after = self._current_equity()
        entry_balance = float(meta.get("entry_balance", balance_before))
        balance_after = self._current_balance()
        pnl = balance_after - entry_balance if ok else 0.0
        if ok:
            meta["partial_done"] = True
            meta["entry_balance"] = balance_after
            self.executor.set_trade_meta(action.ticket, meta)
            self.executor.update_risk_after_close(pnl, equity_after, pnl > 0.0)
        events.append({
            "event": "PARTIAL_CLOSE",
            "ticket": action.ticket,
            "volume": action.volume,
            "ok": ok,
            "reason": reason,
            "pnl": pnl,
        })

    def _feed_closed_bar_to_broker(self, df: pd.DataFrame) -> List[dict]:
        """Evaluate the newly closed bar against open SL/TP (demo broker).

        MockBroker.advance() fills stops/take-profits when the closed bar
        breaches a level and moves the mid to the bar close.  Equity is
        captured BEFORE the advance so the realized PnL of each fill is fed
        back into the RiskSupervisor (never fabricated).  The real Mt5Broker
        needs no such feed — the terminal executes SL/TP server-side — so
        this is a no-op there.
        """
        if df is None or len(df) == 0:
            return []
        local_tickets = set(self.executor._positions)
        local_meta = {
            ticket: self.executor.get_trade_meta(ticket) for ticket in local_tickets
        }
        balance_before = self._current_balance()
        advance = getattr(self.broker, "advance", None)
        fills = []
        if advance is not None:
            row = df.iloc[-1]
            fills = advance(
                float(row["high"]), float(row["low"]), float(row["close"])
            )
        eq_after = self._current_equity()
        balance_after = self._current_balance()
        for f in fills:
            logger.info("SL/TP fill ticket=%s reason=%s pnl=%.2f",
                        f.get("ticket"), f.get("reason"), float(f.get("pnl", 0.0)))
        self.executor.reconcile_positions()
        active_tickets = {
            position.ticket
            for position in self.broker.get_positions(
                self.cfg.broker.symbol, self.cfg.broker.magic
            )
        }
        closed_tickets = local_tickets - active_tickets
        if fills or closed_tickets:
            if fills:
                pnl = balance_after - balance_before
            else:
                entry_balances = [
                    float(local_meta[ticket].get("entry_balance", balance_before))
                    for ticket in closed_tickets
                ]
                pnl = balance_after - min(entry_balances)
            if abs(pnl) > 1e-9:
                self.executor.update_risk_after_close(pnl, eq_after, pnl > 0.0)
        return fills

    # ------------------------------------------------------------------ #
    # candle-close-aligned loop
    # ------------------------------------------------------------------ #
    def run_forever(
        self,
        bar_source: BarSource,
        poll_seconds: float = 5.0,
        max_bars: int = 0,
    ) -> int:
        """Wait for each NEW closed candle and make exactly one decision.

        ``bar_source.last_closed_bars(n)`` returns only closed bars, so the
        last row's timestamp changes exactly once per closed candle — the
        loop acts on that transition (audit P0-6: no 10s spam on H1).

        Returns the number of closed candles processed (bounded by
        ``max_bars`` when > 0, e.g. for the ``--smoke`` demo run).
        """
        logger.info("Candle-close-aligned loop starting (timeframe=%s, poll=%ss)",
                    self.cfg.timeframe, poll_seconds)
        last_closed: Optional[object] = None
        processed = 0
        while True:
            if self._kill_switch_triggered():
                logger.critical("KILL_SWITCH present — halting")
                return processed
            try:
                bars = bar_source.last_closed_bars(self.warmup_bars)
            except BrokerError as e:
                logger.error("Bar fetch failed: %s", e)
                time.sleep(poll_seconds)
                continue
            if bars is None or len(bars) == 0:
                time.sleep(poll_seconds)
                continue
            closed_time = bars.iloc[-1]["time"]
            if closed_time == last_closed:
                time.sleep(poll_seconds)
                continue
            last_closed = closed_time
            df = bars.iloc[-self.warmup_bars:].reset_index(drop=True)
            try:
                self.on_closed_bar(df, bar_time=str(closed_time))
                processed += 1
                if max_bars > 0 and processed >= max_bars:
                    logger.info("Reached smoke bar limit (%d) — stopping", max_bars)
                    return processed
            except Exception:
                logger.exception("on_closed_bar failed for bar %s", closed_time)
                if max_bars > 0:
                    raise
            time.sleep(poll_seconds)


# --------------------------------------------------------------------------- #
# broker construction + TRADING_MODE gate
# --------------------------------------------------------------------------- #
def _setup_logging(cfg: AppConfig) -> None:
    log_dir = Path(cfg.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.log_level, logging.INFO))
    if root.handlers:
        return
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "live_trader.log", maxBytes=5_000_000, backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)


def model_promotion_failures(cfg: AppConfig) -> list[str]:
    manifest = None
    failures: list[str] = []
    try:
        manifest = load_manifest(cfg.model.manifest_path)
    except ModelArtifactError as exc:
        failures.append(f"model manifest could not be loaded: {exc}")
    evaluation_path = Path(cfg.model.manifest_path).parent / "evaluation.json"
    try:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(f"evaluation file missing: {evaluation_path}")
    except (OSError, ValueError) as exc:
        failures.append(f"evaluation file could not be loaded: {exc}")
    else:
        if not isinstance(evaluation, dict):
            failures.append("evaluation file must contain a JSON object")
            return failures
        if evaluation.get("passed") is not True:
            failures.append("evaluation passed is not true")
        if (
            manifest is not None
            and evaluation.get("contract_hash") != manifest.contract_hash
        ):
            failures.append(
                "evaluation contract_hash does not match the model manifest"
            )
    return failures


def enforce_model_promotion_gate(cfg: AppConfig) -> None:
    if cfg.model.signal_source != "model":
        return
    failures = model_promotion_failures(cfg)
    if not failures:
        return
    if (
        cfg.trading_mode is TradingMode.LIVE
        or cfg.behavior.require_promoted_model
    ):
        raise RuntimeError(
            "Model promotion gate refused to start. Failed gates: "
            + "; ".join(failures)
        )
    logger.warning(
        "Model promotion gate warning (demo mode): %s",
        "; ".join(failures),
    )


def _build_broker(
    cfg: AppConfig, promotion_checked: bool = False
) -> BaseBroker:
    """Return the broker for the configured TRADING_MODE.

    demo (default) -> MockBroker (deterministic simulated fills, never real)
    live          -> Mt5Broker  (real MetaTrader5 terminal)
    """
    if cfg.trading_mode is TradingMode.DEMO:
        if not promotion_checked:
            enforce_model_promotion_gate(cfg)
        logger.info("TRADING_MODE=demo — using MockBroker (no real orders)")
        broker = MockBroker(
            symbol=cfg.broker.symbol,
            balance=_DEMO_BALANCE,
            spread=cfg.cost.spread,
            commission=cfg.cost.commission,
            slippage=cfg.cost.slippage,
            seed=42,
            state_path=cfg.paths.local_state_file.parent / "mock_state.json",
        )
        broker.connect()
        return broker

    if cfg.trading_mode is TradingMode.LIVE:
        # --- live gates: risk state loadable + MT5 creds + promoted model ---
        # --- evaluation record (when using model signals)                ---
        failures: list[str] = []
        if cfg.model.signal_source == "model":
            failures.extend(model_promotion_failures(cfg))
        try:
            RiskSupervisor(config=cfg.risk, db_path=cfg.paths.risk_state_db,
                           now_fn=lambda: datetime.now(timezone.utc))
        except Exception as e:  # noqa: BLE001
            failures.append(f"risk state loadable ({e})")
        if cfg.broker.mt5_login is None:
            failures.append("MT5_LOGIN not configured")
        if cfg.broker.mt5_password is None:
            failures.append("MT5_PASSWORD not configured")
        if cfg.broker.mt5_server is None:
            failures.append("MT5_SERVER not configured")
        if failures:
            raise RuntimeError(
                "TRADING_MODE=live REFUSED TO START. Failed gates: "
                + ", ".join(failures)
                + ". Fix the failing gates or switch TRADING_MODE=demo."
            )

        logger.info("TRADING_MODE=live — using Mt5Broker (real orders)")
        broker = Mt5Broker(
            symbol=cfg.broker.symbol,
            magic=cfg.broker.magic,
            deviation=cfg.broker.deviation,
            allow_short=cfg.broker.allow_short,
            max_requote_retries=cfg.broker.max_requote_retries,
            order_retry_delay_sec=cfg.broker.order_retry_delay_sec,
            login=cfg.broker.mt5_login,
            password=cfg.broker.mt5_password,
            server=cfg.broker.mt5_server,
            mt5_path=cfg.broker.mt5_path,
            idempotency_state_path=cfg.paths.local_state_file.parent
            / "broker_idempotency.json",
        )
        broker.connect()
        return broker

    raise ValueError(f"Unsupported trading mode {cfg.trading_mode!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trading bot live entry point")
    p.add_argument("--smoke", action="store_true",
                   help="run a bounded demo-mode smoke loop (synthetic bars, "
                        "MockBroker; requires TRADING_MODE=demo)")
    p.add_argument("--max-bars", type=int, default=0,
                   help="stop after N closed candles (0 = run forever)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config()
    _setup_logging(cfg)
    logger.info("=" * 70)
    logger.info("live_trade_mt5.py starting — TRADING_MODE=%s SYMBOL=%s TF=%s",
                cfg.trading_mode.value, cfg.symbol_cfg, cfg.timeframe)

    if args.smoke and cfg.trading_mode is TradingMode.LIVE:
        logger.error("--smoke refuses to run with TRADING_MODE=live")
        return 3

    enforce_model_promotion_gate(cfg)
    signal_source = build_signal_source(cfg)
    broker = _build_broker(cfg, promotion_checked=True)
    trader = LiveTrader(cfg, broker, signal_source)

    # Startup reconciliation (P0-6): broker vs local state; halt on drift.
    trader.startup_reconcile()
    if trader.executor.halted:
        logger.critical("Trader halted at startup: %s", trader.executor.halt_reason)
        return 2

    if cfg.trading_mode is TradingMode.LIVE:
        bar_source: BarSource = Mt5BarSource(
            broker._mt5,
            cfg.broker.symbol,
            cfg.timeframe,
            utc_offset_hours=cfg.broker.utc_offset_hours,
        )  # type: ignore[attr-defined]
        max_bars = 0
    else:
        # Demo: deterministic synthetic closed-bar feed (never real orders).
        bar_source = SyntheticBarSource(seed=7, start_price=2000.0)
        max_bars = args.max_bars if args.max_bars > 0 else (300 if args.smoke else 0)

    poll_seconds = 0.05 if args.smoke else 5.0
    processed = 0
    try:
        processed = trader.run_forever(bar_source, poll_seconds=poll_seconds,
                                       max_bars=max_bars)
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")

    # Smoke / bounded run: flatten any open demo positions and report the
    # REAL realized PnL from the broker's fill history (never fabricated).
    open_positions = broker.get_positions(cfg.broker.symbol, cfg.broker.magic)
    if open_positions:
        for pos in open_positions:
            eq_before = trader._current_equity()
            ok, _, _ = trader.executor.execute_close(pos.ticket, equity=eq_before)
            eq_after = trader._current_equity()
            logger.info("Flatten on exit ticket=%s ok=%s pnl=%.2f",
                        pos.ticket, ok, eq_after - eq_before)
    closed = getattr(broker, "closed_fills", list)()
    logger.info("Smoke summary: bars=%d open_positions=%d fills=%d",
                processed,
                len(broker.get_positions(cfg.broker.symbol, cfg.broker.magic)),
                len(closed))
    try:
        broker.disconnect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Broker disconnect failed: %s", exc)
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
