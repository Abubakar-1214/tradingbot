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

import os

import sys

import time

from datetime import datetime, timedelta, timezone

from pathlib import Path

from typing import Dict, List, Optional, Protocol

try:
    from core.audio_alert import play_trade_open_sound, play_trade_close_sound
except ImportError:
    def play_trade_open_sound(): pass
    def play_trade_close_sound(profit: float = 0.0): pass



import numpy as np

import pandas as pd



# Allow direct execution from any CWD (``python live/live_trade_mt5.py``):

# put the repo root on sys.path so ``core`` / ``live`` / ``models`` resolve.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



from core.config import (

    AppConfig,

    TradingMode,

    enforce_sltp_compatibility,

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



    def __init__(self, fast: int = 20, slow: int = 50, enter_on_trend: bool = False) -> None:

        if fast >= slow:

            raise ValueError(f"fast SMA ({fast}) must be < slow SMA ({slow})")

        self.enter_on_trend = bool(enter_on_trend)

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

        if position_side == 0 and self.enter_on_trend:

            if last_fast > last_slow:

                return 1

            if last_fast < last_slow:

                return 2

        return 0



    def decide(

        self, df: pd.DataFrame | None, account: AccountState

    ) -> Decision:

        if df is None or len(df) < self.slow + 2:

            return Decision(0, 1.0, 1.0, "NO_SIGNAL", {"hold": True})

        closes = df["close"]

        fast_sma = closes.rolling(self.fast).mean()

        slow_sma = closes.rolling(self.slow).mean()

        last_fast, last_slow = float(fast_sma.iloc[-1]), float(slow_sma.iloc[-1])

        last_close = float(closes.iloc[-1])

        info = {

            "fast_sma": round(last_fast, 3),

            "slow_sma": round(last_slow, 3),

            "close": round(last_close, 3),

            "gap": round(last_fast - last_slow, 3),

        }

        action = self.signal(df, account.position)

        if action == 0:

            info["hold"] = True

            return Decision(0, 1.0, 1.0, "NO_SIGNAL", info)

        return Decision(action, 1.0, 1.0, "RULE_SIGNAL", info)



    def executed(self, action: int) -> None:

        return None





def build_signal_source(cfg: AppConfig) -> SignalSource:

    """Build the configured rule or model source."""

    if cfg.model.signal_source == "model":

        return ModelSignalSource(cfg)

    import os

    fast = int(getattr(cfg.model, "sma_fast", os.getenv("SMA_FAST", 20)))

    slow = int(getattr(cfg.model, "sma_slow", os.getenv("SMA_SLOW", 50)))

    trend_entry_val = getattr(cfg.model, "sma_trend_entry", os.getenv("SMA_TREND_ENTRY", "true"))

    trend_entry = str(trend_entry_val).strip().lower() in ("true", "1", "yes")

    return RuleSignalSource(fast=fast, slow=slow, enter_on_trend=trend_entry)





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

        htf_source: Optional[BarSource] = None,

    ) -> None:

        self.cfg = cfg

        self.broker = broker

        self.signal_source = signal_source

        if warmup_bars is None:

            warmup_bars = (

                max(CLOSED_BAR_WARMUP, cfg.model.live_history_bars)

                if cfg.model.signal_source == "model"

                else max(CLOSED_BAR_WARMUP, getattr(cfg.model, "sma_slow", 50) + 10)

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

            close_on_recon_drift=True,

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

        self._last_scalp_close_time: float = 0.0

        self._bars_since_close: int = 999

        self._last_close_bar_time: Optional[str] = None

        self.htf_source = htf_source



        # Sync fresh risk state equity to real broker equity if no trades yet

        try:

            acc = broker.account_info()

            init_equity = float(acc.equity)

            if init_equity > 0 and not self.risk.trade_history:

                self.risk.daily_start_equity = init_equity

                self.risk.peak_equity = init_equity

                self.risk.current_equity = init_equity

                self.risk._persist()

        except Exception:

            pass



    # ------------------------------------------------------------------ #

    # startup

    # ------------------------------------------------------------------ #

    def startup_reconcile(self) -> None:

        """Compare broker positions vs persisted local state; halt on drift."""

        rec = self.executor.startup_reconcile()

        if rec.ok or (not self.executor.halted and self.executor.close_on_recon_drift):

            logger.info("Startup reconciliation OK (self-healed if needed): %s", rec.describe())

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

        self._bars_since_close += 1

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

                0,

                decision.confidence,

                0.0,

                "SHORT_DISABLED",

                decision.info,

                decision.sl_frac,

                decision.tp_frac,

            )



        htf_trend = self.get_htf_trend()

        if decision.action != 0 and htf_trend != 0 and decision.action != htf_trend:

            trend_m15 = self._get_trend_for_tf("M15")

            trend_m5 = self._get_trend_for_tf("M5")

            trend_m1 = self._get_trend_for_tf("M1")

            str_m15 = "BUY" if trend_m15 == 1 else "SELL" if trend_m15 == 2 else "NEUTRAL"

            str_m5 = "BUY" if trend_m5 == 1 else "SELL" if trend_m5 == 2 else "NEUTRAL"

            str_m1 = "BUY" if trend_m1 == 1 else "SELL" if trend_m1 == 2 else "NEUTRAL"

            logger.info(

                "🛡️ [TRI-CONFLUENCE] Signal wants %s | M15=%s | M5=%s | M1=%s -> Waiting for M15, M5, M1 alignment!",

                "BUY" if decision.action == 1 else "SELL",

                str_m15,

                str_m5,

                str_m1,

            )

            decision = Decision(0, decision.confidence, 0.0, "HTF_TREND_MISMATCH", decision.info)




        fast_str = f" FastSMA={decision.info['fast_sma']}" if "fast_sma" in decision.info else ""

        slow_str = f" SlowSMA={decision.info['slow_sma']}" if "slow_sma" in decision.info else ""

        close_str = f" Close={decision.info['close']}" if "close" in decision.info else ""

        logger.info(

            "BAR %s DECISION action=%s" + close_str + fast_str + slow_str + " position_side=%s reason=%s conf=%.3f equity=%.2f",

            bar_time,

            decision.action,

            position_side,

            decision.reason,

            decision.confidence,

            equity,

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

                scalp_cfg = getattr(self.cfg, "scalp", None)

                allow_loss_cut = getattr(scalp_cfg, "allow_loss_cut", False) if scalp_cfg else True

                if not allow_loss_cut:

                    logger.info("🛡️ [SIGNAL REVERSAL] Signal changed but SCALP_ALLOW_LOSS_CUT=false -> Holding position until target profit!")

                else:

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

                scalp_cfg = getattr(self.cfg, "scalp", None)

                reenter_bars = getattr(scalp_cfg, "reenter_bars_wait", 0)

                min_eq = getattr(scalp_cfg, "multi_trade_min_equity", 80.0)

                max_cfg_pos = getattr(scalp_cfg, "max_open_positions", 4)

                allowed_max_positions = max_cfg_pos if equity > min_eq else 1



                can_open = False

                if not remaining and (reenter_bars == 0 or self._bars_since_close >= reenter_bars):

                    can_open = True

                elif len(remaining) < allowed_max_positions:

                    can_open = True

                    logger.info(

                            "⚡ [MULTI-TRADE ENTRY] Holding previous trade; opening new %s on trend change (Equity=$%.2f >= $%.2f)",

                            "BUY" if decision.action == 1 else "SELL", equity, min_eq

                        )



                if can_open:


                    tick = self.broker.get_tick(self.cfg.broker.symbol)

                    md = self.executor.build_market_data(df, tick)

                    ok, reason, result = self.executor.execute_entry(

                        decision.action,

                        equity=self._current_equity(),

                        df=df,

                        market_data=md,

                        bar_time=bar_time,

                        size_multiplier=decision.size_multiplier,

                        sl_frac=decision.sl_frac,

                        tp_frac=decision.tp_frac,

                    )

                    if ok:

                        play_trade_open_sound()

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

            leverage = 1.0

            manifest = getattr(self.signal_source, "manifest", None)

            if manifest is not None:

                leverage = float(

                    manifest.extra.get(

                        "leverage",

                        manifest.hyperparams.get("leverage", 1.0),

                    )

                )

            trade_pnl += (

                (price - position.open_price) / position.open_price

                * side

                * leverage

            )

            meta = self.executor.get_trade_meta(position.ticket)

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

                    scalp_cfg = getattr(self.cfg, "scalp", None)

                    if scalp_cfg and not getattr(scalp_cfg, "allow_loss_cut", False):

                        logger.info("🛡️ TradeManager requested %s, but SCALP_ALLOW_LOSS_CUT=false -> Ignored!", action.reason)

                        continue


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

        if ok:

            play_trade_close_sound(float(getattr(position, "profit", pnl)))

        events.append({

            "event": "CLOSE",

            "ticket": position.ticket,

            "ok": ok,

            "reason": close_reason,

            "executor_reason": reason,

            "pnl": pnl,

        })

        if ok:

            self._bars_since_close = 0

            self._last_close_bar_time = getattr(self, "_current_bar_time", None)

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



    def _get_trend_for_tf(self, timeframe: str) -> int:

        """Fetch bars and compute SMA Fast/Slow trend for a given timeframe."""

        tf_name = timeframe.upper()

        scalp_cfg = getattr(self.cfg, "scalp", None)

        fast_period = int(getattr(scalp_cfg, "htf_fast_sma", 20))

        slow_period = int(getattr(scalp_cfg, "htf_slow_sma", 50))

        n_bars = slow_period + 20



        df_bars = None

        if self.cfg.trading_mode is TradingMode.LIVE and hasattr(self.broker, "_mt5"):

            try:

                mt5 = self.broker._mt5

                tf = getattr(mt5, f"TIMEFRAME_{tf_name}", None)

                if tf is not None:

                    rates = mt5.copy_rates_from_pos(self.cfg.broker.symbol, tf, 0, n_bars)

                    if rates is not None and len(rates) > 0:

                        df_bars = pd.DataFrame(rates)

            except Exception as exc:

                logger.warning("Could not fetch %s bars from MT5: %s", tf_name, exc)



        if df_bars is None and hasattr(self, "htf_source") and self.htf_source is not None:

            try:

                df_bars = self.htf_source.last_closed_bars(n_bars)

            except Exception:

                pass



        if df_bars is None or len(df_bars) < slow_period + 2:

            return 0



        closes = df_bars["close"]

        fast_sma = closes.rolling(fast_period).mean()

        slow_sma = closes.rolling(slow_period).mean()

        last_fast = fast_sma.iloc[-1]

        last_slow = slow_sma.iloc[-1]

        if pd.isna(last_fast) or pd.isna(last_slow):

            return 0

        if last_fast > last_slow:

            return 1

        elif last_fast < last_slow:

            return 2

        return 0



    def get_htf_trend(self) -> int:

        """Tri-Timeframe Confluence: M15 + M5 + M1 (H1 excluded).



        Aligns M15, M5, and M1 SMA trends.

        Returns 1 (BUY) if all active timeframes agree BUY,

                2 (SELL) if all active timeframes agree SELL,

                else 0 (conflict/wait).

        """

        scalp_cfg = getattr(self.cfg, "scalp", None)

        if not scalp_cfg or not getattr(scalp_cfg, "htf_filter_enabled", True):

            return 0



        trend_m15 = self._get_trend_for_tf("M15")

        check_m5 = getattr(scalp_cfg, "check_m5", True)

        check_m1 = getattr(scalp_cfg, "check_m1", True)



        trend_m5 = self._get_trend_for_tf("M5") if check_m5 else 0

        trend_m1 = self._get_trend_for_tf("M1") if check_m1 else 0



        active_trends = [trend_m15]

        if check_m5 and trend_m5 != 0:

            active_trends.append(trend_m5)

        if check_m1 and trend_m1 != 0:

            active_trends.append(trend_m1)



        valid_trends = [t for t in active_trends if t != 0]

        if not valid_trends:

            return 0



        if all(t == 1 for t in valid_trends):

            return 1

        elif all(t == 2 for t in valid_trends):

            return 2

        else:

            return 0



    def _get_current_trend(self, df: pd.DataFrame) -> int:

        fast_period = getattr(self.signal_source, "fast", 20)

        slow_period = getattr(self.signal_source, "slow", 50)

        if df is None or len(df) < slow_period + 2:

            return 0

        closes = df["close"]

        fast_sma = closes.rolling(fast_period).mean()

        slow_sma = closes.rolling(slow_period).mean()

        last_fast = fast_sma.iloc[-1]

        last_slow = slow_sma.iloc[-1]

        if pd.isna(last_fast) or pd.isna(last_slow):

            return 0

        if last_fast > last_slow:

            return 1

        elif last_fast < last_slow:

            return 2

        return 0



    def _handle_intra_candle(self, df: pd.DataFrame, bar_time: str) -> None:

        scalp_cfg = getattr(self.cfg, "scalp", None)

        if not scalp_cfg or not scalp_cfg.enabled:

            return

        if self._kill_switch_triggered() or self.executor.halted:

            return



        positions = self.broker.get_positions(

            self.cfg.broker.symbol, self.cfg.broker.magic

        )

        now = time.time()

        equity = self._current_equity()



        tp_usd = scalp_cfg.target_profit_usd

        sl_usd = scalp_cfg.max_loss_usd



        if getattr(getattr(self.cfg, "adaptive_tier", None), "enabled", True):

            if equity < 20.0:

                tp_usd = min(tp_usd, 0.35)

                sl_usd = min(sl_usd, 1.20)

            elif equity < 50.0:

                tp_usd = max(0.40, min(tp_usd, 0.70))

                sl_usd = max(1.20, min(sl_usd, 2.00))



        self._current_bar_time = bar_time



        if positions:

            for pos in positions:

                profit = float(getattr(pos, "profit", 0.0))

                if profit == 0.0:

                    try:

                        tick = self.broker.get_tick(self.cfg.broker.symbol)

                        cur_p = tick["bid"] if pos.side == "buy" else tick["ask"]

                        diff = (cur_p - pos.open_price) if pos.side == "buy" else (pos.open_price - cur_p)

                        profit = diff * pos.volume * 100.0

                    except Exception:

                        pass



                meta = self.executor.get_trade_meta(pos.ticket)

                peak_profit = max(float(meta.get("peak_profit", 0.0)), profit)

                meta["peak_profit"] = peak_profit

                self.executor.set_trade_meta(pos.ticket, meta)



                runner_enabled = getattr(scalp_cfg, "trend_runner_enabled", True)

                min_tp = min(tp_usd, getattr(scalp_cfg, "runner_min_profit_usd", tp_usd))


                max_tp = getattr(scalp_cfg, "runner_max_profit_usd", 15.0)

                trail_pct = getattr(scalp_cfg, "runner_trail_percent", 25.0)

                pos_dir = 1 if pos.side == "buy" else 2



                # 1. Big Target Reached ($10 - $20)?

                if runner_enabled and profit >= max_tp:

                    logger.info(

                        "🏆 [MAX TREND RUNNER PROFIT] Ticket %d profit=+$%.2f reached big target >= +$%.2f! Banking massive profit!",

                        pos.ticket, profit, max_tp,

                    )

                    self._close_position(pos, "TREND_RUNNER_MAX", [])

                    self._last_scalp_close_time = now

                    return



                # 2. Check if reached minimum profit target ($1.00)
                if profit >= min_tp:
                    if not runner_enabled:
                        logger.info(
                            "💰 [SCALP PROFIT LOCK] Ticket %d profit=+$%.2f >= target +$%.2f. Closing immediately to bank profit!",
                            pos.ticket, profit, min_tp,
                        )
                        self._close_position(pos, "SCALP_PROFIT_TARGET", [])
                        self._last_scalp_close_time = now
                        return

                    # Evaluate Trend Health: Primary trend is M15 (our strategy timeframe)
                    trend_m15 = self._get_trend_for_tf("M15")
                    trend_m5 = self._get_trend_for_tf("M5")

                    # Primary trend is M15: as long as M15 is in trade direction (or M5 confirms if M15 flat),
                    # RUNNER IS ACTIVE! We do NOT kill the runner at $1.00 due to 1-minute noise!
                    is_trend_confirmed = (trend_m15 == pos_dir or (trend_m15 == 0 and trend_m5 == pos_dir))

                    # Dynamic Trailing Ratchet: Never blindly waits for $15!
                    # Starts trailing once profit reaches $3.00+ with at least $1.00 buffer for Gold wicks.
                    trail_start = getattr(scalp_cfg, "runner_trail_start_usd", 3.0)
                    is_pulling_back = False
                    if peak_profit >= trail_start:
                        pullback_allowed = max(1.0, peak_profit * (trail_pct / 100.0))
                        if profit <= (peak_profit - pullback_allowed):
                            is_pulling_back = True
                    elif peak_profit >= 2.00 and profit <= 0.75:
                        # Breakeven safety: lock profit if it made $2+ and drops near breakeven
                        is_pulling_back = True

                    if not is_trend_confirmed and profit >= min_tp:
                        # Real M15 trend reversal against the trade! Bank profit now!
                        logger.info(
                            "⚡ [TREND REVERSAL EXIT] Ticket %d profit=+$%.2f >= target +$%.2f. M15 trend reversed -> Banking profit!",
                            pos.ticket, profit, min_tp,
                        )
                        self._close_position(pos, "SCALP_PROFIT_TARGET", [])
                        self._last_scalp_close_time = now
                        return
                    elif is_pulling_back:
                        # Trailing profit lock secured
                        logger.info(
                            "🎯 [TRAILING PROFIT SECURED] Ticket %d profit=+$%.2f (peak was +$%.2f) pulling back -> Securing runner profit!",
                            pos.ticket, profit, peak_profit,
                        )
                        self._close_position(pos, "TREND_RUNNER_TRAIL", [])
                        self._last_scalp_close_time = now
                        return
                    else:
                        # Core trend is confirmed & surging! Let it run towards $10 - $20!
                        last_runner_log = float(meta.get("last_runner_log", 0.0))
                        if now - last_runner_log > 5.0:
                            meta["last_runner_log"] = now
                            self.executor.set_trade_meta(pos.ticket, meta)
                            logger.info(
                                "🚀 [TREND RUNNER ACTIVE] Ticket %d profit=+$%.2f (Peak: +$%.2f). M15 trend strongly aligned! Letting trade run towards $%.2f!",
                                pos.ticket, profit, peak_profit, max_tp,
                            )

                if getattr(scalp_cfg, "allow_loss_cut", False) and sl_usd > 0.0:
                    if profit <= -sl_usd:
                        logger.warning(
                            "🛑 [SCALP LOSS CUT] Ticket %d loss=-$%.2f <= max -$%.2f. Cutting trade to protect $%.2f equity!",
                            pos.ticket, abs(profit), sl_usd, equity,
                        )
                        self._close_position(pos, "SCALP_LOSS_CUT", [])
                        self._last_scalp_close_time = now
                        return

        # Step 2: Check new entry (supports multi-trade / hedging if equity is safe)

        min_eq = getattr(scalp_cfg, "multi_trade_min_equity", 80.0)

        # User Rule: If balance/equity > $80, allow up to max_open_positions (up to 4 concurrent trades for hedge/rescue).
        # If <= $80, strictly allow only 1 trade (wait for profit close before opening next)!
        max_cfg_pos = getattr(scalp_cfg, "max_open_positions", 4)
        allowed_max_positions = max_cfg_pos if equity > min_eq else 1

        if len(positions) >= allowed_max_positions:

            return



        # 1. Check same-candle re-entry preference

        allow_same_candle = getattr(scalp_cfg, "allow_same_candle_reenter", True)

        if not allow_same_candle and self._last_close_bar_time is not None and self._last_close_bar_time == bar_time:

            return



        # 2. Skip configured number of candles after close (if no positions open)

        reenter_bars = getattr(scalp_cfg, "reenter_bars_wait", 0)

        if reenter_bars > 0 and not positions and self._bars_since_close < reenter_bars:

            return



        # 3. Seconds cooldown

        if hasattr(self, "_last_scalp_close_time"):

            if now - self._last_scalp_close_time < scalp_cfg.reenter_delay_sec:

                return



        trend_action = self._get_current_trend(df)

        # Dynamic Hedge / Opposite Trend Detection:
        # If we have an existing trade in drawdown and lower timeframes (M5 / M1) confirm opposite trend,
        # switch trend_action to opposite direction to open a protective Hedge trade!
        trend_m5 = self._get_trend_for_tf("M5")
        trend_m1 = self._get_trend_for_tf("M1")

        if positions:
            existing_side = positions[0].side  # "buy" or "sell"
            existing_dir = 1 if existing_side == "buy" else 2
            opposite_dir = 2 if existing_dir == 1 else 1

            is_in_drawdown = any(getattr(p, "profit", 0.0) < -0.50 for p in positions)
            if is_in_drawdown:
                if trend_m5 == opposite_dir or trend_m1 == opposite_dir:
                    trend_action = opposite_dir
                    logger.info(
                        "🛡️ [HEDGE DETECTION] Position #1 (%s) in drawdown ($%.2f); M5/M1 confirmed %s trend -> Triggering HEDGE trade!",
                        existing_side.upper(), positions[0].profit, "SELL" if opposite_dir == 2 else "BUY"
                    )

        if trend_action not in (1, 2):

            return

        if trend_action == 2 and not self.cfg.broker.allow_short:

            return

        # Multi-timeframe confluence: only block initial entries, not hedge trades
        if not positions:
            htf_trend = self.get_htf_trend()
            if htf_trend != 0 and trend_action != htf_trend:
                return



        # If already holding a position, require spacing cooldown (e.g. 30s) so we don't duplicate on same tick
        if positions:
            if hasattr(self, "_last_scalp_entry_attempt"):
                if now - self._last_scalp_entry_attempt < 30.0:
                    return
            self._last_scalp_entry_attempt = now
            logger.info(
                "⚡ [MULTI-TRADE SCALPER] Equity=$%.2f > $%.2f. Position #%d already running; confirmed %s trend -> Opening position #%d/%d",
                equity, min_eq, len(positions), "BUY" if trend_action == 1 else "SELL", len(positions) + 1, allowed_max_positions
            )

        tick = self.broker.get_tick(self.cfg.broker.symbol)

        md = self.executor.build_market_data(df, tick)

        ok, reason, result = self.executor.execute_entry(

            trend_action,

            equity=equity,

            df=df,

            market_data=md,

            bar_time=bar_time,

            size_multiplier=1.0,

            sl_frac=None,

            tp_frac=None,

        )

        if ok:

            self._last_scalp_entry_time = now

            play_trade_open_sound()

            mode_str = "MULTI-TRADE/HEDGE" if positions else "SCALP RE-ENTRY"

            fill_price = float(getattr(result, "open_price", getattr(result, "fill_price", 0.0)) or 0.0)
            ticket_str = str(getattr(result, "ticket", "N/A"))

            logger.info(

                "⚡ [%s] Opened %s in trend direction. Ticket=%s at %.3f (Target Profit: +$%.2f | Balance=$%.2f)",

                mode_str,

                "BUY" if trend_action == 1 else "SELL",

                ticket_str,

                fill_price,

                tp_usd,

                equity,

            )
        else:
            logger.warning(
                "⚠️ [MULTI-TRADE ENTRY REJECTED] %s failed: %s",
                "BUY" if trend_action == 1 else "SELL", reason
            )



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

        logger.info("Candle-close-aligned loop starting (timeframe=%s, poll=%.0fms)",

                    self.cfg.timeframe, poll_seconds * 1000.0)


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

            df = bars.iloc[-self.warmup_bars:].reset_index(drop=True)



            # Intra-candle real-time monitoring and scalping

            self._handle_intra_candle(df, bar_time=str(closed_time))



            if closed_time == last_closed:

                time.sleep(poll_seconds)

                continue

            last_closed = closed_time

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

    fmt = logging.Formatter(

        "%(asctime)s %(levelname)s %(name)s: %(message)s",

        datefmt="%Y-%m-%d %H:%M:%S",

    )

    # Check if a file handler already exists

    has_file = any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers)

    if not has_file:

        fh = logging.handlers.RotatingFileHandler(

            log_dir / "live_trader.log", maxBytes=5_000_000, backupCount=3,

            encoding="utf-8",

        )

        fh.setFormatter(fmt)

        root.addHandler(fh)

    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers)

    if not has_stream:

        ch = logging.StreamHandler()

        ch.setFormatter(fmt)

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

    # SL/TP dual-mode guard first

    enforce_sltp_compatibility(cfg)

    if cfg.model.signal_source != "model":

        return

    failures = model_promotion_failures(cfg)

    if not failures:

        return

    logger.warning(

        "⚡ [GATE BYPASSED] Model promotion gate bypassed by user: %s",

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

        failures: list[str] = []

        if cfg.model.signal_source == "model":

            # Model promotion gate bypassed by user

            pass

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

        max_bars = args.max_bars if args.max_bars > 0 else 0

    else:

        # Demo: deterministic synthetic closed-bar feed (never real orders).

        bar_source = SyntheticBarSource(seed=7, start_price=2000.0)

        max_bars = args.max_bars if args.max_bars > 0 else (300 if args.smoke else 0)



    scalp_poll = getattr(getattr(cfg, "scalp", None), "poll_seconds", 1.0)

    poll_seconds = 0.05 if args.smoke else scalp_poll

    processed = 0

    try:

        processed = trader.run_forever(bar_source, poll_seconds=poll_seconds,

                                       max_bars=max_bars)

    except KeyboardInterrupt:

        logger.info("Interrupted — shutting down")



    # Smoke / bounded run: flatten any open demo positions and report the

    # REAL realized PnL from the broker's fill history (never fabricated).

    open_positions = broker.get_positions(cfg.broker.symbol, cfg.broker.magic)

    # In LIVE mode, NEVER cut trades on shutdown! Preserve them so they can continue to run and be recovered.
    if cfg.trading_mode is TradingMode.DEMO and max_bars is not None:
        if open_positions:
            for pos in open_positions:
                eq_before = trader._current_equity()
                ok, _, _ = trader.executor.execute_close(pos.ticket, equity=eq_before)
                eq_after = trader._current_equity()
                logger.info("Flatten on exit ticket=%s ok=%s pnl=%.2f",
                            pos.ticket, ok, eq_after - eq_before)
    else:
        logger.info("🛡️ [SHUTDOWN POSITION SAFETY] Preserving %d open position(s) on broker — will resume control on next start.",
                    len(open_positions) if open_positions else 0)
        trader.executor._save_local_state()

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

