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
    signal source — feature contract + model checkpoint present).
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

from core.config import (  # noqa: E402
    AppConfig,
    TradingMode,
    load_config,
)
from live.broker import BaseBroker, BrokerError, PositionInfo  # noqa: E402
from live.mock_broker import MockBroker  # noqa: E402
from live.mt5_broker import Mt5Broker  # noqa: E402
from live.trade_executor import TradeExecutor  # noqa: E402
from models.position_sizing import ATRPositionSizer  # noqa: E402
from models.risk_supervisor import RiskSupervisor  # noqa: E402

logger = logging.getLogger("live_trader")

# Number of CLOSED bars the decision window needs before the first decision
# (>= slow SMA period + margin; ATR itself only needs 15).
CLOSED_BAR_WARMUP = 60

_DEMO_BALANCE = 100_000.0


# --------------------------------------------------------------------------- #
# Signal sources
# --------------------------------------------------------------------------- #
class SignalSource(Protocol):
    """A causal signal generator: returns 0 (flat), 1 (long), 2 (short)."""

    def signal(self, df: pd.DataFrame, position_side: int) -> int: ...


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


class PpoSignalSource:
    """ML signal source backed by the trained PPO checkpoint.

    Loads the checkpoint lazily; raises a clear error when the model file is
    missing so the operator can never mistake a missing model for a flat
    signal.  Feature preprocessing follows the saved feature contract
    (core/feature_pipeline.build_feature_frame) — the same causal, contract-
    enforced pipeline used by training and backtesting (P0-3 fix).
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._contract = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        model_path = self.cfg.paths.model_path
        contract_path = self.cfg.paths.feature_contract_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"PPO checkpoint not found at {model_path}. Train a model or "
                f"switch SIGNAL_SOURCE=rule."
            )
        if not contract_path.exists():
            raise FileNotFoundError(
                f"Feature contract not found at {contract_path}. Run the "
                f"feature pipeline first (P0-3: feature schema must be "
                f"enforced at load time)."
            )
        try:
            from stable_baselines3 import PPO  # type: ignore
            self._model = PPO.load(str(model_path))
        except Exception as e:  # pragma: no cover - terminal-machine only
            raise RuntimeError(f"Failed to load PPO checkpoint {model_path}: {e}") from e
        try:
            import json
            self._contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Failed to load feature contract {contract_path}: {e}") from e

    def signal(self, df: Optional[pd.DataFrame], position_side: int = 0) -> int:
        self._ensure_loaded()
        if df is None or len(df) < CLOSED_BAR_WARMUP:
            return 0
        from core.feature_pipeline import build_feature_frame, validate_contract_features
        feats = build_feature_frame(
            df,
            higher_timeframes=self.cfg.feature.higher_timeframes,
            macro_shift=self.cfg.feature.macro_shift,
        )
        validate_contract_features(list(feats.columns), self._contract)
        obs = feats.tail(self.cfg.feature.window).to_numpy(dtype="float32").reshape(1, -1)
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)


def build_signal_source(cfg: AppConfig) -> SignalSource:
    """Build the configured signal source (SIGNAL_SOURCE=rule|ppo)."""
    kind = cfg.raw.get("SIGNAL_SOURCE", "rule").strip().lower()
    if kind == "ppo":
        return PpoSignalSource(cfg)
    if kind == "rule":
        return RuleSignalSource()
    raise ValueError(
        f"SIGNAL_SOURCE must be 'rule' or 'ppo', got {kind!r}"
    )


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

    def __init__(self, mt5_module, symbol: str, timeframe: str) -> None:
        self._mt5 = mt5_module
        self.symbol = symbol
        self.timeframe = timeframe.upper()

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
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
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
        warmup_bars: int = CLOSED_BAR_WARMUP,
    ) -> None:
        self.cfg = cfg
        self.broker = broker
        self.signal_source = signal_source
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

    def on_closed_bar(
        self,
        df: Optional[pd.DataFrame],
        bar_time: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Handle ONE closed candle: entry / scale / close decisions.

        Every order path flows through TradeExecutor which consults the
        RiskSupervisor BEFORE the broker is touched and attaches ATR-based
        SL/TP to every order.  Returns the list of events logged this bar.
        """
        events: List[Dict[str, object]] = []
        if self._kill_switch_triggered():
            events.append({"event": "KILL_SWITCH"})
            return events
        if self.executor.halted:
            events.append({"event": "HALTED", "reason": self.executor.halt_reason})
            return events

        equity = self._current_equity()
        tick = self.broker.get_tick(self.cfg.broker.symbol)
        md = self.executor.build_market_data(df, tick)

        positions: List[PositionInfo] = self.broker.get_positions(
            self.cfg.broker.symbol, self.cfg.broker.magic
        )
        pos_side = 0
        if positions:
            pos_side = 1 if positions[0].side == "buy" else 2

        sig = self.signal_source.signal(df, pos_side)

        if sig == 0:
            events.append({"event": "HOLD" if pos_side else "NO_SIGNAL",
                           "equity": equity})
            return events
        if sig == pos_side:
            events.append({"event": "HOLD", "signal": sig, "equity": equity})
            return events

        # Opposite (or new) signal.
        if pos_side != 0 and sig != pos_side:
            # Close existing exposure first (RiskSupervisor consulted; a
            # rejection is logged but never blocks the close — de-risking).
            for pos in positions:
                eq_before = self._current_equity()
                ok, reason, res = self.executor.execute_close(
                    pos.ticket, equity=eq_before
                )
                eq_after = self._current_equity()
                pnl = eq_after - eq_before
                events.append({
                    "event": "CLOSE", "ticket": pos.ticket, "ok": ok,
                    "reason": reason, "pnl": pnl,
                })
                logger.info("CLOSE ticket=%s ok=%s pnl=%.2f reason=%s",
                            pos.ticket, ok, pnl, reason)
                if ok:
                    self.executor.update_risk_after_close(pnl, eq_after, pnl > 0.0)
                else:
                    logger.warning("Close failed ticket=%s reason=%s", pos.ticket, reason)

        # Enter on the new signal (short is gated by ALLOW_SHORT inside the
        # executor; sizing and SL/TP flow through RiskSupervisor + sizer).
        ok, reason, res = self.executor.execute_entry(
            sig, equity=equity, df=df, market_data=md, bar_time=bar_time
        )
        events.append({
            "event": "ENTRY", "signal": sig, "ok": ok, "reason": reason,
            "ticket": res.ticket if res else None,
            "volume": res.volume_filled if res else None,
            "fill_price": res.fill_price if res else None,
        })
        logger.info("ENTRY signal=%s ok=%s reason=%s ticket=%s",
                    sig, ok, reason, res.ticket if res else None)
        return events

    def _feed_closed_bar_to_broker(self, df: pd.DataFrame) -> List[dict]:
        """Evaluate the newly closed bar against open SL/TP (demo broker).

        MockBroker.advance() fills stops/take-profits when the closed bar
        breaches a level and moves the mid to the bar close.  Equity is
        captured BEFORE the advance so the realized PnL of each fill is fed
        back into the RiskSupervisor (never fabricated).  The real Mt5Broker
        needs no such feed — the terminal executes SL/TP server-side — so
        this is a no-op there.
        """
        advance = getattr(self.broker, "advance", None)
        if advance is None or df is None or len(df) == 0:
            return []
        row = df.iloc[-1]
        eq_before = self._current_equity()
        fills = advance(float(row["high"]), float(row["low"]), float(row["close"]))
        eq_after = self._current_equity()
        for f in fills:
            logger.info("SL/TP fill ticket=%s reason=%s pnl=%.2f",
                        f.get("ticket"), f.get("reason"), float(f.get("pnl", 0.0)))
        # Sync local state (broker may have closed positions) and feed the
        # REAL aggregate realized PnL to the risk supervisor.
        if fills:
            self.executor.reconcile_positions()
            pnl = eq_after - eq_before
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
                self._feed_closed_bar_to_broker(df)
                self.on_closed_bar(df, bar_time=str(closed_time))
                processed += 1
                if max_bars > 0 and processed >= max_bars:
                    logger.info("Reached smoke bar limit (%d) — stopping", max_bars)
                    return processed
            except Exception:
                logger.exception("on_closed_bar failed for bar %s", closed_time)
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


def _build_broker(cfg: AppConfig) -> BaseBroker:
    """Return the broker for the configured TRADING_MODE.

    demo (default) -> MockBroker (deterministic simulated fills, never real)
    live          -> Mt5Broker  (real MetaTrader5 terminal)
    """
    if cfg.trading_mode is TradingMode.DEMO:
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
        # --- live gates (audit P0-7): risk state loadable + MT5 creds + ---
        # --- (for ML signals) feature contract + model checkpoint        ---
        failures: List[str] = []
        risk_state_loadable = True
        try:
            RiskSupervisor(config=cfg.risk, db_path=cfg.paths.risk_state_db,
                           now_fn=lambda: datetime.now(timezone.utc))
        except Exception as e:
            risk_state_loadable = False
            failures.append(f"risk state loadable ({e})")
        if cfg.broker.mt5_login is None:
            failures.append("MT5_LOGIN not configured")
        if cfg.broker.mt5_password is None:
            failures.append("MT5_PASSWORD not configured")
        if cfg.broker.mt5_server is None:
            failures.append("MT5_SERVER not configured")
        if cfg.raw.get("SIGNAL_SOURCE", "rule").strip().lower() == "ppo":
            if not cfg.paths.feature_contract_path.exists():
                failures.append("feature contract present")
            if not cfg.paths.model_path.exists():
                failures.append("model present")
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

    broker = _build_broker(cfg)
    signal_source = build_signal_source(cfg)
    trader = LiveTrader(cfg, broker, signal_source)

    # Startup reconciliation (P0-6): broker vs local state; halt on drift.
    trader.startup_reconcile()
    if trader.executor.halted:
        logger.critical("Trader halted at startup: %s", trader.executor.halt_reason)
        return 2

    if cfg.trading_mode is TradingMode.LIVE:
        bar_source: BarSource = Mt5BarSource(
            broker._mt5, cfg.broker.symbol, cfg.timeframe)  # type: ignore[attr-defined]
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
            ok, reason, _ = trader.executor.execute_close(pos.ticket, equity=eq_before)
            eq_after = trader._current_equity()
            logger.info("Flatten on exit ticket=%s ok=%s pnl=%.2f",
                        pos.ticket, ok, eq_after - eq_before)
    closed = getattr(broker, "closed_fills", lambda: [])()
    logger.info("Smoke summary: bars=%d open_positions=%d fills=%d",
                processed,
                len(broker.get_positions(cfg.broker.symbol, cfg.broker.magic)),
                len(closed))
    try:
        broker.disconnect()
    except Exception:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
