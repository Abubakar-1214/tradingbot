"""
Risk Supervisor — deterministic, persistent safety layer.

This is NOT a neural network. It is a set of deterministic circuit breakers
that override AI / strategy decisions. It MUST always be active during live
trading.

P0-2 fixes (audit):
  * ``check_trade`` now takes an EXPLICIT position fraction (``requested_size``)
    instead of treating the action index (0/1/2) as a fraction — the old
    ``abs(action)`` logic rejected every non-flat trade with
    ``POSITION_TOO_LARGE: 100.00% > 10.00%``.
  * State is persisted to SQLite (``state/risk_state.db``) and reloaded at
    ``__init__`` — a process restart can no longer bypass the daily-loss /
    drawdown / consecutive-loss / halt circuit breakers.
  * The correlation guard is configured via ``RiskConfig`` (default OFF) and
    applies to BOTH long and short legs; the old hard-coded DXY rule is gone.

State persisted:
    daily_pnl (absolute), daily_start_equity (denominator for the daily-loss
    ratio), peak_equity, current_equity, trades_today, consecutive_losses,
    halt_until_ts, last_trade_ts, trade_date (UTC YYYY-MM-DD).

All counters auto-reset at the UTC day boundary (via the injectable clock).

Windows note: SQLite connections are explicitly closed on every read/write
(``self._db()`` context manager).  Python's ``with sqlite3.connect(...)``
context manager only commits/rolls back the transaction — it does NOT close
the connection, which keeps the DB file locked on Windows and breaks temp-dir
cleanup / file replacement.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

# Defaults — kept in sync with core.config.RiskConfig so the supervisor can be
# constructed standalone (dict config) as well as from AppConfig.
DEFAULTS: Dict[str, Any] = {
    "max_daily_loss": 0.05,
    "max_position": 0.10,
    "max_drawdown": 0.15,
    "vol_threshold": 3.0,
    "max_spread": 0.0005,
    "max_trades_per_day": 20,
    "min_trade_interval_sec": 300.0,
    "max_consecutive_losses": 5,
    "market_hours_only": False,
    "correlation_guard_enabled": False,
    "correlation_asset": "DXY",
    "correlation_block_long_on_up": 0.01,
    "event_position_scale": 0.5,  # fraction of max_position allowed in event windows
    "initial_equity": 10_000.0,
}


def _cfg(config: Optional[Any], name: str) -> Any:
    """Read a config value from either a dict or a dataclass-like object."""
    if config is None:
        return DEFAULTS[name]
    if isinstance(config, dict):
        return config.get(name, DEFAULTS[name])
    return getattr(config, name, DEFAULTS[name])


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RiskSupervisor:
    """Deterministic, SQLite-persisted circuit-breaker layer."""

    def __init__(
        self,
        config: Optional[Any] = None,
        db_path: Optional[os.PathLike] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        """
        Args:
            config: dict or RiskConfig. Defaults to conservative defaults.
            db_path: SQLite file for persistent risk state. Defaults to
                ``state/risk_state.db`` under the repo root.
            now_fn: injectable clock (used by tests); defaults to UTC now.
        """
        if config is None:
            config = DEFAULTS

        # --- configuration ------------------------------------------------- #
        self.max_daily_loss = float(_cfg(config, "max_daily_loss"))
        self.max_position_size = float(_cfg(config, "max_position"))
        self.max_drawdown = float(_cfg(config, "max_drawdown"))
        self.volatility_threshold = float(_cfg(config, "vol_threshold"))
        self.max_spread = float(_cfg(config, "max_spread"))
        self.max_trades_per_day = int(_cfg(config, "max_trades_per_day"))
        self.min_time_between_trades = float(_cfg(config, "min_trade_interval_sec"))
        self.max_consecutive_losses = int(_cfg(config, "max_consecutive_losses"))
        self.market_hours_only = bool(_cfg(config, "market_hours_only"))
        self.correlation_guard_enabled = bool(_cfg(config, "correlation_guard_enabled"))
        self.correlation_asset = str(_cfg(config, "correlation_asset"))
        self.correlation_block_long_on_up = float(_cfg(config, "correlation_block_long_on_up"))
        self.event_position_scale = float(_cfg(config, "event_position_scale"))
        self.initial_equity = float(_cfg(config, "initial_equity"))

        for name, val in [
            ("max_daily_loss", self.max_daily_loss),
            ("max_position", self.max_position_size),
            ("max_drawdown", self.max_drawdown),
        ]:
            if not 0.0 < val < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {val}")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be >= 1")

        # --- clock & persistence ------------------------------------------- #
        self._now_fn = now_fn or _utc_now
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent / "state" / "risk_state.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # --- in-memory runtime state (synced from/to SQLite) ---------------- #
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = self.initial_equity
        self.peak_equity: float = self.initial_equity
        self.current_equity: float = self.initial_equity
        self.trades_today: int = 0
        self.consecutive_losses: int = 0
        self.halt_until: Optional[datetime] = None
        self.last_trade_time: Optional[datetime] = None
        self.trade_date: Optional[str] = None
        self.trade_history: list = []

        # --- statistics ----------------------------------------------------- #
        self.total_trades_approved = 0
        self.total_trades_rejected = 0
        self.rejection_reasons: Dict[str, int] = {}

        self._init_db()
        self._load_state()

        logger.info(
            "RiskSupervisor initialized (daily_loss=%.1f%%, dd=%.1f%%, max_pos=%.1f%%, "
            "max_losses=%d, corr_guard=%s)",
            self.max_daily_loss, self.max_drawdown, self.max_position_size,
            self.max_consecutive_losses, self.correlation_guard_enabled,
        )

    # ------------------------------------------------------------------ #
    # Persistence (SQLite) — connections are ALWAYS explicitly closed
    # ------------------------------------------------------------------ #
    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """Yield an open connection; commit + close on exit (Windows-safe)."""
        con = sqlite3.connect(str(self.db_path))
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _init_db(self) -> None:
        with self._db() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_state (
                    id                 INTEGER PRIMARY KEY CHECK (id = 1),
                    daily_pnl          REAL NOT NULL DEFAULT 0,
                    daily_start_equity REAL NOT NULL DEFAULT 10000,
                    peak_equity        REAL NOT NULL DEFAULT 10000,
                    current_equity     REAL NOT NULL DEFAULT 10000,
                    trades_today       INTEGER NOT NULL DEFAULT 0,
                    consecutive_losses INTEGER NOT NULL DEFAULT 0,
                    halt_until_ts      REAL,
                    last_trade_ts      REAL,
                    trade_date         TEXT
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_history (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts         REAL NOT NULL,
                    pnl        REAL NOT NULL,
                    equity     REAL NOT NULL,
                    daily_pnl  REAL NOT NULL,
                    is_win     INTEGER
                )
                """
            )

    def _load_state(self) -> None:
        try:
            with self._db() as con:
                row = con.execute("SELECT * FROM risk_state WHERE id = 1").fetchone()
            if row is None:
                return  # fresh DB -> keep defaults
            (
                _id, self.daily_pnl, self.daily_start_equity, self.peak_equity,
                self.current_equity, self.trades_today, self.consecutive_losses,
                halt_ts, last_ts, self.trade_date,
            ) = row
            self.halt_until = datetime.fromtimestamp(halt_ts, tz=timezone.utc) if halt_ts else None
            self.last_trade_time = datetime.fromtimestamp(last_ts, tz=timezone.utc) if last_ts else None
            with self._db() as con:
                rows = con.execute(
                    "SELECT ts, pnl, equity, daily_pnl, is_win FROM trade_history ORDER BY id"
                ).fetchall()
            self.trade_history = [
                {
                    "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc),
                    "pnl": pnl, "equity": equity, "daily_pnl": dpnl, "is_win": bool(iw),
                }
                for ts, pnl, equity, dpnl, iw in rows
            ]
            logger.info("Risk state loaded from %s", self.db_path)
        except Exception:  # pragma: no cover - defensive: never crash on load
            logger.exception("Failed to load risk state from %s; using defaults", self.db_path)

    def _persist(self) -> None:
        halt_ts = self.halt_until.timestamp() if self.halt_until else None
        last_ts = self.last_trade_time.timestamp() if self.last_trade_time else None
        with self._db() as con:
            con.execute(
                """
                INSERT INTO risk_state (id, daily_pnl, daily_start_equity, peak_equity,
                                        current_equity, trades_today, consecutive_losses,
                                        halt_until_ts, last_trade_ts, trade_date)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    daily_pnl=excluded.daily_pnl,
                    daily_start_equity=excluded.daily_start_equity,
                    peak_equity=excluded.peak_equity,
                    current_equity=excluded.current_equity,
                    trades_today=excluded.trades_today,
                    consecutive_losses=excluded.consecutive_losses,
                    halt_until_ts=excluded.halt_until_ts,
                    last_trade_ts=excluded.last_trade_ts,
                    trade_date=excluded.trade_date
                """,
                (self.daily_pnl, self.daily_start_equity, self.peak_equity,
                 self.current_equity, self.trades_today, self.consecutive_losses,
                 halt_ts, last_ts, self.trade_date),
            )

    # ------------------------------------------------------------------ #
    # Daily reset
    # ------------------------------------------------------------------ #
    def _maybe_reset_daily(self) -> None:
        today = self._now_fn().strftime("%Y-%m-%d")
        if self.trade_date is None:
            self.trade_date = today
            self.daily_start_equity = self.current_equity or self.initial_equity
            self._persist()
            return
        if self.trade_date != today:
            logger.info(
                "Daily reset: %s -> %s (daily_pnl was %.4f, trades %d)",
                self.trade_date, today, self.daily_pnl, self.trades_today,
            )
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.daily_start_equity = self.current_equity or self.initial_equity
            self.trade_date = today
            if self.halt_until and self._now_fn() >= self.halt_until:
                self.halt_until = None
                logger.info("Trading halt cleared by daily reset")
            self._persist()

    def reset_daily(self) -> None:
        """Force a daily reset (e.g. called at midnight UTC by the live loop)."""
        self.trade_date = None
        self._maybe_reset_daily()

    # ------------------------------------------------------------------ #
    # Trade gate
    # ------------------------------------------------------------------ #
    def check_trade(
        self,
        signal: Any,
        requested_size: float,
        state: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Approve or reject a proposed trade.

        Args:
            signal: strategy action — int 0=flat, 1=long, 2=short (matching the
                env action space), or a str like 'flat'/'long'/'short'.
            requested_size: EXPLICIT position fraction of equity (0 < f <= 1).
                This is computed by the PositionSizer; it is NEVER derived from
                ``abs(action)`` (P0-2 fix).
            state: dict with 'position', 'equity', 'is_market_open', ...
            market_data: dict with 'volatility', 'spread',
                '<asset>_momentum', 'is_high_impact_event', 'is_event_window', ...

        Returns:
            (approved: bool, reason: str)
        """
        state = state or {}
        market_data = market_data or {}

        self._maybe_reset_daily()

        direction = self._direction(signal)
        if direction is None:
            return self._reject("INVALID_SIGNAL")

        # Equity source: live callers pass state['equity']; fall back to stored.
        equity = float(state.get("equity", self.current_equity) or self.current_equity)
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        # 1. CIRCUIT BREAKER: daily loss limit
        ratio = self._daily_loss_ratio()
        if ratio <= -self.max_daily_loss:
            self._set_halt(timedelta(hours=24))
            return self._reject(
                f"CIRCUIT_BREAKER: daily loss {ratio:.2%} <= -{self.max_daily_loss:.2%}"
            )

        # 2. Trading halt
        if self.halt_until and self._now_fn() < self.halt_until:
            remaining = (self.halt_until - self._now_fn()).total_seconds() / 3600
            return self._reject(
                f"HALTED until {self.halt_until.strftime('%H:%M')} ({remaining:.1f}h remaining)"
            )

        # 3. Max drawdown vs peak equity
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.current_equity) / self.peak_equity
            if dd > self.max_drawdown:
                return self._reject(f"MAX_DRAWDOWN: {dd:.2%} > {self.max_drawdown:.2%}")

        # 4. Position size limit (explicit fraction, not abs(action))
        if not 0.0 < requested_size <= 1.0:
            return self._reject(f"INVALID_SIZE: {requested_size:.4f}")
        if requested_size > self.max_position_size:
            return self._reject(
                f"POSITION_TOO_LARGE: {requested_size:.2%} > {self.max_position_size:.2%}"
            )

        # 5. Consecutive losses breaker
        if self.consecutive_losses >= self.max_consecutive_losses:
            return self._reject(
                f"TOO_MANY_LOSSES: {self.consecutive_losses} consecutive losses"
            )

        # 6. Volatility filter — block NEW entries in violent markets.
        vol = float(market_data.get("volatility", 0.0) or 0.0)
        current_position = int(state.get("position", 0) or 0)
        is_new_entry = direction != 0 and current_position == 0
        if vol > self.volatility_threshold and is_new_entry:
            return self._reject(f"HIGH_VOLATILITY: {vol:.2f} > {self.volatility_threshold}")

        # 7. Correlation guard (config-driven, default OFF)
        if self.correlation_guard_enabled and direction != 0:
            guard_reason = self._correlation_reject(direction, market_data)
            if guard_reason:
                return self._reject(guard_reason)

        # 8. Event risk filter — halve the allowed size in high-impact windows.
        is_high_impact = bool(market_data.get("is_high_impact_event", False))
        is_event_window = bool(market_data.get("is_event_window", False))
        if (is_high_impact or is_event_window) and requested_size > (
            self.event_position_scale * self.max_position_size
        ):
            allowed = self.event_position_scale * self.max_position_size
            return self._reject(f"EVENT_RISK: max position {allowed:.2%} during news")

        # 9. Overtrading / rate limits
        if self.trades_today >= self.max_trades_per_day:
            return self._reject(f"MAX_TRADES: daily limit reached ({self.max_trades_per_day})")

        if self.last_trade_time is not None:
            elapsed = (self._now_fn() - self.last_trade_time).total_seconds()
            if elapsed < self.min_time_between_trades:
                remaining = self.min_time_between_trades - elapsed
                return self._reject(f"COOLDOWN: wait {remaining:.0f}s before next trade")

        # 10. Spread filter
        spread = float(market_data.get("spread", 0.0) or 0.0)
        if spread > self.max_spread:
            return self._reject(f"SPREAD_TOO_WIDE: {spread:.5f} > {self.max_spread:.5f}")

        # 11. Market-hours filter
        if self.market_hours_only:
            is_open = market_data.get("is_market_open", True)
            if not is_open:
                return self._reject("MARKET_CLOSED: trading outside market hours")

        # Persist equity/peak changes made during the check.
        self._persist()
        return self._approve()

    # ------------------------------------------------------------------ #
    # Post-trade state update
    # ------------------------------------------------------------------ #
    def update_state(
        self,
        pnl: float,
        equity: float,
        is_win: Optional[bool] = None,
    ) -> None:
        """
        Update + persist supervisor state after a trade closes.

        Args:
            pnl: realised P&L for this trade (absolute).
            equity: current account equity after the trade.
            is_win: True if profitable, False if a loss, None if unknown.
        """
        self._maybe_reset_daily()

        self.daily_pnl += float(pnl)
        self.current_equity = float(equity)
        self.peak_equity = max(self.peak_equity, self.current_equity)
        self.trades_today += 1
        self.last_trade_time = self._now_fn()

        if is_win is not None:
            if is_win:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1

        self.trade_history.append({
            "timestamp": self.last_trade_time,
            "pnl": float(pnl),
            "equity": self.current_equity,
            "daily_pnl": self.daily_pnl,
            "is_win": is_win,
        })

        with self._db() as con:
            con.execute(
                "INSERT INTO trade_history (ts, pnl, equity, daily_pnl, is_win) VALUES (?, ?, ?, ?, ?)",
                (
                    self.last_trade_time.timestamp(), float(pnl), self.current_equity,
                    self.daily_pnl, 1 if is_win else 0,
                ),
            )
        self._persist()

        if self.consecutive_losses >= 3:
            logger.warning("%d consecutive losses", self.consecutive_losses)
        dd = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity else 0.0
        if dd > 0.10:
            logger.warning("Drawdown: %.2%%", dd)

    # ------------------------------------------------------------------ #
    # Halts & introspection
    # ------------------------------------------------------------------ #
    def emergency_shutdown(self) -> str:
        """Halt all trading until a manual restart (365 days)."""
        self._set_halt(timedelta(days=365))
        logger.critical("EMERGENCY SHUTDOWN ACTIVATED — all trading halted, manual restart required")
        return "EMERGENCY_SHUTDOWN"

    def halt_remaining_seconds(self) -> float:
        if not self.halt_until:
            return 0.0
        remaining = (self.halt_until - self._now_fn()).total_seconds()
        return max(0.0, remaining)

    def get_statistics(self) -> Dict[str, Any]:
        total = self.total_trades_approved + self.total_trades_rejected
        return {
            "total_checks": total,
            "approved": self.total_trades_approved,
            "rejected": self.total_trades_rejected,
            "approval_rate": (self.total_trades_approved / total) if total else 0.0,
            "rejection_rate": (self.total_trades_rejected / total) if total else 0.0,
            "rejection_reasons": self.rejection_reasons,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "daily_pnl": self.daily_pnl,
            "daily_loss_ratio": self._daily_loss_ratio(),
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "halt_remaining_sec": self.halt_remaining_seconds(),
            "trade_date": self.trade_date,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Default conservative configuration (dict form)."""
        return dict(DEFAULTS)

    def _daily_loss_ratio(self) -> float:
        base = self.daily_start_equity if self.daily_start_equity > 0 else self.initial_equity
        return self.daily_pnl / base if base else 0.0

    def _direction(self, signal: Any) -> Optional[int]:
        """Normalise signal -> 0 flat / 1 long / 2 short."""
        if isinstance(signal, str):
            s = signal.strip().lower()
            if s in ("flat", "close", "exit", "0", "hold"):
                return 0
            if s in ("long", "buy", "1"):
                return 1
            if s in ("short", "sell", "2"):
                return 2
            return None
        if isinstance(signal, (int, float)) and not isinstance(signal, bool):
            v = int(signal)
            if v in (0, 1, 2):
                return v
        if isinstance(signal, dict):
            return self._direction(signal.get("action", 0))
        return None

    def _correlation_reject(self, direction: int, market_data: Dict[str, Any]) -> Optional[str]:
        """Correlation guard (long blocked on strong up-momentum, short on down)."""
        asset_key = f"{self.correlation_asset.lower()}_momentum"
        momentum = float(market_data.get(asset_key, market_data.get("dxy_momentum", 0.0)) or 0.0)
        threshold = self.correlation_block_long_on_up
        if direction == 1 and momentum > threshold:
            return f"CORRELATION_GUARD: {self.correlation_asset} momentum {momentum:.3f} blocks LONG"
        if direction == 2 and momentum < -threshold:
            return f"CORRELATION_GUARD: {self.correlation_asset} momentum {momentum:.3f} blocks SHORT"
        return None

    def _set_halt(self, duration: timedelta) -> None:
        self.halt_until = self._now_fn() + duration
        self._persist()

    def _approve(self) -> Tuple[bool, str]:
        self.total_trades_approved += 1
        return True, "APPROVED"

    def _reject(self, reason: str) -> Tuple[bool, str]:
        self.total_trades_rejected += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        logger.warning("Trade REJECTED: %s", reason)
        return False, reason


# --------------------------------------------------------------------------- #
# AI wrapper
# --------------------------------------------------------------------------- #
class SafeTradingAgent:
    """
    Wrapper that combines an AI/strategy agent with the RiskSupervisor.

    The agent proposes an action + size; the supervisor approves/rejects.
    On rejection the action is overridden to flat (0) and a reason recorded.
    """

    def __init__(self, ai_agent: Any, risk_supervisor: Optional[RiskSupervisor] = None) -> None:
        self.ai_agent = ai_agent
        self.risk_supervisor = risk_supervisor or RiskSupervisor()
        logger.info("SafeTradingAgent initialized")

    def act(self, obs, state, market_data, requested_size=None):
        """Get an action with safety checks.

        Returns:
            (final_action, info) — final_action is 0 when rejected.
        """
        decision = self.ai_agent.act(obs)
        action = decision if isinstance(decision, (int, float)) else decision.get("action", 0)
        if requested_size is None:
            requested_size = float(decision.get("size", self.risk_supervisor.max_position_size * 0.5))
        approved, reason = self.risk_supervisor.check_trade(action, requested_size, state, market_data)
        if not approved:
            return 0, {
                "approved": False, "reason": reason, "ai_action": action,
                "final_action": 0, "requested_size": requested_size,
            }
        return action, {
            "approved": True, "reason": reason, "ai_action": action,
            "final_action": action, "requested_size": requested_size,
        }


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        sup = RiskSupervisor(db_path=Path(td) / "demo.db")
        state = {"position": 0, "equity": 10_000.0}
        market_data = {
            "volatility": 1.5, "spread": 0.0003, "dxy_momentum": -0.005,
            "is_high_impact_event": False, "is_event_window": False,
        }

        ok, reason = sup.check_trade(1, requested_size=0.05, state=state, market_data=market_data)
        print(f"normal long 5%:  {ok} — {reason}")

        ok, reason = sup.check_trade(1, requested_size=0.50, state=state, market_data=market_data)
        print(f"oversized 50%:  {ok} — {reason}")

        market_data["volatility"] = 4.0
        ok, reason = sup.check_trade(1, requested_size=0.05, state=state, market_data=market_data)
        print(f"high vol entry: {ok} — {reason}")

        sup.update_state(pnl=-700.0, equity=9_300.0, is_win=False)
        ok, reason = sup.check_trade(1, requested_size=0.05, state={"equity": 9_300.0}, market_data=market_data)
        print(f"after -7% day:  {ok} — {reason}")

        print("\nstatistics:", sup.get_statistics())
