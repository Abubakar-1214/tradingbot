"""Shared deterministic helpers for the pytest suite (mirrors verify_risk_integration.py)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

from core.config import AppConfig, RiskConfig
from live.mock_broker import MockBroker
from live.trade_executor import TradeExecutor
from models.position_sizing import ATRPositionSizer
from models.risk_supervisor import RiskSupervisor

EQUITY = 100_000.0


class FakeClock:
    """Injectable deterministic clock for the RiskSupervisor."""

    def __init__(self, start: datetime):
        self.now = start

    def advance(self, **kw) -> None:
        self.now += timedelta(**kw)

    def __call__(self) -> datetime:
        return self.now


class SmallSizer(ATRPositionSizer):
    """Target 3% of equity per order (below the 10% cap) so scaling is testable."""

    def compute_position_size(self, atr, price, equity=1.0):
        return 0.03


def make_config(**risk_overrides) -> AppConfig:
    risk_kw = dict(
        max_daily_loss=0.05,
        max_drawdown=0.15,
        max_consecutive_losses=5,
        risk_per_trade=0.02,
        max_position=0.10,
        max_trades_per_day=20,
        min_trade_interval_sec=300.0,
        vol_threshold=3.0,
        max_spread=0.0005,
        market_hours_only=False,
        correlation_guard_enabled=False,
        correlation_asset="DXY",
        correlation_block_long_on_up=0.01,
    )
    risk_kw.update(risk_overrides)
    return AppConfig(risk=RiskConfig(**risk_kw))


def sample_df(n: int = 200, start_price: float = 2000.0) -> pd.DataFrame:
    """Deterministic flat-ish OHLC frame with a stable ~4.0 ATR for sizing."""
    rng = np.random.default_rng(7)
    closes = start_price + rng.normal(0, 1.5, n).cumsum() * 0.05
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "time": times, "open": closes, "high": closes + 2.0,
        "low": closes - 2.0, "close": closes, "volume": 100.0,
    })


def build_executor(clock: FakeClock, cfg: AppConfig, tmpdir: Path,
                   equity: float = EQUITY) -> TradeExecutor:
    """TradeExecutor on a seeded MockBroker + fresh SQLite risk state."""
    broker = MockBroker(symbol="XAUUSD", balance=equity, seed=42)
    broker.connect()
    broker.set_mid(2000.0)
    risk = RiskSupervisor(config=cfg.risk, db_path=tmpdir / "risk.db", now_fn=clock)
    # Ratio breakers use the equity base as the denominator — set it to the
    # trading equity instead of the RiskSupervisor DEFAULTS $10k.
    risk.initial_equity = equity
    risk.current_equity = equity
    risk.daily_start_equity = equity
    risk.peak_equity = equity
    ex = TradeExecutor(broker, risk, cfg, sizer=SmallSizer(),
                       local_state_file=tmpdir / "bot_state.json")
    return ex


def calm_md(ex: TradeExecutor, df: pd.DataFrame) -> Dict[str, float]:
    """Market-data dict with volatility below the 3.0 z-score threshold."""
    md = ex.build_market_data(df, ex.broker.get_tick("XAUUSD"))
    md["volatility"] = 0.5
    return md
