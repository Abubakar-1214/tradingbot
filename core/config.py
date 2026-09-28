"""
core/config.py — validated configuration layer for the trading bot.

Loads configuration from environment variables (via python-dotenv from a
``.env`` file at the repo root), validates every value into typed dataclasses
and provides the ``TRADING_MODE`` gate used by every entry point.

The live system MUST refuse to start in ``live`` mode unless all safety gates
pass (feature contract present, model present, risk state loadable,
reconciliation pass).  ``demo`` mode is the default and never sends real
orders to a live account.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is pinned in requirements
    load_dotenv = None  # type: ignore

# --------------------------------------------------------------------------- #
# Repo layout
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Enums / small value types
# --------------------------------------------------------------------------- #
class TradingMode(str, Enum):
    DEMO = "demo"
    LIVE = "live"


class Timeframe(str, Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureConfig:
    """Feature pipeline settings — shared by train, backtest and live."""

    window: int = 64                       # observation window (bars)
    higher_timeframes: tuple = ("H4", "D1")  # causal higher-TF context sources
    macro_shift: int = 1                   # shift applied to daily macro data
    drop_warmup: int = 200                 # bars dropped before features are stable
    normalize: bool = True                 # z-score using train-window stats only
    scaler_eps: float = 1e-8


@dataclass(frozen=True)
class RiskConfig:
    """Deterministic circuit-breaker configuration."""

    max_daily_loss: float = 0.05          # 5% max daily loss (fraction of equity)
    max_drawdown: float = 0.15            # 15% max drawdown from peak equity
    max_consecutive_losses: int = 5
    risk_per_trade: float = 0.02          # 2% of equity risked per trade
    max_position: float = 0.10            # 10% of equity max notional exposure
    max_trades_per_day: int = 20
    min_trade_interval_sec: float = 300.0
    vol_threshold: float = 3.0            # z-score of volatility vs baseline
    max_spread: float = 0.0005            # max spread fraction to trade into
    market_hours_only: bool = False       # True = block outside market hours
    correlation_guard_enabled: bool = False  # default OFF (was hardcoded before)
    correlation_asset: str = "DXY"        # asset whose momentum gates longs
    correlation_block_long_on_up: float = 0.01   # momentum threshold

    def __post_init__(self) -> None:
        for name, val in [
            ("max_daily_loss", self.max_daily_loss),
            ("max_drawdown", self.max_drawdown),
            ("risk_per_trade", self.risk_per_trade),
            ("max_position", self.max_position),
        ]:
            if not 0.0 < val < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {val}")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be >= 1")
        # Per-trade risk must not exceed the daily loss cap (equality allowed).
        if self.risk_per_trade > self.max_daily_loss:
            raise ValueError(
                f"risk_per_trade ({self.risk_per_trade}) must be <= "
                f"max_daily_loss ({self.max_daily_loss})"
            )


@dataclass(frozen=True)
class CostConfig:
    """Realistic cost model used by backtests and the mock broker."""

    spread: float = 0.00025          # 2.5 bp round-trip spread (~$0.50 on $2000)
    commission: float = 0.00003      # 0.3 bp per side
    slippage: float = 0.00005        # mean |slippage| per fill (vol-scaled)
    swap_long: float = -0.00004      # daily financing fraction of notional
    swap_short: float = -0.00002


@dataclass(frozen=True)
class BrokerConfig:
    """MT5 / execution settings."""

    symbol: str = "XAUUSD"
    timeframe: str = "H1"             # candle-close-aligned loop timeframe
    magic: int = 234000
    deviation: int = 20               # max price deviation for market orders (pts)
    mt5_login: Optional[str] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    mt5_path: Optional[str] = None    # path to terminal64.exe if auto-detect fails
    allow_short: bool = False         # enable SHORT positions (long+short)
    sl_atr_mult: float = 2.0          # stop-loss distance in ATR units
    tp_atr_mult: float = 3.0          # take-profit distance in ATR units
    max_requote_retries: int = 3
    order_retry_delay_sec: float = 1.0


@dataclass(frozen=True)
class PathConfig:
    """Filesystem locations (relative to repo root unless absolute)."""

    data_dir: Path = REPO_ROOT / "data"
    model_path: Path = REPO_ROOT / "train" / "ppo_xauusd_latest.zip"
    feature_contract_path: Path = REPO_ROOT / "train" / "feature_contract.json"
    risk_state_db: Path = REPO_ROOT / "state" / "risk_state.db"
    local_state_file: Path = REPO_ROOT / "state" / "bot_state.json"
    log_dir: Path = REPO_ROOT / "logs"
    kill_switch_path: Path = REPO_ROOT / "KILL_SWITCH"
    backtest_results_dir: Path = REPO_ROOT / "research" / "backtest_results"

    def __post_init__(self) -> None:
        for attr in ("data_dir", "log_dir", "backtest_results_dir"):
            p: Path = getattr(self, attr)
            if not p.is_absolute():
                object.__setattr__(self, attr, REPO_ROOT / p)


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    trading_mode: TradingMode = TradingMode.DEMO
    symbol_cfg: str = "XAUUSD"
    timeframe: str = "H1"
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    log_level: str = "INFO"
    raw: Dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_float(name: str, val: Optional[str], default: float) -> float:
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Env var {name}={val!r} is not a valid float")


def _as_int(name: str, val: Optional[str], default: int) -> int:
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Env var {name}={val!r} is not a valid int")


def _as_bool(name: str, val: Optional[str], default: bool) -> bool:
    if val is None or val.strip() == "":
        return default
    low = val.strip().lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Env var {name}={val!r} is not a valid boolean")


def load_config(env_file: Optional[str] = None, _env: Optional[Dict[str, str]] = None) -> AppConfig:
    """
    Load and validate the full configuration.

    Args:
        env_file: Optional path to a .env file.  Defaults to ``<repo>/.env``.
        _env: Optional pre-built environment dict (used by tests).

    Returns:
        Validated :class:`AppConfig`.
    """
    env: Dict[str, str] = dict(os.environ if _env is None else _env)

    if _env is None and load_dotenv is not None:
        dotenv_path = Path(env_file) if env_file else REPO_ROOT / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
            for k, v in os.environ.items():
                env[k] = v

    raw = dict(env)

    mode_raw = env.get("TRADING_MODE", "demo").strip().lower()
    try:
        mode = TradingMode(mode_raw)
    except ValueError:
        raise ValueError(
            f"TRADING_MODE must be 'demo' or 'live', got {mode_raw!r}"
        )

    symbol_raw = env.get("SYMBOL", "XAUUSD").strip().upper()
    if symbol_raw not in ("XAUUSD",):
        raise ValueError(f"Unsupported SYMBOL {symbol_raw!r} (only XAUUSD has data)")

    tf_raw = env.get("TIMEFRAME", "H1").strip().upper()
    if tf_raw not in Timeframe.__members__:
        raise ValueError(
            f"Unsupported TIMEFRAME {tf_raw!r}; supported: {list(Timeframe.__members__)}"
        )

    allow_short = _as_bool("ALLOW_SHORT", env.get("ALLOW_SHORT"), False)

    risk = RiskConfig(
        max_daily_loss=_as_float("MAX_DAILY_LOSS", env.get("MAX_DAILY_LOSS"), 0.05),
        max_drawdown=_as_float("MAX_DRAWDOWN", env.get("MAX_DRAWDOWN"), 0.15),
        max_consecutive_losses=_as_int("MAX_CONSECUTIVE_LOSSES", env.get("MAX_CONSECUTIVE_LOSSES"), 5),
        risk_per_trade=_as_float("MAX_RISK_PER_TRADE", env.get("MAX_RISK_PER_TRADE"), 0.02),
        max_position=_as_float("MAX_POSITION", env.get("MAX_POSITION"), 0.10),
        max_trades_per_day=_as_int("MAX_TRADES_PER_DAY", env.get("MAX_TRADES_PER_DAY"), 20),
        min_trade_interval_sec=_as_float("MIN_TRADE_INTERVAL_SEC", env.get("MIN_TRADE_INTERVAL_SEC"), 300.0),
        vol_threshold=_as_float("VOL_THRESHOLD", env.get("VOL_THRESHOLD"), 3.0),
        max_spread=_as_float("MAX_SPREAD", env.get("MAX_SPREAD"), 0.0005),
        market_hours_only=_as_bool("MARKET_HOURS_ONLY", env.get("MARKET_HOURS_ONLY"), False),
        correlation_guard_enabled=_as_bool("CORRELATION_GUARD_ENABLED", env.get("CORRELATION_GUARD_ENABLED"), False),
        correlation_asset=env.get("CORRELATION_ASSET", "DXY"),
        correlation_block_long_on_up=_as_float("CORRELATION_BLOCK_LONG_ON_UP", env.get("CORRELATION_BLOCK_LONG_ON_UP"), 0.01),
    )

    cost = CostConfig(
        spread=_as_float("COST_SPREAD", env.get("COST_SPREAD"), 0.00025),
        commission=_as_float("COST_COMMISSION", env.get("COST_COMMISSION"), 0.00003),
        slippage=_as_float("COST_SLIPPAGE", env.get("COST_SLIPPAGE"), 0.00005),
        swap_long=_as_float("COST_SWAP_LONG", env.get("COST_SWAP_LONG"), -0.00004),
        swap_short=_as_float("COST_SWAP_SHORT", env.get("COST_SWAP_SHORT"), -0.00002),
    )

    broker = BrokerConfig(
        symbol=symbol_raw,
        timeframe=tf_raw,
        magic=_as_int("MT5_MAGIC", env.get("MT5_MAGIC"), 234000),
        deviation=_as_int("MT5_DEVIATION", env.get("MT5_DEVIATION"), 20),
        mt5_login=env.get("MT5_LOGIN"),
        mt5_password=env.get("MT5_PASSWORD"),
        mt5_server=env.get("MT5_SERVER"),
        mt5_path=env.get("MT5_PATH"),
        allow_short=allow_short,
        sl_atr_mult=_as_float("SL_ATR_MULT", env.get("SL_ATR_MULT"), 2.0),
        tp_atr_mult=_as_float("TP_ATR_MULT", env.get("TP_ATR_MULT"), 3.0),
        max_requote_retries=_as_int("MAX_REQUOTE_RETRIES", env.get("MAX_REQUOTE_RETRIES"), 3),
        order_retry_delay_sec=_as_float("ORDER_RETRY_DELAY_SEC", env.get("ORDER_RETRY_DELAY_SEC"), 1.0),
    )

    paths = PathConfig(
        data_dir=Path(env.get("DATA_DIR", str(REPO_ROOT / "data"))),
        model_path=Path(env.get("MODEL_PATH", str(REPO_ROOT / "train" / "ppo_xauusd_latest.zip"))),
        feature_contract_path=Path(env.get("FEATURE_CONTRACT_PATH", str(REPO_ROOT / "train" / "feature_contract.json"))),
        risk_state_db=Path(env.get("RISK_STATE_DB", str(REPO_ROOT / "state" / "risk_state.db"))),
        local_state_file=Path(env.get("LOCAL_STATE_FILE", str(REPO_ROOT / "state" / "bot_state.json"))),
        log_dir=Path(env.get("LOG_DIR", str(REPO_ROOT / "logs"))),
        kill_switch_path=Path(env.get("KILL_SWITCH_PATH", str(REPO_ROOT / "KILL_SWITCH"))),
        backtest_results_dir=Path(env.get("BACKTEST_RESULTS_DIR", str(REPO_ROOT / "research" / "backtest_results"))),
    )

    feature = FeatureConfig(
        window=_as_int("FEATURE_WINDOW", env.get("FEATURE_WINDOW"), 64),
        macro_shift=_as_int("MACRO_SHIFT", env.get("MACRO_SHIFT"), 1),
        drop_warmup=_as_int("FEATURE_DROP_WARMUP", env.get("FEATURE_DROP_WARMUP"), 200),
        normalize=_as_bool("FEATURE_NORMALIZE", env.get("FEATURE_NORMALIZE"), True),
    )

    cfg = AppConfig(
        trading_mode=mode,
        symbol_cfg=symbol_raw,
        timeframe=tf_raw,
        feature=feature,
        risk=risk,
        cost=cost,
        broker=broker,
        paths=paths,
        log_level=env.get("LOG_LEVEL", "INFO").upper(),
        raw=raw,
    )
    return cfg


# --------------------------------------------------------------------------- #
# TRADING_MODE gate
# --------------------------------------------------------------------------- #
class GateResult:
    """Result of a live-mode gate check."""

    def __init__(self, passed: bool, failures: Optional[List[str]] = None):
        self.passed = passed
        self.failures = failures or []

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"GateResult(passed={self.passed}, failures={self.failures})"


def check_live_gates(
    cfg: AppConfig,
    feature_contract_present: bool,
    model_present: bool,
    risk_state_loadable: bool,
    reconciliation_passed: bool,
) -> GateResult:
    """
    Verify every gate required before live trading may start.

    Args:
        cfg: validated app config.
        feature_contract_present: feature_contract.json exists and loads.
        model_present: the trained model checkpoint exists.
        risk_state_loadable: the SQLite risk state could be opened/reloaded.
        reconciliation_passed: broker positions match local persisted state.

    Returns:
        GateResult — passed only if every gate passes.
    """
    failures: List[str] = []

    if cfg.trading_mode is TradingMode.LIVE:
        checks = [
            ("feature contract present", feature_contract_present),
            ("model present", model_present),
            ("risk state loadable", risk_state_loadable),
            ("reconciliation passed", reconciliation_passed),
        ]
        for name, ok in checks:
            if not ok:
                failures.append(name)

        if cfg.broker.mt5_login is None:
            failures.append("MT5_LOGIN not configured")
        if cfg.broker.mt5_password is None:
            failures.append("MT5_PASSWORD not configured")
        if cfg.broker.mt5_server is None:
            failures.append("MT5_SERVER not configured")

    return GateResult(passed=not failures, failures=failures)


def ensure_trading_allowed(cfg: AppConfig, gates: Optional[GateResult] = None) -> None:
    """
    Refuse to run live trading when gates do not pass.

    In demo mode trading is always permitted (no real money is at stake).
    In live mode a GateResult with failures raises RuntimeError listing them.
    """
    if cfg.trading_mode is TradingMode.LIVE:
        if gates is None:
            raise RuntimeError(
                "TRADING_MODE=live requires explicit gate verification; "
                "call check_live_gates() first and pass the result."
            )
        if not gates.passed:
            raise RuntimeError(
                "TRADING_MODE=live REFUSED TO START. Failed gates: "
                + ", ".join(gates.failures)
                + ". Fix the failing gates or switch TRADING_MODE=demo."
            )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover - manual verification entry
    cfg = load_config()
    print("=" * 60)
    print("CONFIG LOADED (validated)")
    print("=" * 60)
    print(f"TRADING_MODE      : {cfg.trading_mode.value}")
    print(f"SYMBOL / TIMEFRAME: {cfg.symbol_cfg} / {cfg.timeframe}")
    print(f"ALLOW_SHORT       : {cfg.broker.allow_short}")
    print(f"Risk             : daily_loss={cfg.risk.max_daily_loss:.2%} "
          f"dd={cfg.risk.max_drawdown:.2%} risk/trade={cfg.risk.risk_per_trade:.2%} "
          f"max_pos={cfg.risk.max_position:.2%} corr_guard={cfg.risk.correlation_guard_enabled}")
    print(f"Cost             : spread={cfg.cost.spread:.5f} comm={cfg.cost.commission:.5f} "
          f"slip={cfg.cost.slippage:.5f}")
    print(f"Broker           : magic={cfg.broker.magic} deviation={cfg.broker.deviation} "
          f"sl_atr={cfg.broker.sl_atr_mult} tp_atr={cfg.broker.tp_atr_mult}")
    print(f"Model            : {cfg.paths.model_path}")
    print(f"Contract         : {cfg.paths.feature_contract_path}")
    print(f"Risk state db    : {cfg.paths.risk_state_db}")
    print(f"Kill switch      : {cfg.paths.kill_switch_path}")

    # Demo mode: trading allowed without gates.
    gates = check_live_gates(
        cfg,
        feature_contract_present=cfg.paths.feature_contract_path.exists(),
        model_present=cfg.paths.model_path.exists(),
        risk_state_loadable=True,
        reconciliation_passed=True,
    )
    ensure_trading_allowed(cfg, gates)
    print("\nTRADING_MODE gate: OK (trading permitted)")
