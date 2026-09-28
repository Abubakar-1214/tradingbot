"""Config-layer pytest: validated defaults, env parsing, TRADING_MODE live gate."""
from __future__ import annotations

import pytest

from core.config import (
    AppConfig,
    BrokerConfig,
    GateResult,
    RiskConfig,
    TradingMode,
    check_live_gates,
    ensure_trading_allowed,
    load_config,
)


def test_defaults_demo_mode() -> None:
    cfg = load_config(_env={})
    assert cfg.trading_mode is TradingMode.DEMO
    assert cfg.symbol_cfg == "XAUUSD"
    assert cfg.timeframe == "H1"
    assert cfg.broker.allow_short is False
    assert cfg.broker.sl_atr_mult == 2.0
    assert cfg.broker.tp_atr_mult == 3.0
    assert cfg.risk.max_daily_loss == 0.05
    assert cfg.risk.max_position == 0.10


def test_env_parsing() -> None:
    cfg = load_config(_env={
        "TRADING_MODE": "live",
        "SYMBOL": "xauusd",
        "TIMEFRAME": "D1",
        "ALLOW_SHORT": "true",
        "MAX_DAILY_LOSS": "0.03",
        "MAX_POSITION": "0.08",
        "SL_ATR_MULT": "2.5",
        "TP_ATR_MULT": "4.0",
        "MT5_LOGIN": "12345",
        "MT5_PASSWORD": "pw",
        "MT5_SERVER": "broker",
    })
    assert cfg.trading_mode is TradingMode.LIVE
    assert cfg.symbol_cfg == "XAUUSD"          # uppercased
    assert cfg.timeframe == "D1"
    assert cfg.broker.allow_short is True
    assert cfg.risk.max_daily_loss == 0.03
    assert cfg.risk.max_position == 0.08
    assert cfg.broker.sl_atr_mult == 2.5
    assert cfg.broker.tp_atr_mult == 4.0
    assert cfg.broker.mt5_login == "12345"


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        load_config(_env={"TRADING_MODE": "paper"})


def test_invalid_symbol_raises() -> None:
    with pytest.raises(ValueError):
        load_config(_env={"SYMBOL": "BTCUSD"})


def test_invalid_timeframe_raises() -> None:
    with pytest.raises(ValueError):
        load_config(_env={"TIMEFRAME": "M2"})


def test_invalid_float_raises() -> None:
    with pytest.raises(ValueError):
        load_config(_env={"MAX_DAILY_LOSS": "not-a-number"})


def test_risk_config_validation() -> None:
    with pytest.raises(ValueError):
        RiskConfig(risk_per_trade=0.06, max_daily_loss=0.05)  # risk > daily cap
    with pytest.raises(ValueError):
        RiskConfig(max_position=1.5)
    with pytest.raises(ValueError):
        RiskConfig(max_consecutive_losses=0)


def test_live_gates_demo_mode_passes() -> None:
    cfg = load_config(_env={})  # demo
    gates = check_live_gates(cfg, feature_contract_present=False,
                             model_present=False, risk_state_loadable=True,
                             reconciliation_passed=True)
    assert gates.passed
    ensure_trading_allowed(cfg, gates)  # no raise


def test_live_gates_live_requires_all() -> None:
    cfg = load_config(_env={"TRADING_MODE": "live"})
    gates = check_live_gates(cfg, feature_contract_present=False,
                             model_present=True, risk_state_loadable=True,
                             reconciliation_passed=True)
    assert not gates.passed
    assert "feature contract present" in gates.failures
    assert "MT5_LOGIN not configured" in gates.failures
    assert "MT5_SERVER not configured" in gates.failures
    with pytest.raises(RuntimeError, match="REFUSED TO START"):
        ensure_trading_allowed(cfg, gates)


def test_live_gates_pass_with_everything() -> None:
    cfg = load_config(_env={
        "TRADING_MODE": "live", "MT5_LOGIN": "1", "MT5_PASSWORD": "p",
        "MT5_SERVER": "s",
    })
    gates = check_live_gates(cfg, feature_contract_present=True,
                             model_present=True, risk_state_loadable=True,
                             reconciliation_passed=True)
    assert gates.passed
    ensure_trading_allowed(cfg, gates)


def test_ensure_live_without_gates_raises() -> None:
    cfg = load_config(_env={"TRADING_MODE": "live"})
    with pytest.raises(RuntimeError, match="requires explicit gate verification"):
        ensure_trading_allowed(cfg, None)


def test_gate_result_bool() -> None:
    assert bool(GateResult(passed=True))
    assert not bool(GateResult(passed=False, failures=["x"]))


def test_broker_config_defaults() -> None:
    b = BrokerConfig()
    assert b.magic == 234000
    assert b.deviation == 20
    assert b.max_requote_retries == 3
    assert b.order_retry_delay_sec == 1.0
