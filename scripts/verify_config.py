"""
scripts/verify_config.py — verification for the config layer (subtask 2).

Proves:
  1. demo mode (default) loads and trading is permitted without gates.
  2. live mode REFUSES to start when gates fail (RuntimeError with reasons).
  3. live mode starts when every gate passes and MT5 credentials are present.
  4. missing MT5 credentials are listed as gate failures.
  5. invalid TRADING_MODE value raises ValueError at load time.
  6. RiskConfig rejects inconsistent risk settings.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import (
    RiskConfig,
    TradingMode,
    check_live_gates,
    ensure_trading_allowed,
    load_config,
)

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


def demo_env() -> dict:
    return {"TRADING_MODE": "demo", "SYMBOL": "XAUUSD", "TIMEFRAME": "H1"}


def live_env() -> dict:
    return {
        "TRADING_MODE": "live",
        "SYMBOL": "XAUUSD",
        "TIMEFRAME": "H1",
        "MT5_LOGIN": "12345",
        "MT5_PASSWORD": "secret",
        "MT5_SERVER": "ICMarkets-Demo",
    }


def has_failure(gates, needle: str) -> bool:
    """True if any failure string contains `needle` as a substring."""
    return any(needle in f for f in gates.failures)


def main() -> int:
    print("=" * 64)
    print("verify_config.py — config layer + TRADING_MODE gate")
    print("=" * 64)

    # 1. Demo mode default.
    print("\n[1] demo mode loads; trading permitted without gates")
    cfg = load_config(_env=demo_env())
    check("TRADING_MODE parsed as demo", cfg.trading_mode is TradingMode.DEMO)
    check("default symbol XAUUSD", cfg.symbol_cfg == "XAUUSD")
    check("default timeframe H1", cfg.timeframe == "H1")
    check("ALLOW_SHORT default False", cfg.broker.allow_short is False)
    check("risk_per_trade <= max_daily_loss", cfg.risk.risk_per_trade <= cfg.risk.max_daily_loss)
    ensure_trading_allowed(cfg, gates=None)
    check("demo permitted without gates", True)
    bad = check_live_gates(cfg, False, False, False, False)
    ensure_trading_allowed(cfg, gates=bad)
    check("demo ignores failing gates", True)

    # 2. live mode refuses without gates (MT5 creds present -> not listed).
    print("\n[2] live mode refuses to start without gates")
    cfg_live = load_config(_env=live_env())
    g = check_live_gates(cfg_live, False, False, False, False)
    check("live gate result not passed", not g.passed)
    raised = False
    try:
        ensure_trading_allowed(cfg_live, gates=g)
    except RuntimeError as exc:
        raised = True
        msg = str(exc)
        check("failure message lists feature contract", "feature contract" in msg)
        check("failure message lists model", "model present" in msg)
        check("failure message lists risk state", "risk state" in msg)
        check("failure message lists reconciliation", "reconciliation" in msg)
        # live_env DOES provide MT5 credentials -> they must NOT be failures
        check("MT5_LOGIN not listed when credentials present", "MT5_LOGIN" not in msg)
        check("failure message contains REFUSED", "REFUSED" in msg)
    check("live raises RuntimeError on failed gates", raised)
    raised2 = False
    try:
        ensure_trading_allowed(cfg_live, gates=None)
    except RuntimeError:
        raised2 = True
    check("live with gates=None raises (explicit gate required)", raised2)

    # 3. live mode starts when every gate passes.
    print("\n[3] live mode permits when all gates pass")
    cfg_live2 = load_config(_env=live_env())
    g2 = check_live_gates(cfg_live2, True, True, True, True)
    check("all-gates-pass result", g2.passed)
    ensure_trading_allowed(cfg_live2, gates=g2)
    check("live starts with all gates", True)

    # 4. missing MT5 credentials ARE listed as gate failures.
    print("\n[4] missing MT5 credentials fail the gate")
    env_nosrv = live_env()
    env_nosrv.pop("MT5_SERVER")
    env_nosrv.pop("MT5_LOGIN")
    cfg_nosrv = load_config(_env=env_nosrv)
    g3 = check_live_gates(cfg_nosrv, True, True, True, True)
    check("live without MT5_LOGIN/SERVER fails gate", not g3.passed)
    check("MT5_LOGIN listed as failure", has_failure(g3, "MT5_LOGIN"))
    check("MT5_SERVER listed as failure", has_failure(g3, "MT5_SERVER"))

    # 5. invalid TRADING_MODE raises ValueError.
    print("\n[5] invalid TRADING_MODE raises")
    raised3 = False
    try:
        load_config(_env={"TRADING_MODE": "paper"})
    except ValueError:
        raised3 = True
    check("TRADING_MODE=paper raises ValueError", raised3)

    # 6. validated dataclasses reject inconsistent risk config.
    print("\n[6] RiskConfig validation")
    raised4 = False
    try:
        RiskConfig(risk_per_trade=0.10, max_daily_loss=0.05)
    except ValueError:
        raised4 = True
    check("risk_per_trade > max_daily_loss raises", raised4)
    check("risk defaults valid", RiskConfig().risk_per_trade <= RiskConfig().max_daily_loss)

    print("\n" + "=" * 64)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
