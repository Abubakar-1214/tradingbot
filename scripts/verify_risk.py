"""
scripts/verify_risk.py — acceptance gate for the RiskSupervisor rewrite.

Proves (P0-2 fixes):
  1. Normal, real-size trade (explicit fraction) is APPROVED.
  2. Oversized position is REJECTED (POSITION_TOO_LARGE) — requested_size is
     the fraction, NOT abs(action).
  3. Daily-loss circuit breaker triggers halt after losses exceed the cap and
     SURVIVES A RESTART (fresh process, same SQLite DB) — restart cannot
     bypass breakers.
  4. Halt state survives restart.
  5. Consecutive-loss breaker trips at max_consecutive_losses.
  6. Correlation guard is OFF by default, and when enabled blocks long on
     strong up-momentum of the configured asset.
  7. Max trades / cooldown / spread / volatility / event-window breakers work.
  8. update_state() persists trade history + counters.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models.risk_supervisor import RiskSupervisor

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


def main() -> int:
    print("=" * 66)
    print("verify_risk.py - RiskSupervisor rewrite acceptance gate")
    print("=" * 66)

    # --- [1] normal trade approved with explicit real size ----------------- #
    print("\n[1] normal trade with real position fraction approved")
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "risk.db"
        sup = RiskSupervisor(db_path=db)
        state = {"position": 0, "equity": 10_000.0}
        md = {"volatility": 1.5, "spread": 0.0003, "is_high_impact_event": False,
              "is_event_window": False, "dxy_momentum": -0.005}
        ok, reason = sup.check_trade(1, requested_size=0.05, state=state, market_data=md)
        check("long 5% fraction approved", ok, reason)
        check("reason is APPROVED", reason == "APPROVED", reason)

        ok2, reason2 = sup.check_trade(2, requested_size=0.03, state=state, market_data=md)
        check("short 3% fraction approved (allow_short caller)", ok2, reason2)

    # --- [2] oversized rejected (explicit fraction, not abs(action)) -------- #
    print("\n[2] oversized position rejected")
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "risk.db"
        sup = RiskSupervisor(db_path=db)
        state = {"position": 0, "equity": 10_000.0}
        md = {"volatility": 1.5, "spread": 0.0003}
        # action=1 (abs would be 1.0 = 100%) but requested_size 0.5 must be rejected
        ok, reason = sup.check_trade(1, requested_size=0.50, state=state, market_data=md)
        check("50% fraction rejected", not ok, reason)
        check("rejection reason is POSITION_TOO_LARGE", "POSITION_TOO_LARGE" in reason, reason)

        ok, reason = sup.check_trade(1, requested_size=0.09, state=state, market_data=md)
        check("9% fraction accepted (<= 10% max)", ok, reason)

        ok, reason = sup.check_trade(1, requested_size=1.5, state=state, market_data=md)
        check("fraction >1 rejected INVALID_SIZE", not ok and "INVALID_SIZE" in reason, reason)

    # --- [3] daily-loss breaker + SURVIVES RESTART -------------------------- #
    print("\n[3] daily-loss breaker persists across restart")
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "risk.db"
        sup = RiskSupervisor(db_path=db)
        md = {"volatility": 1.5, "spread": 0.0003}
        # lose 7% (daily cap is 5%)
        sup.update_state(pnl=-700.0, equity=9_300.0, is_win=False)
        ok, reason = sup.check_trade(1, requested_size=0.05, state={"equity": 9_300.0}, market_data=md)
        check("daily loss 7% trips breaker", not ok, reason)
        check("breaker reason", "CIRCUIT_BREAKER" in reason, reason)

        # NEW INSTANCE (simulates process restart) on same DB
        sup2 = RiskSupervisor(db_path=db)
        st = sup2.get_statistics()
        check("restart reloads daily_pnl", abs(st["daily_pnl"] - (-700.0)) < 1e-9,
              f"got {st['daily_pnl']}")
        ok, reason = sup2.check_trade(1, requested_size=0.05, state={"equity": 9_300.0}, market_data=md)
        check("restart still blocks trading (breaker survived)", not ok, reason)

    # --- [4] halt survives restart ----------------------------------------- #
    print("\n[4] halt state survives restart")
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "risk.db"
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        clock = [now]

        def fake_now():
            return clock[0]

        sup = RiskSupervisor(db_path=db, now_fn=fake_now)
        sup.emergency_shutdown()
        check("emergency halt set", sup.halt_remaining_seconds() > 86400)

        # fresh process: halt must be restored
        sup2 = RiskSupervisor(db_path=db, now_fn=fake_now)
        ok, reason = sup2.check_trade(1, requested_size=0.05,
                                      state={"equity": 10_000.0}, market_data={"volatility": 1.5})
        check("restart restores halt (trading blocked)", not ok, reason)
        check("reason mentions HALTED", "HALTED" in reason, reason)

        # advance clock past halt -> halt cleared on daily reset path
        clock[0] = now + timedelta(days=366)
        sup3 = RiskSupervisor(db_path=db, now_fn=fake_now)
        ok, reason = sup3.check_trade(1, requested_size=0.05,
                                      state={"equity": 10_000.0}, market_data={"volatility": 1.5})
        check("halt cleared after expiry via daily reset", ok, reason)

    # --- [5] consecutive-loss breaker -------------------------------------- #
    print("\n[5] consecutive-loss breaker")
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "risk.db"
        clock = [datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)]

        def fake_now():
            return clock[0]

        sup = RiskSupervisor(db_path=db, now_fn=fake_now)
        md = {"volatility": 1.5, "spread": 0.0003}
        for i in range(4):
            clock[0] = clock[0] + timedelta(minutes=10)
            sup.update_state(pnl=-50.0, equity=9_950.0 - 50.0 * i, is_win=False)
        check("4 consecutive losses counted", sup.consecutive_losses == 4,
              f"got {sup.consecutive_losses}")
        # Advance past the trade cooldown (default 300s) so this check isolates
        # the consecutive-loss breaker: 4 losses (< max 5) must still allow a trade.
        clock[0] = clock[0] + timedelta(minutes=10)
        ok, reason = sup.check_trade(1, requested_size=0.05,
                                     state={"equity": 9_750.0}, market_data=md)
        check("4 consecutive losses do NOT trip breaker (max 5)",
              ok, reason)  # max_consecutive_losses default 5, we are at 4
        sup.update_state(pnl=-50.0, equity=9_700.0, is_win=False)
        ok, reason = sup.check_trade(1, requested_size=0.05,
                                     state={"equity": 9_700.0}, market_data=md)
        check("5 consecutive losses block", not ok, reason)
        check("reason TOO_MANY_LOSSES", "TOO_MANY_LOSSES" in reason, reason)
        # a win resets the streak
        clock[0] = clock[0] + timedelta(minutes=10)
        sup.update_state(pnl=100.0, equity=9_800.0, is_win=True)
        check("win resets consecutive losses", sup.consecutive_losses == 0,
              f"got {sup.consecutive_losses}")

    # --- [6] correlation guard default OFF, then ON blocks ----------------- #
    print("\n[6] correlation guard config-driven (default OFF)")
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "risk.db"
        sup = RiskSupervisor(db_path=db)
        md = {"volatility": 1.5, "spread": 0.0003, "dxy_momentum": 0.05}
        ok, reason = sup.check_trade(1, requested_size=0.05,
                                     state={"equity": 10_000.0}, market_data=md)
        check("long approved with strong DXY momentum when guard OFF", ok, reason)

        cfg = {
            "max_daily_loss": 0.05, "max_position": 0.10, "max_drawdown": 0.15,
            "correlation_guard_enabled": True, "correlation_asset": "DXY",
            "correlation_block_long_on_up": 0.01,
        }
        sup2 = RiskSupervisor(config=cfg, db_path=Path(td) / "risk2.db")
        ok, reason = sup2.check_trade(1, requested_size=0.05,
                                      state={"equity": 10_000.0}, market_data=md)
        check("long blocked by correlation guard when ON", not ok, reason)
        check("reason CORRELATION_GUARD", "CORRELATION_GUARD" in reason, reason)

        md["dxy_momentum"] = -0.05
        ok, reason = sup2.check_trade(1, requested_size=0.05,
                                      state={"equity": 10_000.0}, market_data=md)
        check("long allowed when DXY falling", ok, reason)

    # --- [7] other breakers ------------------------------------------------- #
    print("\n[7] max trades, cooldown, spread, volatility, event window")
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "risk.db"
        clock = [datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)]

        def fake_now():
            return clock[0]

        cfg = {"max_trades_per_day": 2, "min_trade_interval_sec": 300.0,
               "max_spread": 0.0005, "vol_threshold": 3.0}
        sup = RiskSupervisor(config=cfg, db_path=db, now_fn=fake_now)
        md = {"volatility": 1.5, "spread": 0.0003}
        ok, _ = sup.check_trade(1, requested_size=0.05, state={"equity": 10_000.0}, market_data=md)
        sup.update_state(pnl=10.0, equity=10_010.0, is_win=True)
        clock[0] += timedelta(minutes=10)
        ok2, _ = sup.check_trade(1, requested_size=0.05, state={"equity": 10_010.0}, market_data=md)
        sup.update_state(pnl=10.0, equity=10_020.0, is_win=True)
        clock[0] += timedelta(minutes=10)
        ok3, reason = sup.check_trade(1, requested_size=0.05, state={"equity": 10_020.0}, market_data=md)
        check("3rd trade blocked by MAX_TRADES (cap 2)", not ok3 and "MAX_TRADES" in reason, reason)

        # cooldown
        sup2 = RiskSupervisor(db_path=Path(td) / "risk3.db", now_fn=fake_now)
        ok, _ = sup2.check_trade(1, requested_size=0.05, state={"equity": 10_000.0}, market_data=md)
        sup2.update_state(pnl=10.0, equity=10_010.0, is_win=True)  # sets last_trade_time
        ok2, reason = sup2.check_trade(1, requested_size=0.05, state={"equity": 10_010.0}, market_data=md)
        check("immediate retrade blocked by COOLDOWN", not ok2 and "COOLDOWN" in reason, reason)

        # spread
        sup3 = RiskSupervisor(db_path=Path(td) / "risk4.db", now_fn=fake_now)
        ok, reason = sup3.check_trade(1, requested_size=0.05,
                                      state={"equity": 10_000.0},
                                      market_data={"volatility": 1.5, "spread": 0.003})
        check("wide spread blocked", not ok and "SPREAD_TOO_WIDE" in reason, reason)

        # volatility
        sup4 = RiskSupervisor(db_path=Path(td) / "risk5.db", now_fn=fake_now)
        ok, reason = sup4.check_trade(1, requested_size=0.05,
                                      state={"equity": 10_000.0, "position": 0},
                                      market_data={"volatility": 9.0, "spread": 0.0003})
        check("high volatility blocks new entry", not ok and "HIGH_VOLATILITY" in reason, reason)

        # event window halves allowed size
        sup5 = RiskSupervisor(db_path=Path(td) / "risk6.db", now_fn=fake_now)
        ok, reason = sup5.check_trade(1, requested_size=0.09,
                                      state={"equity": 10_000.0},
                                      market_data={"volatility": 1.5, "spread": 0.0003,
                                                   "is_event_window": True})
        check("event window rejects oversize", not ok and "EVENT_RISK" in reason, reason)

    print("\n" + "=" * 66)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
