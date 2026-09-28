"""
scripts/verify_risk_integration.py — RiskSupervisor wiring gate for TradeExecutor.

Proves (P0-1 / P0-2 live-system fixes):
  1. EVERY order path in live/trade_executor.py consults RiskSupervisor.check_trade
     BEFORE touching the broker.  A probe counts invocations and asserts each of
     entry / scale / TP-adjust / close submits exactly one check.
  2. Six breaker scenarios produce the expected allow/block:
       * normal entry            -> APPROVED, order fills with a ticket
       * 5 consecutive losses    -> TOO_MANY_LOSSES (entry blocked)
       * daily pnl <= -5%        -> CIRCUIT_BREAKER + 24h halt (blocked)
       * drawdown > 15%          -> MAX_DRAWDOWN (blocked)
       * re-trade within 300s    -> COOLDOWN (blocked)
       * correlation guard ON + dxy_momentum > 0.01 -> CORRELATION_GUARD blocks long
  3. 100% of filled MockBroker orders carry non-null SL and TP (ATR-derived).
  4. Determinism: two identical runs produce identical allow/block results.

Sizing setup (honest XAUUSD lot math):
  * XAUUSD 0.01 lot = 1 oz * 100 = $2000 notional at $2000/oz.  With a $10k
    account and a 10% notional cap ($1000) no order can ever fill, so the gate
    trades on a $100k account where 0.01 lot = 2% exposure.
  * The default ATRPositionSizer (2% risk, 2xATR stop on ~$4 ATR) always hits
    the 10% cap, which consumes the entire aggregate cap on the FIRST entry and
    makes scaling untestable.  The gate therefore injects a 3%-target sizer.
  * RiskSupervisor's daily_start_equity defaults to $10k (DEFAULTS) — the gate
    resets the equity base to $100k so ratio breakers behave as documented.
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from core.config import AppConfig, RiskConfig
from live.mock_broker import MockBroker
from live.trade_executor import TradeExecutor
from models.position_sizing import ATRPositionSizer
from models.risk_supervisor import RiskSupervisor

PASS = 0
FAIL = 0

EQUITY = 100_000.0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


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


class ProbeExecutor(TradeExecutor):
    """TradeExecutor with a check_trade invocation counter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_calls: int = 0

    def _wire_probe(self) -> None:
        original_check = self.risk.check_trade

        def counting_check(signal, size, state, market_data):
            self.check_calls += 1
            return original_check(signal, size, state, market_data)

        self.risk.check_trade = counting_check  # type: ignore[method-assign]


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
                   probe: bool = True, equity: float = EQUITY) -> ProbeExecutor:
    broker = MockBroker(symbol="XAUUSD", balance=equity, seed=42)
    broker.connect()
    broker.set_mid(2000.0)
    risk = RiskSupervisor(config=cfg.risk, db_path=tmpdir / "risk.db",
                          now_fn=clock)
    # Ratio breakers use the equity base as the denominator — set it to the
    # gate's trading equity instead of the DEFAULTS $10k.
    risk.initial_equity = equity
    risk.current_equity = equity
    risk.daily_start_equity = equity
    risk.peak_equity = equity
    ex = ProbeExecutor(broker, risk, cfg, sizer=SmallSizer(),
                       local_state_file=tmpdir / "bot_state.json")
    if probe:
        ex._wire_probe()
    return ex


def calm_md(ex: TradeExecutor, df: pd.DataFrame) -> Dict[str, float]:
    """Market-data dict with volatility below the 3.0 z-score threshold."""
    md = ex.build_market_data(df, ex.broker.get_tick("XAUUSD"))
    md["volatility"] = 0.5
    return md


def scenario_normal(tmpdir: Path, clock: FakeClock) -> Tuple[bool, str, Any]:
    ex = build_executor(clock, make_config(), tmpdir)
    df = sample_df()
    return ex.execute_entry(1, equity=EQUITY, df=df, market_data=calm_md(ex, df),
                            bar_time="2024-01-08T10:00:00")


def scenario_consecutive_losses(tmpdir: Path, clock: FakeClock) -> Tuple[bool, str, Any]:
    ex = build_executor(clock, make_config(), tmpdir)
    df = sample_df()
    # 5 consecutive losses (each 0.3% of equity; total -1.5% < 5% daily-loss cap)
    for i in range(5):
        ex.risk.update_state(pnl=-300.0, equity=EQUITY - 300.0 * (i + 1),
                             is_win=False)
        clock.advance(minutes=10)
    return ex.execute_entry(1, equity=ex.risk.current_equity, df=df,
                            market_data=calm_md(ex, df),
                            bar_time="2024-01-08T11:00:00")


def scenario_daily_loss(tmpdir: Path, clock: FakeClock) -> Tuple[bool, str, Any]:
    ex = build_executor(clock, make_config(), tmpdir)
    # -7% day -> daily-loss breaker trips + 24h halt
    ex.risk.update_state(pnl=-7_000.0, equity=93_000.0, is_win=False)
    return ex.execute_entry(1, equity=93_000.0, df=sample_df(),
                            market_data=calm_md(ex, sample_df()),
                            bar_time="2024-01-08T12:00:00")


def scenario_drawdown(tmpdir: Path, clock: FakeClock) -> Tuple[bool, str, Any]:
    ex = build_executor(clock, make_config(), tmpdir)
    # Raise the peak with a win, then let UNREALIZED equity fall 20% below peak
    # (passed via the state['equity'] path; realized daily P&L stays positive)
    ex.risk.update_state(pnl=+2_500.0, equity=102_500.0, is_win=True)
    return ex.execute_entry(1, equity=82_000.0, df=sample_df(),
                            market_data=calm_md(ex, sample_df()),
                            bar_time="2024-01-08T13:00:00")


def scenario_cooldown(tmpdir: Path, clock: FakeClock) -> Tuple[bool, str, Any]:
    ex = build_executor(clock, make_config(), tmpdir)
    df = sample_df()
    # First entry approved; a realized close sets last_trade_time via update_state
    ok0, reason0, res0 = ex.execute_entry(1, equity=EQUITY, df=df,
                                          market_data=calm_md(ex, df),
                                          bar_time="2024-01-08T14:00:00")
    ex.risk.update_state(pnl=+150.0, equity=EQUITY, is_win=True)
    # Immediately re-enter (0s elapsed < 300s cooldown) -> COOLDOWN
    return ex.execute_entry(1, equity=EQUITY, df=df, market_data=calm_md(ex, df),
                            bar_time="2024-01-08T14:01:00")


def scenario_correlation(tmpdir: Path, clock: FakeClock) -> Tuple[bool, str, Any]:
    ex = build_executor(clock, make_config(correlation_guard_enabled=True,
                                           correlation_block_long_on_up=0.01),
                        tmpdir)
    md = calm_md(ex, sample_df())
    md["dxy_momentum"] = 0.02   # strong up-momentum blocks LONG
    return ex.execute_entry(1, equity=EQUITY, df=sample_df(), market_data=md,
                            bar_time="2024-01-08T15:00:00")


def main() -> int:
    print("=" * 70)
    print("verify_risk_integration.py — TradeExecutor RiskSupervisor gate")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # --- [1] every order path calls check_trade ------------------------ #
        print("\n[1] check_trade invoked on EVERY order path")
        clock = FakeClock(datetime(2024, 1, 8, 9, 0, tzinfo=timezone.utc))
        ex = build_executor(clock, make_config(), root / "t1")
        df = sample_df()

        ok, reason, res = ex.execute_entry(1, equity=EQUITY, df=df,
                                           market_data=calm_md(ex, df),
                                           bar_time="2024-01-08T10:00:00")
        calls_after_entry = ex.check_calls
        check("entry approved", ok, f"{reason}")
        check("entry called check_trade exactly once", calls_after_entry == 1,
              f"calls={calls_after_entry}")

        clock.advance(minutes=10)
        ok2, reason2, res2 = ex.execute_scale(1, equity=EQUITY, df=df,
                                              market_data=calm_md(ex, df),
                                              bar_time="2024-01-08T10:10:00")
        calls_after_scale = ex.check_calls
        check("scale approved", ok2, f"{reason2}")
        check("scale called check_trade (cumulative 2)", calls_after_scale == 2,
              f"calls={calls_after_scale}")

        ticket = list(ex._positions.keys())[0]
        ok3, reason3, res3 = ex.modify_sl_tp(ticket, sl=1980.0, tp=2040.0)
        calls_after_modify = ex.check_calls
        check("TP-adjust approved", ok3, f"{reason3}")
        check("modify called check_trade (cumulative 3)", calls_after_modify == 3,
              f"calls={calls_after_modify}")

        ok4, reason4, res4 = ex.execute_close(ticket)
        calls_after_close = ex.check_calls
        check("close approved", ok4, f"{reason4}")
        check("close called check_trade (cumulative 4)", calls_after_close == 4,
              f"calls={calls_after_close}")

        # --- [2] six breaker scenarios ------------------------------------- #
        print("\n[2] breaker scenarios")
        clock2 = FakeClock(datetime(2024, 1, 8, 9, 0, tzinfo=timezone.utc))
        ok_n, reason_n, res_n = scenario_normal(root / "s_normal", clock2)
        check("normal entry APPROVED", ok_n, f"{reason_n}")
        check("normal entry filled with ticket", res_n is not None and res_n.ok
              and res_n.ticket is not None, f"{res_n}")

        clock3 = FakeClock(datetime(2024, 1, 8, 9, 0, tzinfo=timezone.utc))
        ok_cl, reason_cl, _ = scenario_consecutive_losses(root / "s_losses", clock3)
        check("5 consecutive losses -> TOO_MANY_LOSSES", not ok_cl
              and "TOO_MANY_LOSSES" in reason_cl, f"{reason_cl}")

        clock4 = FakeClock(datetime(2024, 1, 8, 9, 0, tzinfo=timezone.utc))
        ok_dl, reason_dl, _ = scenario_daily_loss(root / "s_dailyloss", clock4)
        check("daily loss -7% -> CIRCUIT_BREAKER", not ok_dl
              and "CIRCUIT_BREAKER" in reason_dl, f"{reason_dl}")

        clock5 = FakeClock(datetime(2024, 1, 8, 9, 0, tzinfo=timezone.utc))
        ok_dd, reason_dd, _ = scenario_drawdown(root / "s_drawdown", clock5)
        check("drawdown 20% from peak -> MAX_DRAWDOWN", not ok_dd
              and "MAX_DRAWDOWN" in reason_dd, f"{reason_dd}")

        clock6 = FakeClock(datetime(2024, 1, 8, 9, 0, tzinfo=timezone.utc))
        ok_cd, reason_cd, _ = scenario_cooldown(root / "s_cooldown", clock6)
        check("re-trade within 300s -> COOLDOWN", not ok_cd
              and "COOLDOWN" in reason_cd, f"{reason_cd}")

        clock7 = FakeClock(datetime(2024, 1, 8, 9, 0, tzinfo=timezone.utc))
        ok_cr, reason_cr, _ = scenario_correlation(root / "s_corr", clock7)
        check("correlation guard blocks long on DXY up", not ok_cr
              and "CORRELATION_GUARD" in reason_cr, f"{reason_cr}")

        # --- [3] every filled order carries SL+TP --------------------------- #
        print("\n[3] SL/TP on every filled MockBroker order")
        clock8 = FakeClock(datetime(2024, 1, 8, 9, 0, tzinfo=timezone.utc))
        ex3 = build_executor(clock8, make_config(), root / "s_sltp")
        df3 = sample_df()
        ok_a, _, ra = ex3.execute_entry(1, equity=EQUITY, df=df3,
                                        market_data=calm_md(ex3, df3),
                                        bar_time="2024-01-08T10:00:00")
        clock8.advance(minutes=10)
        ok_b, _, rb = ex3.execute_entry(1, equity=EQUITY, df=df3,
                                        market_data=calm_md(ex3, df3),
                                        bar_time="2024-01-08T10:10:00")
        check("two entries filled", ok_a and ok_b and ra.ok and rb.ok,
              f"{ra} | {rb}")
        positions = ex3.broker.get_positions()
        check("all filled entries carry SL", all(p.sl is not None for p in positions))
        check("all filled entries carry TP", all(p.tp is not None for p in positions))

        clock8.advance(minutes=10)
        ok_s, _, rs = ex3.execute_scale(1, equity=EQUITY, df=df3,
                                        market_data=calm_md(ex3, df3),
                                        bar_time="2024-01-08T10:20:00")
        check("scale filled", ok_s and rs.ok, f"{rs}")
        positions2 = ex3.broker.get_positions()
        check("scaled fill carries SL", all(p.sl is not None for p in positions2))
        check("scaled fill carries TP", all(p.tp is not None for p in positions2))

        # --- [4] determinism: identical second run -------------------------- #
        print("\n[4] determinism (identical second run)")
        clock_d1 = FakeClock(datetime(2024, 1, 8, 9, 0, tzinfo=timezone.utc))
        ex_d1 = build_executor(clock_d1, make_config(), root / "det1")
        ok_d1, reason_d1, res_d1 = ex_d1.execute_entry(
            1, equity=EQUITY, df=sample_df(), market_data=calm_md(ex_d1, sample_df()),
            bar_time="2024-01-08T10:00:00")
        clock_d2 = FakeClock(datetime(2024, 1, 8, 9, 0, tzinfo=timezone.utc))
        ex_d2 = build_executor(clock_d2, make_config(), root / "det2")
        ok_d2, reason_d2, res_d2 = ex_d2.execute_entry(
            1, equity=EQUITY, df=sample_df(), market_data=calm_md(ex_d2, sample_df()),
            bar_time="2024-01-08T10:00:00")
        check("identical approval across runs", ok_d1 == ok_d2,
              f"{reason_d1} vs {reason_d2}")
        check("identical fill price across runs",
              res_d1.fill_price == res_d2.fill_price,
              f"{res_d1.fill_price} vs {res_d2.fill_price}")

    print("\n" + "=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
