from __future__ import annotations

import pytest
from core.adaptive_tier import AccountTier, AdaptiveTierConfig, resolve_tier
from live.broker import PositionInfo
from live.trade_manager import ModifySL, TradeManager


def test_resolve_tier_standard_usd() -> None:
    cfg = AdaptiveTierConfig(enabled=True, is_cent_account=False)

    # $5.00 account -> Micro Scalper
    t1 = resolve_tier(5.0, cfg)
    assert t1.name == "MICRO_SCALPER"
    assert t1.tp_atr_mult == 0.8
    assert t1.sl_atr_mult == 1.2
    assert t1.breakeven_at_r == 0.4
    assert t1.trail_start_r == 0.6
    assert t1.max_bars_in_trade == 20

    # $25.00 account -> Growth Scalper
    t2 = resolve_tier(25.0, cfg)
    assert t2.name == "GROWTH_SCALPER"
    assert t2.tp_atr_mult == 1.8
    assert t2.breakeven_at_r == 0.8

    # $75.00 account -> Expansion Rider
    t3 = resolve_tier(75.0, cfg)
    assert t3.name == "EXPANSION_RIDER"
    assert t3.tp_atr_mult == 3.0
    assert t3.breakeven_at_r == 1.0


def test_resolve_tier_cent_account() -> None:
    cfg = AdaptiveTierConfig(enabled=True, is_cent_account=True)

    # 500 USC = $5.00 USD -> Micro Scalper
    t1 = resolve_tier(500.0, cfg)
    assert t1.name == "MICRO_SCALPER"
    assert t1.tp_atr_mult == 0.8

    # 3000 USC = $30.00 USD -> Growth Scalper
    t2 = resolve_tier(3000.0, cfg)
    assert t2.name == "GROWTH_SCALPER"


def test_trade_manager_micro_tier_fast_breakeven() -> None:
    # Set up trade manager
    from core.config import AppConfig
    app_cfg = AppConfig()
    tm = TradeManager(app_cfg)

    # Simulate Buy position at 4000.0, initial SL at 3996.0 (risk = 4.0)
    pos = PositionInfo(
        ticket=101,
        symbol="XAUUSD",
        side="buy",
        volume=0.01,
        open_price=4000.0,
        sl=3996.0,
        tp=4003.2,
    )

    trade_meta = {
        "initial_sl": 3996.0,
        "entry_atr": 2.0,
        "tier_name": "MICRO_SCALPER",
        "breakeven_at_r": 0.4,       # Micro tier trigger
        "trail_start_r": 0.6,
        "trail_atr_mult": 0.8,
        "max_bars_in_trade": 20,
    }

    # Price moves to 4001.7 (+1.7 points / 4.0 risk = +0.425 R)
    last_bar = {"close": 4001.7}
    actions = tm.manage(
        position=pos,
        entry_atr=2.0,
        bars_held=2,
        last_bar=last_bar,
        atr_now=2.0,
        trade_meta=trade_meta,
    )

    # Should trigger ModifySL to Breakeven (>= 4000.0)
    assert len(actions) == 1
    assert isinstance(actions[0], ModifySL)
    assert actions[0].ticket == 101
    assert actions[0].sl >= 4000.0, f"Expected breakeven SL >= 4000.0, got {actions[0].sl}"

