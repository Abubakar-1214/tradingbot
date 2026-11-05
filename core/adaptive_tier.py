from __future__ import annotations

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccountTier:
    name: str
    tp_atr_mult: float
    sl_atr_mult: float
    breakeven_at_r: float
    trail_start_r: float
    trail_atr_mult: float
    max_bars_in_trade: int


@dataclass(frozen=True)
class AdaptiveTierConfig:
    enabled: bool = True
    is_cent_account: bool = False
    tier1_max_equity: float = 20.0
    tier2_max_equity: float = 50.0

    # Tier 1: Micro Scalper ($5 - $20)
    tier1_tp_atr_mult: float = 0.8
    tier1_sl_atr_mult: float = 1.2
    tier1_breakeven_r: float = 0.4
    tier1_trail_start_r: float = 0.6
    tier1_trail_atr_mult: float = 0.8
    tier1_max_bars: int = 20

    # Tier 2: Growth Scalper ($20 - $50)
    tier2_tp_atr_mult: float = 1.8
    tier2_sl_atr_mult: float = 1.8
    tier2_breakeven_r: float = 0.8
    tier2_trail_start_r: float = 1.2
    tier2_trail_atr_mult: float = 1.5
    tier2_max_bars: int = 40

    # Tier 3: Trend Expansion ($50+)
    tier3_tp_atr_mult: float = 3.0
    tier3_sl_atr_mult: float = 2.0
    tier3_breakeven_r: float = 1.0
    tier3_trail_start_r: float = 1.5
    tier3_trail_atr_mult: float = 2.0
    tier3_max_bars: int = 72


def resolve_tier(
    equity: float,
    config: AdaptiveTierConfig | None = None,
    default_sl_atr_mult: float = 2.0,
    default_tp_atr_mult: float = 3.0,
) -> AccountTier:
    """Resolve the trading tier based on effective account equity.

    Converts Cent account units to normalized USD if is_cent_account is set.
    """
    cfg = config or AdaptiveTierConfig()

    if not cfg.enabled:
        return AccountTier(
            name="STANDARD_STATIC",
            tp_atr_mult=default_tp_atr_mult,
            sl_atr_mult=default_sl_atr_mult,
            breakeven_at_r=1.0,
            trail_start_r=1.5,
            trail_atr_mult=2.0,
            max_bars_in_trade=72,
        )

    # Normalize equity if cent account
    eff_equity = float(equity) / 100.0 if cfg.is_cent_account else float(equity)

    if eff_equity < cfg.tier1_max_equity:
        return AccountTier(
            name="MICRO_SCALPER",
            tp_atr_mult=cfg.tier1_tp_atr_mult,
            sl_atr_mult=cfg.tier1_sl_atr_mult,
            breakeven_at_r=cfg.tier1_breakeven_r,
            trail_start_r=cfg.tier1_trail_start_r,
            trail_atr_mult=cfg.tier1_trail_atr_mult,
            max_bars_in_trade=cfg.tier1_max_bars,
        )
    elif eff_equity < cfg.tier2_max_equity:
        return AccountTier(
            name="GROWTH_SCALPER",
            tp_atr_mult=cfg.tier2_tp_atr_mult,
            sl_atr_mult=cfg.tier2_sl_atr_mult,
            breakeven_at_r=cfg.tier2_breakeven_r,
            trail_start_r=cfg.tier2_trail_start_r,
            trail_atr_mult=cfg.tier2_trail_atr_mult,
            max_bars_in_trade=cfg.tier2_max_bars,
        )
    else:
        return AccountTier(
            name="EXPANSION_RIDER",
            tp_atr_mult=cfg.tier3_tp_atr_mult,
            sl_atr_mult=cfg.tier3_sl_atr_mult,
            breakeven_at_r=cfg.tier3_breakeven_r,
            trail_start_r=cfg.tier3_trail_start_r,
            trail_atr_mult=cfg.tier3_trail_atr_mult,
            max_bars_in_trade=cfg.tier3_max_bars,
        )

