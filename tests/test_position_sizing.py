"""Position-sizing pytest: Kelly math, fractional Kelly cap, ATR sizing, volatility adjust."""
from __future__ import annotations

from models.position_sizing import ATRPositionSizer, FixedFractionSizer, KellyPositionSizer


def test_kelly_strong_edge() -> None:
    sizer = KellyPositionSizer(max_position=0.10, kelly_fraction=0.25)
    frac = sizer.compute_position_size(0.60, 0.02, 0.01, equity=10_000.0)
    # Full Kelly for p=0.6, b=2: f=(0.6*2-0.4)/2 = 0.4; quarter Kelly = 0.10.
    assert abs(frac - 0.10) < 1e-9


def test_kelly_no_edge_is_zero() -> None:
    sizer = KellyPositionSizer(max_position=0.10, kelly_fraction=0.25)
    # p=0.5, b=1 -> f = (0.5*1 - 0.5)/1 = 0 -> no trade.
    frac = sizer.compute_position_size(0.50, 0.01, 0.01, equity=10_000.0)
    assert frac == 0.0


def test_kelly_negative_edge_is_zero() -> None:
    sizer = KellyPositionSizer(max_position=0.10, kelly_fraction=0.25)
    frac = sizer.compute_position_size(0.40, 0.01, 0.01, equity=10_000.0)
    assert frac == 0.0


def test_kelly_capped_at_max_position() -> None:
    sizer = KellyPositionSizer(max_position=0.05, kelly_fraction=1.0)
    # Full Kelly f=0.4 > max_position 0.05 -> capped.
    frac = sizer.compute_position_size(0.60, 0.02, 0.01, equity=10_000.0)
    assert abs(frac - 0.05) < 1e-9


def test_kelly_avg_loss_zero_returns_min() -> None:
    sizer = KellyPositionSizer()
    frac = sizer.compute_position_size(0.60, 0.02, 0.0, equity=10_000.0)
    assert frac == 0.01


def test_fixed_fraction_sizer() -> None:
    sizer = FixedFractionSizer(risk_per_trade=0.02)
    assert sizer.compute_position_size(equity=10_000.0) == 0.02


def test_atr_sizing_larger_atr_smaller_position() -> None:
    sizer = ATRPositionSizer(account_risk=0.02, atr_multiplier=2.0)
    # fraction = (equity*risk)/(atr*mult) * price / equity = risk*price/(atr*mult)
    # atr=10  -> 0.02*2000/20 = 2.0   -> capped 0.10
    # atr=100 -> 0.02*2000/200 = 0.20  -> capped 0.10
    # atr=200 -> 0.02*2000/400 = 0.10  -> 0.10
    # atr=400 -> 0.02*2000/800 = 0.05  -> 0.05
    f_low = sizer.compute_position_size(atr=10.0, price=2000.0, equity=10_000.0)
    f_mid = sizer.compute_position_size(atr=100.0, price=2000.0, equity=10_000.0)
    f_high = sizer.compute_position_size(atr=400.0, price=2000.0, equity=10_000.0)
    assert f_low == 0.10        # capped
    assert f_mid == 0.10        # exactly at cap
    assert abs(f_high - 0.05) < 1e-9
    # Monotonic: larger ATR -> smaller (or equal) position fraction.
    assert f_low >= f_mid >= f_high
    # Dollar-risk invariant at the uncapped point.  The sizer formula is
    #   fraction = dollar_risk / stop_distance * price / equity
    #            = (equity*risk) / stop_distance * price / equity
    #            = risk * price / stop_distance
    # so the per-trade risk fraction is  fraction * stop_distance / price.
    risk_frac = f_high * (400.0 * 2.0) / 2000.0
    assert abs(risk_frac - 0.02) < 1e-9


def test_atr_sizing_capped_at_10_percent() -> None:
    sizer = ATRPositionSizer(account_risk=0.02, atr_multiplier=2.0)
    frac = sizer.compute_position_size(atr=1.0, price=2000.0, equity=10_000.0)
    assert frac <= 0.10


def test_atr_sizing_zero_atr_default() -> None:
    sizer = ATRPositionSizer()
    assert sizer.compute_position_size(atr=0.0, price=2000.0, equity=10_000.0) == 0.02


def test_volatility_adjusted_sizing() -> None:
    sizer = KellyPositionSizer()
    # Higher volatility -> smaller position (inverse scaling), capped at max.
    low = sizer.volatility_adjusted_sizing(0.04, current_volatility=1.0,
                                           normal_volatility=2.0)
    high = sizer.volatility_adjusted_sizing(0.04, current_volatility=2.0,
                                            normal_volatility=2.0)
    assert abs(low - 0.08) < 1e-9   # base / 0.5
    assert abs(high - 0.04) < 1e-9  # base / 1.0
    # Capped at max_position.
    capped = sizer.volatility_adjusted_sizing(0.10, current_volatility=1.0,
                                              normal_volatility=2.0)
    assert abs(capped - 0.10) < 1e-9
    # No div-by-zero when normal volatility is 0.
    assert sizer.volatility_adjusted_sizing(0.10, 2.0, 0.0) == 0.10
