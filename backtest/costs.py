"""
backtest/costs.py — explicit, honest cost model shared by every backtest.

The legacy engine faked costs by charging a flat per-trade fee and then
generating ``np.random.randn(100)`` observations.  This module instead turns
the validated :class:`core.config.CostConfig` values into the exact
``spread`` / ``commission`` arguments consumed by kernc/backtesting.py 0.6.2.

Verified broker semantics (source: ``backtesting/backtesting.py`` 0.6.2):
    * fill price for an order of signed size ``s`` is
      ``price * (1 + copysign(spread, s))`` — a long pays the ask, a short
      receives the bid, so ``spread`` is a ONE-WAY fill cost (round-trip cost
      is 2 * spread).
    * commission is ``fixed + |size| * price * relative``; we use the
      relative form only.

We model adverse slippage by folding it into the one-way spread parameter
(conservative: slippage hurts both entry and exit), and document that choice
in every report so the cost assumptions are auditable.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.config import CostConfig


@dataclass(frozen=True)
class CostModel:
    """Explicit backtest cost model.

    Attributes:
        spread: ONE-WAY relative fill cost passed to ``Backtest(spread=...)``.
            Includes half the round-trip spread plus adverse slippage.
        commission: relative commission per side (fraction of notional).
        slippage: the adverse-slippage component, kept for reporting only.
    """

    spread: float = 0.000175
    commission: float = 0.00003
    slippage: float = 0.00005

    @classmethod
    def from_config(cls, cost: CostConfig) -> "CostModel":
        """Build the backtesting.py cost arguments from the validated config.

        cost.spread  = round-trip spread fraction (e.g. 0.00025 = 2.5 bp)
        cost.slippage = per-fill adverse slippage fraction
        cost.commission = per-side relative commission

        Backtesting.py applies ``spread`` on EVERY fill, so the equivalent
        one-way spread parameter is ``round_trip_spread / 2 + slippage``.
        """
        one_way_spread = cost.spread / 2.0 + cost.slippage
        return cls(spread=one_way_spread, commission=cost.commission, slippage=cost.slippage)

    @property
    def effective_spread(self) -> float:
        return self.spread

    @property
    def round_trip_cost(self) -> float:
        """Total round-trip cost (entry + exit), as fraction of notional."""
        return 2.0 * (self.spread + self.commission)

    def describe(self) -> str:
        return (
            f"one-way spread={self.spread:.6f} "
            f"(= round-trip {self.spread * 2:.6f}), "
            f"commission/side={self.commission:.6f}, "
            f"slippage/fill={self.slippage:.6f}, "
            f"round-trip total={self.round_trip_cost:.6f}"
        )
