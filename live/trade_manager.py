from __future__ import annotations

import math
from dataclasses import dataclass

from live.broker import PositionInfo

MIN_LOT = 0.01
LOT_STEP = 0.01


@dataclass(frozen=True)
class ModifySL:
    ticket: int
    sl: float


@dataclass(frozen=True)
class PartialClose:
    ticket: int
    volume: float


@dataclass(frozen=True)
class Close:
    ticket: int
    reason: str


class TradeManager:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.behavior = getattr(cfg, "behavior", cfg)
        self.spread_fraction = float(
            getattr(getattr(cfg, "cost", None), "spread", 0.00025)
        )
        self.sl_atr_mult = float(
            getattr(getattr(cfg, "broker", None), "sl_atr_mult", 2.0)
        )

    def manage(
        self,
        position: PositionInfo,
        entry_atr: float,
        bars_held: int,
        last_bar,
        atr_now: float,
        trade_meta: dict | None = None,
    ) -> list[ModifySL | PartialClose | Close]:
        meta = trade_meta or {}
        initial_sl = meta.get("initial_sl", position.sl)
        risk = (
            abs(float(position.open_price) - float(initial_sl))
            if initial_sl is not None
            else float(entry_atr) * self.sl_atr_mult
        )
        if risk <= 0.0:
            return []
        close = float(last_bar["close"])
        sign = 1.0 if position.side == "buy" else -1.0
        r_multiple = (close - float(position.open_price)) * sign / risk
        behavior = self.behavior
        if (
            behavior.max_bars_in_trade > 0
            and bars_held >= behavior.max_bars_in_trade
            and r_multiple < 0.5
        ):
            return [Close(position.ticket, "TIME_STOP")]

        actions: list[ModifySL | PartialClose | Close] = []
        current_sl = position.sl
        target_sl = current_sl
        spread_buffer = float(position.open_price) * self.spread_fraction
        if r_multiple >= behavior.breakeven_at_r:
            breakeven = float(position.open_price) + sign * spread_buffer
            if self._tightens(position.side, current_sl, breakeven):
                target_sl = breakeven
        if r_multiple >= behavior.trail_start_r and atr_now > 0.0:
            trailing = close - sign * behavior.trail_atr_mult * float(atr_now)
            reference_sl = target_sl
            if self._tightens(position.side, reference_sl, trailing):
                target_sl = trailing
        if target_sl is not None and target_sl != current_sl:
            actions.append(ModifySL(position.ticket, float(target_sl)))

        if (
            behavior.partial_close_fraction > 0.0
            and r_multiple >= behavior.partial_close_at_r
            and not bool(meta.get("partial_done", False))
        ):
            volume = math.floor(
                position.volume * behavior.partial_close_fraction / LOT_STEP + 1e-9
            ) * LOT_STEP
            remaining = position.volume - volume
            if volume >= MIN_LOT and remaining >= MIN_LOT - 1e-9:
                actions.append(PartialClose(position.ticket, round(volume, 8)))
        return actions

    @staticmethod
    def _tightens(side: str, current: float | None, candidate: float) -> bool:
        if current is None:
            return True
        if side == "buy":
            return candidate > current
        return candidate < current
