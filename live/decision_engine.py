from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from models.policy import PolicyOutput
from models.position_sizing import KellyPositionSizer


@dataclass(frozen=True)
class Decision:
    action: int
    confidence: float
    size_multiplier: float = 1.0
    reason: str = "SIGNAL"
    info: dict = field(default_factory=dict)
    sl_frac: float | None = None
    tp_frac: float | None = None


class DecisionEngine:
    def __init__(
        self,
        behavior_cfg,
        kelly: KellyPositionSizer | None = None,
        risk_per_trade: float | None = None,
    ) -> None:
        self.cfg = behavior_cfg
        self.kelly = kelly or KellyPositionSizer(
            max_position=1.0,
            kelly_fraction=float(behavior_cfg.kelly_fraction),
        )
        self.risk_per_trade = float(
            risk_per_trade
            if risk_per_trade is not None
            else getattr(behavior_cfg, "risk_per_trade", 0.02)
        )

    def filter(
        self,
        raw: PolicyOutput | Decision,
        now_utc: datetime,
        position_side: int,
        bars_since_last_loss: int | None,
        recent_trades: list[dict],
    ) -> Decision:
        if isinstance(raw, Decision):
            action = int(raw.action)
            confidence = float(raw.confidence)
            info = dict(raw.info)
            reason = raw.reason
            sl_frac = raw.sl_frac
            tp_frac = raw.tp_frac
        else:
            action = int(raw.action)
            confidence = float(raw.confidence)
            info = dict(raw.info)
            reason = "MODEL_SIGNAL"
            sl_frac = None
            tp_frac = None

        if reason == "NO_SIGNAL" or info.get("hold"):
            return Decision(position_side, confidence, 0.0, reason, info, sl_frac, tp_frac)
        if confidence < self.cfg.min_confidence:
            return Decision(position_side, confidence, 0.0, "LOW_CONFIDENCE", info, sl_frac, tp_frac)
        if (
            ("consensus" in info and not bool(info["consensus"]))
            or (
                "agreement" in info
                and float(info["agreement"]) < self.cfg.min_ensemble_agreement
            )
        ):
            return Decision(position_side, confidence, 0.0, "NO_CONSENSUS", info, sl_frac, tp_frac)

        if action == position_side:
            return Decision(action, confidence, 1.0, "HOLD", info, sl_frac, tp_frac)
        is_entry = action != 0
        if not is_entry:
            return Decision(action, confidence, 1.0, reason, info, sl_frac, tp_frac)

        entry_block = self._entry_block(now_utc, bars_since_last_loss)
        if entry_block:
            target = 0 if position_side and action != position_side else position_side
            return Decision(target, confidence, 0.0, entry_block, info, sl_frac, tp_frac)

        confidence_scale = float(
            np.clip(
                (confidence - self.cfg.min_confidence)
                / (1.0 - self.cfg.min_confidence),
                0.25,
                1.0,
            )
        )
        kelly_scale = self._kelly_scale(recent_trades)
        if kelly_scale <= 0.0:
            target = 0 if position_side and action != position_side else position_side
            return Decision(target, confidence, 0.0, "NO_EDGE", info, sl_frac, tp_frac)
        return Decision(
            action,
            confidence,
            min(1.0, confidence_scale * kelly_scale),
            reason,
            info,
            sl_frac,
            tp_frac,
        )

    def _entry_block(
        self, now_utc: datetime, bars_since_last_loss: int | None
    ) -> str:
        if self.cfg.session_filter and self._session_closed(now_utc):
            return "SESSION_BLOCKED"
        if (
            bars_since_last_loss is not None
            and self.cfg.cooldown_bars_after_loss > 0
            and bars_since_last_loss < self.cfg.cooldown_bars_after_loss
        ):
            return "COOLDOWN_AFTER_LOSS"
        return ""

    def _session_closed(self, now_utc: datetime) -> bool:
        now = now_utc
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        now = now.astimezone(timezone.utc)
        if now.weekday() == 4 and now.hour >= 20:
            return True
        if now.weekday() in (6, 0) and now.hour < 1:
            return True
        return now.hour in self.cfg.no_trade_hours_utc

    def _kelly_scale(self, recent_trades: list[dict]) -> float:
        if not self.cfg.use_kelly or len(recent_trades) < self.cfg.kelly_min_trades:
            return 1.0
        returns = []
        for trade in recent_trades:
            equity = float(trade.get("equity", 0.0) or 0.0)
            if equity <= 0.0:
                continue
            returns.append(float(trade.get("pnl", 0.0)) / equity)
        if len(returns) < self.cfg.kelly_min_trades:
            return 1.0
        wins = [value for value in returns if value > 0.0]
        losses = [-value for value in returns if value < 0.0]
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        win_prob = len(wins) / len(returns)
        if avg_win <= 0.0:
            return 0.0
        fraction = self.kelly.compute_position_size(
            win_prob, avg_win, avg_loss
        )
        if fraction <= 0.0:
            return 0.0
        if self.risk_per_trade <= 0.0:
            return 0.0
        return float(np.clip(fraction / self.risk_per_trade, 0.0, 1.0))
