"""Patch live/live_trade_mt5.py for SL/TP dual-mode wiring.

Exact-match replacements with count assertions:
1. SHORT_DISABLED Decision rebuild preserves decision.sl_frac/tp_frac
   (otherwise model-mode SL/TP would be silently dropped on the disabled-short path).
2. execute_entry() call forwards decision.sl_frac/tp_frac so the
   TradeExecutor can attach model-decided SL/TP prices in model mode.
"""

import sys
from pathlib import Path

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
TARGET = REPO / "live" / "live_trade_mt5.py"

text = TARGET.read_text(encoding="utf-8")


def apply(old: str, new: str, name: str, count: int = 1) -> None:
    global text
    n = text.count(old)
    if n != count:
        raise SystemExit(
            f"FATAL [{name}]: expected {count} hit(s), got {n}. "
            f"File may have changed; aborting without edits."
        )
    text = text.replace(old, new)
    print(f"OK [{name}]: {n} replacement(s)")


# --- 1. SHORT_DISABLED preserves fractions -------------------------------- #
apply(
    """        if decision.action == 2 and not self.cfg.broker.allow_short:
            decision = Decision(
                0, decision.confidence, 0.0, "SHORT_DISABLED", decision.info
            )""",
    """        if decision.action == 2 and not self.cfg.broker.allow_short:
            decision = Decision(
                0,
                decision.confidence,
                0.0,
                "SHORT_DISABLED",
                decision.info,
                decision.sl_frac,
                decision.tp_frac,
            )""",
    "short_disabled_fractions",
)

# --- 2. execute_entry forwards sl_frac/tp_frac ----------------------------- #
apply(
    """                    ok, reason, result = self.executor.execute_entry(
                        decision.action,
                        equity=self._current_equity(),
                        df=df,
                        market_data=md,
                        bar_time=bar_time,
                        size_multiplier=decision.size_multiplier,
                    )""",
    """                    ok, reason, result = self.executor.execute_entry(
                        decision.action,
                        equity=self._current_equity(),
                        df=df,
                        market_data=md,
                        bar_time=bar_time,
                        size_multiplier=decision.size_multiplier,
                        sl_frac=decision.sl_frac,
                        tp_frac=decision.tp_frac,
                    )""",
    "execute_entry_forwards_fractions",
)

TARGET.write_text(text, encoding="utf-8")
print("WROTE", TARGET)
