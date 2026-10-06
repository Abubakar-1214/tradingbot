"""Patch live/trade_executor.py for SL/TP dual-mode (model-decided fractions).

Exact-match replacements with count assertions:
1. Import SLTP_MODE_MODEL alongside AppConfig from core.config.
2. Add entry_sl_tp_model() method (clamped fraction -> entry*(1∓frac) prices)
   right after the existing ATR entry_sl_tp() method (which stays byte-identical).
3. execute_entry() signature gains sl_frac=None/tp_frac=None kwargs.
4. execute_entry() SL/TP computation branches on cfg.behavior.sl_tp_mode
   (model mode uses clamped fractions; rules mode uses the ATR path unchanged).
"""

import sys
from pathlib import Path

REPO = Path(r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader")
TARGET = REPO / "live" / "trade_executor.py"

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


# --- 1. import ------------------------------------------------------------ #
apply(
    "from core.config import AppConfig",
    "from core.config import AppConfig, SLTP_MODE_MODEL",
    "import",
)

# --- 2. entry_sl_tp_model method ------------------------------------------ #
OLD_ENTRY_SL_TP = (
    "    def entry_sl_tp(self, side: str, entry: float, atr: float) -> Tuple[float, float]:\n"
    "        if side == \"buy\":\n"
    "            return entry - atr * self.cfg.broker.sl_atr_mult, entry + atr * self.cfg.broker.tp_atr_mult\n"
    "        return entry + atr * self.cfg.broker.sl_atr_mult, entry - atr * self.cfg.broker.tp_atr_mult\n"
)
NEW_ENTRY_SL_TP = OLD_ENTRY_SL_TP + (
    "\n"
    "    def entry_sl_tp_model(\n"
    "        self, side: str, entry: float, sl_frac: float, tp_frac: float\n"
    "    ) -> Tuple[float, float]:\n"
    "        \"\"\"SL/TP prices from model-decided fractions, clamped to bounds.\n"
    "\n"
    "        Long:  SL = entry * (1 - sl_frac), TP = entry * (1 + tp_frac)\n"
    "        Short: SL = entry * (1 + sl_frac), TP = entry * (1 - tp_frac)\n"
    "        Each fraction is clamped to its configured bound range\n"
    "        (sl_tp_min_frac..sl_tp_max_frac / tp_min_frac..tp_max_frac) so a\n"
    "        model output can never create a stop/target outside the permitted\n"
    "        risk envelope.\n"
    "        \"\"\"\n"
    "        sl_frac = float(\n"
    "            min(max(sl_frac, self.cfg.behavior.sl_tp_min_frac), self.cfg.behavior.sl_tp_max_frac)\n"
    "        )\n"
    "        tp_frac = float(\n"
    "            min(max(tp_frac, self.cfg.behavior.tp_min_frac), self.cfg.behavior.tp_max_frac)\n"
    "        )\n"
    "        if side == \"buy\":\n"
    "            return entry * (1.0 - sl_frac), entry * (1.0 + tp_frac)\n"
    "        return entry * (1.0 + sl_frac), entry * (1.0 - tp_frac)\n"
)
apply(OLD_ENTRY_SL_TP, NEW_ENTRY_SL_TP, "entry_sl_tp_model")

# --- 3. execute_entry signature -------------------------------------------- #
OLD_SIG = (
    "        bar_time: Optional[str] = None,\n"
    "        size_multiplier: float = 1.0,\n"
    "    ) -> Tuple[bool, str, Optional[OrderResult]]:\n"
    "        \"\"\"Open a new position.  Risk-gated BEFORE the broker is touched.\"\"\"\n"
)
NEW_SIG = (
    "        bar_time: Optional[str] = None,\n"
    "        size_multiplier: float = 1.0,\n"
    "        sl_frac: Optional[float] = None,\n"
    "        tp_frac: Optional[float] = None,\n"
    "    ) -> Tuple[bool, str, Optional[OrderResult]]:\n"
    "        \"\"\"Open a new position.  Risk-gated BEFORE the broker is touched.\n"
    "\n"
    "        ``sl_frac``/``tp_frac`` carry the model-decided SL/TP fractions\n"
    "        (fractions of entry price) when ``cfg.behavior.sl_tp_mode ==\n"
    "        \"model\"``.  They are clamped to the configured bounds and converted\n"
    "        to stop/target prices ``entry * (1 - sl_frac)`` / ``entry * (1 +\n"
    "        tp_frac)`` (long) or the mirrored levels (short).  In rules mode\n"
    "        (default) the ATR rule path is used unchanged.\n"
    "        \"\"\"\n"
)
apply(OLD_SIG, NEW_SIG, "execute_entry_signature")

# --- 4. execute_entry SL/TP computation ------------------------------------- #
OLD_COMPUTE = (
    "        sl, tp = self.entry_sl_tp(\"buy\" if direction == 1 else \"sell\", entry, atr)\n"
    "        req = OrderRequest(\n"
    "            symbol=self.cfg.broker.symbol,\n"
    "            side=\"buy\" if direction == 1 else \"sell\",\n"
    "            volume=volume,\n"
    "            sl=sl,\n"
)
NEW_COMPUTE = (
    "        if (\n"
    "            self.cfg.behavior.sl_tp_mode == SLTP_MODE_MODEL\n"
    "            and sl_frac is not None\n"
    "            and tp_frac is not None\n"
    "        ):\n"
    "            sl, tp = self.entry_sl_tp_model(\n"
    "                \"buy\" if direction == 1 else \"sell\", entry, sl_frac, tp_frac\n"
    "            )\n"
    "        else:\n"
    "            sl, tp = self.entry_sl_tp(\"buy\" if direction == 1 else \"sell\", entry, atr)\n"
    "        req = OrderRequest(\n"
    "            symbol=self.cfg.broker.symbol,\n"
    "            side=\"buy\" if direction == 1 else \"sell\",\n"
    "            volume=volume,\n"
    "            sl=sl,\n"
)
apply(OLD_COMPUTE, NEW_COMPUTE, "execute_entry_sltp")

TARGET.write_text(text, encoding="utf-8")
print(f"patched {TARGET.name}: all 4 replacements applied")
