"""One-shot patch: fix MockBroker.set_mid (remove non-existent method call)."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
p = REPO / "live" / "mock_broker.py"
text = p.read_text(encoding="utf-8")

old = """    def set_mid(self, mid: float) -> None:
        \"\"\"Set the mid price directly (test convenience).\"\"\"
        self._mid = float(mid)
        self._check_stops_and_tps()"""

new = """    def set_mid(self, mid: float) -> None:
        \"\"\"Set the mid price directly (test convenience).

        SL/TP evaluation happens on ``advance(bar)`` where a full high/low/close
        bar is available; ``set_mid`` only moves the quote for the next fills.
        \"\"\"
        self._mid = float(mid)"""

if old not in text:
    print("PATTERN NOT FOUND — aborting")
    raise SystemExit(1)
text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")
print("PATCHED live/mock_broker.py (set_mid)")
