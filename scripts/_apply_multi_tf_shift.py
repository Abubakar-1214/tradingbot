"""
scripts/_apply_multi_tf_shift.py — add causal shift(1) to multi_timeframe.py.

After the resample with label='right', closed='right', every higher-timeframe
frame is shifted one period so a base bar at time t only ever sees higher-TF
candles that already closed.  The base timeframe itself is not shifted.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "features" / "multi_timeframe.py"


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    old = """        }).dropna()

        data_dict[tf] = resampled"""
    new = """        }).dropna()

        # P0-4 FIX: a higher-TF candle is only known once it CLOSES.  Shifting
        # every non-base frame by one period guarantees that forward-filling it
        # onto the base index can never leak the current (unclosed) candle's
        # close into the lower timeframe.
        if tf != base_tf:
            resampled = resampled.shift(1).dropna()

        data_dict[tf] = resampled"""

    if old not in text:
        print("[SKIP] pattern not found (already applied?)")
        return 1
    text = text.replace(old, new, 1)
    TARGET.write_text(text, encoding="utf-8")
    print("[OK] multi_timeframe shift(1) applied for higher timeframes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
