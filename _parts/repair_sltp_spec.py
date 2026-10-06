"""Subtask 10 repair: fix three splice-placement issues in frontend_dashboard_design.md.

1. Split the merged table row  "| Device | `--device` | auto || SL/TP action mode | ..."
   into two rows (BLOCK_A had no leading newline when inserted).
2. Move the SL/TP source-badge note (BLOCK_C) from AFTER the "### 4.5" header
   (where it wrongly landed inside section 4.5) to the END of section 4.4
   (immediately BEFORE the 4.5 header).
3. Move the SL/TP bounds-clamp note (BLOCK_D) from AFTER the "### 4.7" header
   (wrongly inside section 4.7) to the END of section 4.6 (before the 4.7 header).
4. Cosmetic: "entry*(1-+frac)" -> "entry*(1-frac) / entry*(1+frac)" and
   "entry -+ atr*sl_atr_mult" -> "entry - atr*sl_atr_mult".

Block constants are imported from _parts/splice_sltp_spec.py so the search
patterns match the file byte-for-byte (same conv() CRLF conversion). CRLF is
preserved. Idempotent: each step prints SKIP when its pattern is already fixed.

Run with a file-I/O interpreter:
    E:\\python_3.11.9_installed\\python.exe _parts/repair_sltp_spec.py
"""
import importlib.util
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "splice_sltp_spec.py")
TARGET = os.path.join(HERE, "..", "frontend_dashboard_design.md")

spec = importlib.util.spec_from_file_location("splice_sltp_spec", SPEC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
BLOCK_C = mod.BLOCK_C
BLOCK_D = mod.BLOCK_D


def main() -> None:
    target = os.path.normpath(TARGET)
    with io.open(target, "r", encoding="utf-8", newline="") as f:
        raw = f.read()
    nl = "\r\n" if "\r\n" in raw else "\n"

    def conv(s: str) -> str:
        return s.replace("\n", nl)

    text = raw

    # ---- 1. split merged Device/SLTP table row ------------------------------
    merged = conv("| Device | `--device` | auto || SL/TP action mode |")
    split = conv("| Device | `--device` | auto |\n| SL/TP action mode |")
    if merged in text:
        text = text.replace(merged, split, 1)
        print("FIXED: merged Device/SLTP table row split")
    else:
        print("SKIP: merged Device/SLTP row not found (already split)")

    # ---- 2. move BLOCK_C from after 4.5 header to before it (end of 4.4) ----
    anchor45 = conv("### 4.5 Brain panel (per model type)")
    if anchor45 + conv(BLOCK_C) in text:
        text = text.replace(anchor45 + conv(BLOCK_C), anchor45, 1)
        note = conv(BLOCK_C).lstrip("\r\n").rstrip("\r\n")
        text = text.replace(anchor45, note + nl + nl + anchor45 + nl, 1)
        print("MOVED: SL/TP source-badge note to end of section 4.4")
    else:
        print("SKIP: BLOCK_C move (pattern not found; already moved?)")

    # ---- 3. move BLOCK_D from after 4.7 header to before it (end of 4.6) ----
    anchor47 = conv("### 4.7 Economic calendar event feed")
    if anchor47 + conv(BLOCK_D) in text:
        text = text.replace(anchor47 + conv(BLOCK_D), anchor47, 1)
        note = conv(BLOCK_D).lstrip("\r\n").rstrip("\r\n")
        text = text.replace(anchor47, note + nl + nl + anchor47 + nl, 1)
        print("MOVED: SL/TP bounds-clamp note to end of section 4.6")
    else:
        print("SKIP: BLOCK_D move (pattern not found; already moved?)")

    # ---- 4. cosmetic notation fixes (both occurrences) ----------------------
    if "entry*(1-+frac)" in text:
        text = text.replace("entry*(1-+frac)", "entry*(1-frac) / entry*(1+frac)")
        print("FIXED: entry*(1-+frac) notation")
    else:
        print("SKIP: entry*(1-+frac) already fixed")
    if "entry -+ atr*sl_atr_mult" in text:
        text = text.replace("entry -+ atr*sl_atr_mult", "entry - atr*sl_atr_mult")
        print("FIXED: entry -+ atr notation")
    else:
        print("SKIP: entry -+ atr notation already fixed")

    with io.open(target, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"target now {text.count(nl) + 1} lines")


if __name__ == "__main__":
    main()
