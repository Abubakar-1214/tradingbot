"""Surgical fix #3 for frontend_dashboard_design.md (idempotent).

Inserts the obs_dim formula grounding note after the decision-feed JSON example
(section 7.3) and before the Risk state JSON example. fix_report2's anchor
mismatched the actual payload (COOLDOWN-rejection example, not a filled order),
so the note was never inserted.
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(os.path.dirname(ROOT), "frontend_dashboard_design.md")

ANCHOR = "**Risk state** (GET"
NOTE = (
    "> **obs_dim consistency note:** `obs_dim` always follows `window*n_features + 5` "
    "(the account vector is 5 scalars: position, trade_pnl, bars_in_trade, drawdown, "
    "equity_ratio). The example above uses `64*3 + 5 = 197`. The real Dreamer artifact "
    "in this repo uses `n_features=15 -> obs_dim=965` (see "
    "`artifacts/models/dreamer_20261003T230304213562Z/manifest.json` and its "
    "`feature_contract.json`), so the dashboard must read `obs_dim` from the active "
    "feature contract at runtime — never hard-code it. **[FACT `core/model_artifacts.py::load_manifest`]**\n\n"
)


def main():
    with io.open(REPORT, "r", encoding="utf-8") as f:
        text = f.read()
    if "n_features=15 -> obs_dim=965" in text:
        print("NOTE_ALREADY_PRESENT; no change")
        return 0
    if ANCHOR not in text:
        print("ANCHOR_NOT_FOUND")
        return 2
    text = text.replace(ANCHOR, NOTE + ANCHOR, 1)
    with io.open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"NOTE_INSERTED; lines={text.count(chr(10))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
