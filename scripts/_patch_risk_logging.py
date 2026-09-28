"""scripts/_patch_risk_logging.py — fix invalid printf format spec in risk_supervisor.py.

The __init__ logger.info used ``%.1%%`` which is not a valid printf conversion
(``%`` conversion accepts no precision), raising
``ValueError: unsupported format character '%' (0x25)`` on EVERY construction
with logging enabled.  Fix: ``%.1f%%``.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "models" / "risk_supervisor.py"

BAD = (
    '"RiskSupervisor initialized (daily_loss=%.1%%, dd=%.1%%, max_pos=%.1%%, "'
    '"max_losses=%d, corr_guard=%s)",'
)
GOOD = (
    '"RiskSupervisor initialized (daily_loss=%.1f%%, dd=%.1f%%, max_pos=%.1f%%, "'
    '"max_losses=%d, corr_guard=%s)",'
)

text = TARGET.read_text(encoding="utf-8")
if BAD in text:
    text = text.replace(BAD, GOOD)
    TARGET.write_text(text, encoding="utf-8")
    print("PATCHED models/risk_supervisor.py (%.1%% -> %.1f%%)")
else:
    # Fallback: patch the two-line message text directly.
    msg_bad = '"RiskSupervisor initialized (daily_loss=%.1%%, dd=%.1%%, max_pos=%.1%%, "'
    msg_good = '"RiskSupervisor initialized (daily_loss=%.1f%%, dd=%.1f%%, max_pos=%.1f%%, "'
    if msg_bad in text:
        text = text.replace(msg_bad, msg_good)
        TARGET.write_text(text, encoding="utf-8")
        print("PATCHED models/risk_supervisor.py (msg line, fallback)")
    else:
        print("NOT_FOUND: no bad format string present — already fixed?")
