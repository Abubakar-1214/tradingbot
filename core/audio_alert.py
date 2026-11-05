"""core/audio_alert.py — Non-blocking audio notifications for trade open/close."""

import logging
import threading

logger = logging.getLogger("audio_alert")


def play_trade_open_sound():
    """Rising tone chime when a trade opens."""
    def _worker():
        try:
            import winsound
            winsound.Beep(900, 120)
            winsound.Beep(1400, 180)
        except Exception:
            print("\a", end="", flush=True)

    threading.Thread(target=_worker, daemon=True).start()


def play_trade_close_sound(profit: float = 0.0):
    """Pleasant triumph / profit chime when trade closes."""
    def _worker():
        try:
            import winsound
            if profit >= 0:
                winsound.Beep(1046, 100)  # C6
                winsound.Beep(1318, 100)  # E6
                winsound.Beep(1568, 220)  # G6 (Profit locked!)
            else:
                winsound.Beep(500, 250)
        except Exception:
            print("\a", end="", flush=True)

    threading.Thread(target=_worker, daemon=True).start()

