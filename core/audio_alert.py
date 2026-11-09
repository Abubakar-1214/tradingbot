"""core/audio_alert.py — Non-blocking audio notifications for trade open/close."""

import logging
import os
import threading

logger = logging.getLogger("audio_alert")


def play_trade_open_sound():
    """Play clear, audible notification through Windows speakers when a trade opens."""
    def _worker():
        try:
            import winsound
            wav_candidates = [
                r"C:\Windows\Media\notify.wav",
                r"C:\Windows\Media\Windows Ding.wav",
                r"C:\Windows\Media\chimes.wav",
            ]
            played = False
            for p in wav_candidates:
                if os.path.exists(p):
                    winsound.PlaySound(p, winsound.SND_FILENAME)
                    played = True
                    break
            if not played:
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            try:
                import winsound
                winsound.Beep(1000, 150)
            except Exception:
                print("\a", end="", flush=True)

    threading.Thread(target=_worker, daemon=True, name="audio_open").start()


def play_trade_close_sound(profit: float = 0.0):
    """Play celebratory sound on profit exit or distinct alert on close."""
    def _worker():
        try:
            import winsound
            if profit >= 0:
                wav_candidates = [
                    r"C:\Windows\Media\tada.wav",
                    r"C:\Windows\Media\chimes.wav",
                    r"C:\Windows\Media\notify.wav",
                ]
            else:
                wav_candidates = [
                    r"C:\Windows\Media\chord.wav",
                    r"C:\Windows\Media\Windows Foreground.wav",
                ]
            played = False
            for p in wav_candidates:
                if os.path.exists(p):
                    winsound.PlaySound(p, winsound.SND_FILENAME)
                    played = True
                    break
            if not played:
                winsound.MessageBeep(winsound.MB_OK if profit >= 0 else winsound.MB_ICONHAND)
        except Exception:
            try:
                import winsound
                winsound.Beep(1200, 200)
            except Exception:
                print("\a", end="", flush=True)

    threading.Thread(target=_worker, daemon=True, name="audio_close").start()
