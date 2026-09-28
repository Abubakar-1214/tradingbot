"""
live/idempotency.py — persisted order-idempotency guard.

Solves P0-6's "no idempotency" audit finding at the broker level:

  * Every order intent carries a deterministic ``token`` derived from
    (symbol, side, decision bar close time, signal).  If the process crashes
    between ``order_send`` and confirmation (or a retry duplicates a request
    the broker already executed), the guard detects the token on restart and
    returns the already-recorded ticket instead of submitting again.

State is persisted atomically to ``<state>/broker_idempotency.json`` so the
guard survives process restarts.  Old entries are pruned after ``max_age_days``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional


class IdempotencyGuard:
    """Persist token -> {ticket, ts, symbol} records with atomic writes."""

    def __init__(self, state_path: Path, max_age_days: float = 14.0):
        self.state_path = Path(state_path)
        self.max_age_days = float(max_age_days)
        self._records: Dict[str, Dict[str, object]] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._records = data.get("records", {})
        except (json.JSONDecodeError, OSError, ValueError):
            # Corrupt state file must never block trading; start fresh but
            # leave the corrupt file for manual inspection (renamed).
            if self.state_path.exists():
                try:
                    self.state_path.replace(self.state_path.with_suffix(".json.corrupt"))
                except OSError:
                    pass
            self._records = {}

    def _save(self) -> None:
        payload = {
            "records": self._records,
            "updated_ts": time.time(),
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Atomic-ish replace on Windows (same filesystem).
        tmp.replace(self.state_path)

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def lookup(self, token: str) -> Optional[Dict[str, object]]:
        """Return the recorded result for a token, or None if not submitted."""
        rec = self._records.get(token)
        if rec is None:
            return None
        # Prune stale records lazily (ignore failures silently).
        if time.time() - float(rec.get("ts", 0.0)) > self.max_age_days * 86400:
            self._records.pop(token, None)
            try:
                self._save()
            except OSError:
                pass
            return None
        return rec

    def record(self, token: str, ticket: Optional[int],
               symbol: str = "", fill_price: Optional[float] = None,
               volume: float = 0.0) -> None:
        """Persist that ``token`` was submitted and filled as ``ticket``."""
        if not token:
            return
        self._records[token] = {
            "ticket": ticket,
            "ts": time.time(),
            "symbol": symbol,
            "fill_price": fill_price,
            "volume": volume,
        }
        try:
            self._save()
        except OSError:
            # A failed write must not abort the order path; the in-memory
            # guard still prevents double-submit within this process.
            pass

    def size(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records = {}
        if self.state_path.exists():
            try:
                self.state_path.unlink()
            except OSError:
                pass
