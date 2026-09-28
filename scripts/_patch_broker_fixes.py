"""One-shot patch: fix MockBroker spread proportionality + reconcile ticket matching."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def patch(rel: str, old: str, new: str) -> None:
    p = REPO / rel
    text = p.read_text(encoding="utf-8")
    if old not in text:
        print(f"PATTERN NOT FOUND in {rel}")
        raise SystemExit(1)
    text = text.replace(old, new, 1)
    p.write_text(text, encoding="utf-8")
    print(f"PATCHED {rel}")


# 1) MockBroker: spread must be proportional to price (fraction of mid).
patch(
    "live/mock_broker.py",
    """        self._require_symbol(symbol)
        half = self.spread / 2.0
        return {"bid": self._mid - half, "ask": self._mid + half,
                "last": self._mid, "spread": self.spread}""",
    """        self._require_symbol(symbol)
        half = self.spread / 2.0
        # Spread is a FRACTION of price (CostConfig.spread = 2.5bp round-trip):
        # bid = mid * (1 - spread/2), ask = mid * (1 + spread/2), so on
        # XAUUSD @ $2000 the spread is $0.50, not $0.00025.
        return {"bid": self._mid * (1 - half), "ask": self._mid * (1 + half),
                "last": self._mid, "spread": self.spread}""",
)

# 2) BaseBroker.reconcile: match by TICKET primarily; side+volume fuzzy
#    fallback for local positions whose broker ticket is unknown (0/None).
patch(
    "live/broker.py",
    """        broker = self.get_positions(symbol=symbol, magic=magic)
        local = list(local_positions)

        missing = [p for p in local
                   if not any(b.side == p.side for b in broker)]
        unknown = [b for b in broker
                   if not any(p.side == b.side for p in local)]
        mismatched: List[tuple] = []
        for p in local:
            for b in broker:
                if b.side == p.side:
                    vol_drift = abs(b.volume - p.volume) > 1e-9
                    price_drift = (p.open_price > 0 and b.open_price > 0 and
                                   abs(b.open_price - p.open_price) / p.open_price > 0.01)
                    if vol_drift or price_drift:
                        mismatched.append((p, b))
        ok = not missing and not unknown and not mismatched""",
    """        broker = self.get_positions(symbol=symbol, magic=magic)
        local = list(local_positions)

        broker_by_ticket = {b.ticket: b for b in broker}
        missing: List[PositionInfo] = []
        unknown: List[PositionInfo] = list(broker)
        mismatched: List[tuple] = []

        for p in local:
            # Primary key is the broker ticket (persisted from a prior session).
            if p.ticket > 0 and p.ticket in broker_by_ticket:
                b = broker_by_ticket[p.ticket]
                unknown = [u for u in unknown if u.ticket != b.ticket]
                vol_drift = abs(b.volume - p.volume) > 1e-9
                price_drift = (p.open_price > 0 and b.open_price > 0 and
                               abs(b.open_price - p.open_price) / p.open_price > 0.01)
                if vol_drift or price_drift:
                    mismatched.append((p, b))
                continue
            # Local ticket unknown (0/None): fuzzy-match against the remaining
            # broker positions by side + volume.  Exactly one match consumes
            # it; anything else is a local-only position (missing in broker).
            matches = [b for b in unknown if b.side == p.side]
            exact = [b for b in matches if abs(b.volume - p.volume) <= 1e-9]
            if len(exact) == 1:
                unknown = [u for u in unknown if u.ticket != exact[0].ticket]
            else:
                missing.append(p)

        ok = not missing and not unknown and not mismatched""",
)

print("done")
