"""One-shot patch: make the REQUOTE re-quote check in verify_broker.py real."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
p = REPO / "scripts" / "verify_broker.py"
text = p.read_text(encoding="utf-8")

# 1) FakeMt5: move the price when a transient retcode is served so a real
#    re-quote produces a DIFFERENT price on the next attempt.
old_order_send = """    def order_send(self, request) -> SimpleNamespace:
        self.calls.append(("order_send", (), {"request": request}))
        retcode = self.retcodes.pop(0) if self.retcodes else TRADE_RETCODE_DONE
        return SimpleNamespace("""
new_order_send = """    def order_send(self, request) -> SimpleNamespace:
        self.calls.append(("order_send", (), {"request": request}))
        retcode = self.retcodes.pop(0) if self.retcodes else TRADE_RETCODE_DONE
        # On transient retcodes the broker re-quotes: move the market so the
        # retry genuinely fetches a fresh (different) price.
        if retcode in (10004, 10020, 10021):
            self.ask += 0.50
            self.bid += 0.48
        return SimpleNamespace("""
if old_order_send not in text:
    print("PATTERN 1 NOT FOUND")
    raise SystemExit(1)
text = text.replace(old_order_send, new_order_send, 1)

# 2) Replace the weak `or True` check with a real re-quote-price assertion.
old_check = """    check("retry re-quoted a fresh price", sends[0][2]["request"]["price"] != sends[1][2]["request"]["price"] or True)"""
new_check = """    check("retry re-quoted a fresh price",
          sends[1][2]["request"]["price"] > sends[0][2]["request"]["price"] + 0.4,
          f"attempt1={sends[0][2]['request']['price']} attempt2={sends[1][2]['request']['price']}")"""
if old_check not in text:
    print("PATTERN 2 NOT FOUND")
    raise SystemExit(1)
text = text.replace(old_check, new_check, 1)

p.write_text(text, encoding="utf-8")
print("PATCHED scripts/verify_broker.py (real re-quote price check)")
