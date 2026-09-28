"""One-shot patch: scope the SeededRandom SL/TP AST check to the class body only."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
p = REPO / "scripts" / "verify_backtest.py"
text = p.read_text(encoding="utf-8")

old = """    bad_base: list = []
    tree_base = ast.parse(inspect.getsource(baselines))
    for node in ast.walk(tree_base):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = None
        if isinstance(fn, ast.Attribute) and fn.attr in ("buy", "sell"):
            name = fn.attr
        elif isinstance(fn, ast.Name) and fn.id in ("buy", "sell"):
            name = fn.id
        if name is None:
            continue
        kws = {kw.arg for kw in node.keywords if kw.arg is not None}
        if "sl" not in kws or "tp" not in kws:
            bad_base.append((node.lineno, name, sorted(kws)))
    check("SeededRandom baseline orders carry SL/TP (BuyHold exempt by design)",
          len(bad_base) == 0, f"bad base calls: {bad_base}")"""

new = """    # Scan ONLY the SeededRandom class body.  BuyHold's bare buy(size, tag)
    # (line 31) is intentional buy-and-hold and must never be flagged.
    bad_base: list = []
    tree_base = ast.parse(inspect.getsource(baselines))
    seeded = next((n for n in tree_base.body if isinstance(n, ast.ClassDef)
                   and n.name == "SeededRandom"), None)
    if seeded is not None:
        for inner in ast.walk(seeded):
            if not isinstance(inner, ast.Call):
                continue
            fn = inner.func
            name = None
            if isinstance(fn, ast.Attribute) and fn.attr in ("buy", "sell"):
                name = fn.attr
            elif isinstance(fn, ast.Name) and fn.id in ("buy", "sell"):
                name = fn.id
            if name is None:
                continue
            kws = {kw.arg for kw in inner.keywords if kw.arg is not None}
            if "sl" not in kws or "tp" not in kws:
                bad_base.append((inner.lineno, name, sorted(kws)))
    check("SeededRandom baseline orders carry SL/TP (BuyHold exempt by design)",
          len(bad_base) == 0, f"bad base calls: {bad_base}")"""

if old not in text:
    print("PATTERN NOT FOUND — aborting")
    raise SystemExit(1)
text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")
print("PATCHED scripts/verify_backtest.py (SeededRandom-scoped AST check)")
