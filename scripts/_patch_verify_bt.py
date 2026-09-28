"""One-shot patch helper (surgical replacements immune to CRLF mismatch)."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def patch_file(rel: str, old: str, new: str, count: int = 1) -> bool:
    p = REPO / rel
    text = p.read_text(encoding="utf-8")
    if old not in text:
        print(f"SKIP {rel}: pattern not found")
        return False
    text = text.replace(old, new, count)
    p.write_text(text, encoding="utf-8")
    print(f"PATCHED {rel}")
    return True


# 1) verify_backtest.py — scope the strict SL/TP AST invariant to strategies
#    module only (BuyHold baseline is intentionally stopless by definition).
patch_file(
    "scripts/verify_backtest.py",
    """    # 2a. AST: every buy/sell call passes sl= and tp=
    bad_calls: list = []
    for mod in (strategies, baselines):
        tree = ast.parse(inspect.getsource(mod))""",
    """    # 2a. AST: every buy/sell call in the STRATEGY module passes sl= and tp=
    #     (baselines are excluded: BuyHold is intentionally stopless, and
    #      SeededRandom's calls are checked separately below).
    bad_calls: list = []
    for mod in (strategies,):
        tree = ast.parse(inspect.getsource(mod))""",
)

# 2) verify_backtest.py — SeededRandom baseline orders carry SL/TP (AST check)
patch_file(
    "scripts/verify_backtest.py",
    '    check("every buy()/sell() call has sl= and tp= kwargs", len(bad_calls) == 0,\n          f"bad calls: {bad_calls}")',
    '    check("every buy()/sell() call has sl= and tp= kwargs", len(bad_calls) == 0,\n          f"bad calls: {bad_calls}")\n    bad_base: list = []\n    tree_base = ast.parse(inspect.getsource(baselines))\n    for node in ast.walk(tree_base):\n        if not isinstance(node, ast.Call):\n            continue\n        fn = node.func\n        name = None\n        if isinstance(fn, ast.Attribute) and fn.attr in ("buy", "sell"):\n            name = fn.attr\n        elif isinstance(fn, ast.Name) and fn.id in ("buy", "sell"):\n            name = fn.id\n        if name is None:\n            continue\n        kws = {kw.arg for kw in node.keywords if kw.arg is not None}\n        if "sl" not in kws or "tp" not in kws:\n            bad_base.append((node.lineno, name, sorted(kws)))\n    check("SeededRandom baseline orders carry SL/TP (BuyHold exempt by design)",\n          len(bad_base) == 0, f"bad base calls: {bad_base}")',
)

# 3) verify_backtest.py — probe: the strategy INSTANCE lives on the stats
#    Series (_strategy attr), not on bt._strategy (which is the class).
patch_file(
    "scripts/verify_backtest.py",
    """    bt.run()
    strat: _CaptureOrders = bt._strategy
    stops = getattr(strat, "_order_stops", [])""",
    """    stats_probe = bt.run()
    strat = getattr(stats_probe, "_strategy", None)
    if strat is None:
        strat = stats_probe.get("_strategy", None)
    stops = getattr(strat, "_order_stops", [])""",
)
print("done")
