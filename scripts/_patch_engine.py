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


# 1) engine.py — backtesting.py omits 'Commissions [$]' when commission == 0.0
patch_file(
    "backtest/engine.py",
    '"commissions_usd": float(s["Commissions [$]"]),',
    '"commissions_usd": float(s.get("Commissions [$]", 0.0)),',
)

# 2) verify_backtest.py — the fraud check flagged our own docstring mention of
#    the legacy fraud. Use AST so only real `randn(` CALLS are detected.
patch_file(
    "scripts/verify_backtest.py",
    """    all_src = ""
    for mod in (engine, strategies, baselines, costs, report):
        all_src += inspect.getsource(mod)
    check("no np.random.randn(100) in engine path", "randn(100)" not in all_src and "randn(" not in all_src,
          "found randn in engine source")""",
    """    import ast
    randn_calls: list = []
    for mod in (engine, strategies, baselines, costs, report):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "randn":
                    randn_calls.append((mod.__name__, node.lineno))
    check("no np.random.randn CALLS in engine path", len(randn_calls) == 0,
          f"found randn calls: {randn_calls}")""",
)
print("done")
