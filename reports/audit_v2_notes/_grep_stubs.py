"""Audit v2 helper: grep model/core/live/train for remaining stub patterns.

Scans only the production code dirs (models, core, live, train) and prints
file:line matches for:
  - bare `pass` statements
  - NotImplemented / NotImplementedError
  - TODO / FIXME
  - `return None` in methods
  - commented-out core calls (lines starting with # that mention .train(
     .learn(, .fit(, optimizer.step)
Prints the full result to stdout (captured by the audit run).
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATTERNS = {
    "BARE_PASS": re.compile(r"^\s*pass\s*$"),
    "NOT_IMPLEMENTED": re.compile(r"NotImplemented"),
    "TODO_FIXME": re.compile(r"TODO|FIXME"),
    "RETURN_NONE": re.compile(r"^\s*return None\b"),
    "COMMENTED_CORE_CALL": re.compile(
        r"^\s*#.*\b(trader\.train|\.train_step|\.learn\(|\.fit\(|\.backward\(|optimizer\.step|mm\.learn|agent\.learn)\b"
    ),
}

DIRS = ["models", "core", "live", "train"]


def main():
    rows = []
    for d in DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                path = os.path.join(root, f)
                rel = os.path.relpath(path, ROOT)
                with open(path, encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        stripped = line.rstrip("\n")
                        for tag, pat in PATTERNS.items():
                            if pat.search(stripped):
                                rows.append(f"{tag} {rel}:{i}: {stripped.strip()}")
    print(f"TOTAL MATCHES: {len(rows)}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
