"""Ops acceptance cross-check: env vars documented in .env.example vs used in code;
requirements pins; README honest-status. Prints a numbered report (exit 0 on pass)."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_EXAMPLE = ROOT / ".env.example"

# [1] documented keys from .env.example
doc_keys = set()
if ENV_EXAMPLE.exists():
    for line in ENV_EXAMPLE.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("#") or not line or "=" not in line:
            continue
        doc_keys.add(line.split("=", 1)[0].strip())
print(f"[1] .env.example documented keys: {len(doc_keys)}")

# [2] keys referenced in repo *.py
pat_env = re.compile(r"""os\.environ(?:\.get|\[)?\(?\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""")
pat_getenv = re.compile(r"""os\.getenv\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""")
pat_dotenv = re.compile(r"""os\.environ\s*\[\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""")

used, undocumented = set(), set()
skip_dirs = {"venv", ".venv", "__pycache__", "archive", "research", ".git", "node_modules", "artifacts"}
for py in ROOT.rglob("*.py"):
    rel = py.relative_to(ROOT)
    if any(part in skip_dirs for part in rel.parts):
        continue
    text = py.read_text(encoding="utf-8", errors="replace")
    keys = set(pat_env.findall(text)) | set(pat_getenv.findall(text)) | set(pat_dotenv.findall(text))
    for k in keys:
        used.add(k)
        if k not in doc_keys:
            undocumented.add(k)
print(f"[2] env keys referenced in repo *.py: {len(used)}")
if undocumented:
    print("[3] UNDOCUMENTED keys (in code, missing from .env.example):")
    for k in sorted(undocumented):
        print("    " + k)
    print("CROSSCHECK: FAIL")
else:
    print("[3] every referenced key is documented in .env.example: OK")
    print("CROSSCHECK: PASS")

# [4] config.py loads .env via python-dotenv
cfg_text = (ROOT / "core" / "config.py").read_text(encoding="utf-8", errors="replace")
print(f"[4] config.py uses load_dotenv: {'load_dotenv' in cfg_text}")

# [5/6] requirements pinned
req_text = (ROOT / "requirements.txt").read_text(encoding="utf-8", errors="replace")
pins = [ln for ln in req_text.splitlines() if "==" in ln and not ln.strip().startswith("#")]
print(f"[5] requirements.txt pinned == lines: {len(pins)}")
print(f"[6] backtesting==0.6.2 pinned: {'backtesting==0.6.2' in req_text}")

# [7/8/9] README honest status
readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="replace")
print(f"[7] README contains '80-120%': {'80-120%' in readme}")
print(f"[8] README contains 'Sharpe 3.5': {'Sharpe 3.5' in readme}")
print(f"[9] README contains 'NOT PRODUCTION READY': {'NOT PRODUCTION READY' in readme}")
print("OPS CROSSCHECK DONE")
