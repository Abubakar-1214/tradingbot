"""CRLF-safe append of the legacy/deferred section to FIXES.md (append-only, no rewrite)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXES = ROOT / "FIXES.md"

section = """

---

## Legacy / deferred items — documented honestly (NOT claimed fixed)

These audit findings are **not** marked as fixed; they are intentionally left
as legacy research code or deferred P2 work. Each is verified to be outside
every wired acceptance path (no gate, test, or engine imports them), so they
cannot cause silent failures — but they are NOT production-ready:

| Item | Location | Status |
|---|---|---|
| `MockAgent` placeholder (returns `np.random.choice([0,1])`) + "Simple P&L calculation (placeholder)" | `eval/crisis_validation.py` lines 389-406 / 215 | **NOT fixed** — legacy eval tooling; verified `__NO_REFERENCES__` across `scripts/verify_*.py`, `tests/*.py`, `backtest/*.py`. The new `backtest/` engine + `backtest/report_backtest.md` are the only evaluation paths used. |
| Synthetic rule-based economic calendar (first-Friday NFP, fixed 13:30 UTC, no DST) | `scripts/generate_economic_calendar.py` lines 25-38 | **NOT fixed** — standalone data-generation utility; verified `__NO_REFERENCES__`. Real economic-calendar API + DST-aware ingestion is P2. `features/calendar_features.py` (the production feature layer) is vectorized + causal and consumes whatever event table it is given. |
| CI workflow + pyproject lint config | `.github/workflows/ci.yml`, `pyproject.toml` (absent) | **Deferred P2** — plan.md's Phase-3 ops item is satisfied by pinned `requirements.txt` + six verify gates + evidence runners (`scripts/_run_gates_twice.py`, `scripts/_run_pytest_evidence.py`). |
| Dead research modules (ensemble/mcts/transformer/meta_learning/adversarial_training) | `models/` | **Left in place** — research-only, imported by no wired path. |
| Trained PPO/Dreamer checkpoint | `train/ppo_xauusd_latest.zip` (absent) | **Deferred** — no GPU / ~1.7GB free RAM in sandbox; ML strategy path implemented but refuses to fabricate results without a checkpoint. |
"""

with FIXES.open("a", encoding="utf-8") as f:
    f.write(section)
print("[OK] appended legacy/deferred section to FIXES.md")
