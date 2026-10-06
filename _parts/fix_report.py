"""Surgical fix script for frontend_dashboard_design.md.

Performs (idempotently):
 1. Insert Roman-Urdu framing paragraph after the overview no-server fact.
 2. Insert subsections 6.3-6.5 (from _parts/prep_sub_635.md) before section 7.
 3. Fix acceptance-criterion-6 script names to the REAL train scripts
    (meta_train_dreamer.py, make_mcts_manifest.py) verified on disk.
 4. Append Appendix A: verified fact inventory (real evaluation.json numbers).
 5. Report new line count.

Run: python _parts/fix_report.py  (from the project root)
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(os.path.dirname(ROOT), "frontend_dashboard_design.md")
PREP_SUB = os.path.join(ROOT, "prep_sub_635.md")


def read(path: str) -> str:
    with io.open(path, "r", encoding="utf-8") as f:
        return f.read()


def write(path: str, text: str) -> None:
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    text = read(REPORT)
    changed = False

    # --- 1. Roman-Urdu framing (idempotent) -------------------------------
    anchor = "but **no HTTP server or UI exists today**. **[FACT]**"
    if anchor in text and "is dashboard ka maqsad" not in text:
        framing = (
            anchor
            + "\n\n"
            + "*Light Roman-Urdu framing — is dashboard ka maqsad: system ko monitor "
            "karna, usay safely control karna, aur har number ka asli source dikhana. "
            "Koi fabricated metric nahi. (Purpose: monitor the system, control it safely, "
            "and show the true source of every number. No fabricated metrics.)*"
        )
        text = text.replace(anchor, framing, 1)
        changed = True

    # --- 2. Insert 6.3-6.5 before section 7 (idempotent) -----------------
    marker = "## 7. Proposed API surface"
    if "### 6.3 Hyperparameter preset library" not in text and marker in text:
        sub = read(PREP_SUB).rstrip() + "\n\n"
        text = text.replace(marker, sub + marker, 1)
        changed = True

    # --- 3. Fix acceptance-criterion-6 script names -----------------------
    wrong = (
        "(train_dreamer.py, train_transformer.py, train_ppo.py, train_ensemble.py, "
        "train_dreamer_mcts.py, train_adversarial.py, train_meta.py)"
    )
    right = (
        "(train_dreamer.py, train_transformer.py, train_ppo.py, train_ensemble.py, "
        "train_adversarial.py, meta_train_dreamer.py, make_mcts_manifest.py)"
    )
    if wrong in text:
        text = text.replace(wrong, right, 1)
        changed = True

    # --- 4. Appendix A: verified fact inventory (idempotent) --------------
    appendix_marker = "## Appendix A — Verified fact inventory"
    if appendix_marker not in text:
        appendix = """

---

## Appendix A — Verified fact inventory

Ground-truth values read directly from repo state files while writing this report. These are the numbers the dashboard must render honestly; they are **not** fabricated and none implies promotability. **[FACT]**

### A.1 Evaluation artifacts (every one `passed: false`)

| Artifact (`artifacts/models/.../evaluation.json`) | sharpe | max_dd_pct | trades | total_return_pct | buy_hold_return_pct | passed |
|---|---|---|---|---|---|---|
| `ppo_gold_v1` | -3.21 | 74.78 | 2849 | -74.74 | +136.48 | `false` |
| `dreamer_20261003T230304213562Z` | 3.10 | 3.04 | 3 | +6.04 | -5.47 | `false` |
| `dreamer_20261003T230535411560Z` | 1.71 | 0.02 | 1 | +0.02 | -5.47 | `false` |

Thresholds (identical across all three, from `train/evaluate.py`): `sharpe >= 0.5`, `max_dd_pct <= 20.0`, `trades >= 20`, `total_return_pct > 0.0`. The Dreamer runs fail on `trades`; the PPO run fails on sharpe/max_dd/return and loses to buy-and-hold — the honest baseline truth. **[FACT]**

### A.2 State files

- `artifacts/_smoke_state/mock_state.json`: `balance`, `mid`, `next_ticket`, `positions[]`, `closed[]` — closed-trade entries have `ticket/side/volume/entry/exit/pnl/reason` (reasons seen: `SL`, `TP`, `manual_close`). **[FACT]**
- `artifacts/_smoke_state/bot_state.json`: `last_bar_time` (UTC `+00:00`), `positions{}`, `saved_ts`. **[FACT]**
- `state/risk_state.db` — singleton `risk_state` row + `trade_history`, read via `RiskSupervisor.get_statistics()`. **[FACT]**
- `KILL_SWITCH` file: **not present** at report time (live loop would halt if present). **[FACT]**

### A.3 Metric dicts the UI binds to

- `DreamerV3Agent.train_step()` → `{world_model_loss, recon_loss, reward_loss, kl_loss, value_loss, policy_loss, entropy}` (entropy `None` when buffer cannot sample). **[FACT `models/dreamer_agent.py`]**
- `RealisticTradingEnv.step()` `info` → `{equity, position, pnl, return, cost, swap, drawdown, forced_close, blown_up}`; `episode_stats()` keys listed in Section 3.3. **[FACT `env/dreamer_trading_env.py`]**
- `EnsembleVoting.act()` info → `{agreement, consensus, uncertainty, votes, epistemic_uncertainty}`. **[FACT `models/ensemble.py`]**
- `MCTS.search()` → `{visit_counts, q_values, probs, visit_distribution, root_visits, ...}`. **[FACT `models/mcts.py`]**

### A.4 Environment capabilities

- **No GPU**: `torch.cuda.is_available()` is false in this environment; the dashboard must be CPU-first and must not assume CUDA (`--device auto` resolves to `cpu`). **[FACT]**
- Venv interpreter `.venv\\Scripts\\python.exe` **exists** at repo root — preflight check #6 relies on this. **[FACT]**
"""
        text = text.rstrip() + "\n" + appendix
        changed = True

    if changed:
        write(REPORT, text)
        lines = text.count("\n")
        print(f"REPORT UPDATED; lines={lines}")
    else:
        lines = text.count("\n")
        print(f"NO CHANGES NEEDED; lines={lines}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
