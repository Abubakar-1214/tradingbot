"""Consolidated final verification for frontend_dashboard_design.md (v2).

Checks every acceptance criterion and prints a pass/fail report + final stats.
Corrections vs v1: header strings use the REAL '## N.' format; obs_dim check is
precise ('"obs_dim": 193' absent, '"obs_dim": 197' present); preflight item count
scoped between '### 6.5' and '## 7.'.
Run: python _parts/verify_report.py
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(os.path.dirname(ROOT), "frontend_dashboard_design.md")


def read(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    text = read(REPORT)
    lines = text.splitlines()
    total = len(lines)
    nonempty = [l for l in lines if l.strip()]
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    # 1. File exists + line count within 600-900
    check("file exists", os.path.exists(REPORT))
    check("line count 600-900", 600 <= total <= 900, f"total={total} nonempty={len(nonempty)}")

    # 2. All 12 major section headers (real '## N.' format)
    headers = [
        "## 0. How to read this document",
        "## 1. Overview",
        "## 2. Tech-stack recommendation",
        "## 3. What to show during training",
        "## 4. What to show during trading",
        "## 5. How to control the system from the dashboard",
        "## 6. How to prepare the AI for training from the frontend",
        "## 7. Proposed API surface",
        "## 8. Non-functional requirements",
        "## 9. Risks & mitigations",
        "## 10. Roadmap mapping to the FINAL study experiments",
        "## 11. Verifiable acceptance criteria",
    ]
    for h in headers:
        check("header: " + h, any(h in l for l in lines))
    check("Appendix A present", any("## Appendix A" in l for l in lines))

    # 3. 8 presets mapped to real scripts (6.3)
    sec63 = text[text.find("### 6.3"):text.find("### 6.4")] if "### 6.3" in text else ""
    presets = [
        ("dreamer", "train_dreamer.py"),
        ("transformer", "train_transformer.py"),
        ("ppo", "train_ppo.py"),
        ("ensemble", "train_ensemble.py"),
        ("god-mode", "train_god_mode.py"),
        ("ultimate-150", "train_ultimate_150.py"),
        ("adversarial", "train_adversarial.py"),
        ("meta-train", "meta_train_dreamer.py"),
    ]
    for preset, script in presets:
        ok = f"`{preset}`" in sec63 and script in sec63
        check(f"preset {preset} -> {script}", ok)
    check("honest note: train_dreamer_mcts/train_meta do not exist", "do not exist" in sec63.lower())
    check("6.4 base-manifest selector", "### 6.4" in text and "base manifest" in text.lower() and "obs_dim == window*n_features + 5" in text)
    check("6.4 checkpoint selector", "--resume" in text and "checkpoint_" in text)
    check("6.5 preflight checklist", "### 6.5" in text)
    sec65 = text[text.find("### 6.5"):text.find("## 7.")] if "### 6.5" in text else ""
    preflight_items = re.findall(r"^\d+\.\s", sec65, flags=re.M)
    check("6.5 preflight >= 6 items", len(preflight_items) >= 6, f"items={len(preflight_items)}")

    # 4. 7 train_step metric keys
    for k in ["world_model_loss", "recon_loss", "reward_loss", "kl_loss", "value_loss", "policy_loss", "entropy"]:
        check("train_step key " + k, k in text)

    # 5. env info keys
    for k in ["equity", "position", "pnl", "return", "cost", "swap", "drawdown", "forced_close", "blown_up"]:
        check("env info key " + k, k in text)

    # 6. All 13 RiskSupervisor reason strings verbatim
    reasons = [
        "APPROVED", "CIRCUIT_BREAKER", "HALTED", "MAX_DRAWDOWN", "POSITION_TOO_LARGE",
        "TOO_MANY_LOSSES", "HIGH_VOLATILITY", "CORRELATION_GUARD", "EVENT_RISK",
        "MAX_TRADES", "COOLDOWN", "SPREAD_TOO_WIDE", "MARKET_CLOSED",
    ]
    for r in reasons:
        check("risk reason " + r, r in text)

    # 7. API surface: >= 12 REST endpoints, 5 WS channels, 4 JSON examples
    endpoints = [
        "/api/status", "/api/jobs", "/api/jobs/start", "/api/jobs/{id}/stop",
        "/api/jobs/{id}/kill", "/api/jobs/{id}/resume", "/api/jobs/{id}/pause",
        "/api/jobs/{id}/unpause", "/api/models", "/api/models/activate",
        "/api/risk/state", "/api/config/env", "/api/prep/run", "/api/prep/leakcheck",
        "/api/state/{panel}", "/api/reconciliation", "/api/live/start",
        "/api/live/stop", "/api/live/killswitch",
    ]
    found_ep = [e for e in endpoints if e in text]
    check("REST endpoints >= 12", len(found_ep) >= 12, f"found={len(found_ep)}/19")
    ws = ["/ws/training", "/ws/live/decisions", "/ws/live/risk", "/ws/logs", "/ws/checkpoints"]
    found_ws = [w for w in ws if w in text]
    check("WS channels >= 4", len(found_ws) >= 4, f"found={len(found_ws)}/5")
    json_ex = ["preset", "world_model_loss", "current_equity", "peak_equity"]
    check("JSON examples present", all(j in text for j in json_ex))
    check("state-source mapping table", "state-source mapping" in text.lower() or "state source" in text.lower())

    # 8. HARD SAFETY INVARIANT (bolded)
    check("HARD SAFETY INVARIANT", "HARD SAFETY INVARIANT" in text and "frontend never calls broker" in text.lower())

    # 9. Honest state: no promoted model
    check("honest: no promoted model", "passed: false" in text or "passed:false" in text or "NOT PROMOTABLE" in text)
    check("HONEST-STATE RULE", "HONEST-STATE RULE" in text)

    # 10. E1-E10 roadmap rows
    for e in ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"]:
        check("roadmap " + e, ("| " + e + " " in text) or ("**" + e + "**" in text) or (e + " — " in text))

    # 11. Acceptance criteria >= 8 items (numbered in section 11)
    sec11 = text[text.find("## 11"):] if "## 11" in text else ""
    crit_items = re.findall(r"^\d+\.\s", sec11, flags=re.M)
    check("acceptance criteria >= 8", len(crit_items) >= 8, f"items={len(crit_items)}")
    check("acceptance item 6 uses real scripts", "meta_train_dreamer.py" in sec11 and "make_mcts_manifest.py" in sec11)

    # 12. Claim-marker counts
    counts = {
        "FACT": len(re.findall(r"\*\*\[FACT\]\*\*", text)),
        "RESEARCH FINDING": len(re.findall(r"\*\*\[RESEARCH FINDING\]\*\*", text)),
        "ENGINEERING RECOMMENDATION": len(re.findall(r"\*\*\[ENGINEERING RECOMMENDATION\]\*\*", text)),
        "UNPROVEN": len(re.findall(r"\*\*\[UNPROVEN\]\*\*", text)),
    }
    check("claim markers > 0", sum(counts.values()) > 0, str(counts))

    # 13. obs_dim consistency (precise)
    check("obs_dim no 193", '"obs_dim": 193' not in text)
    check("obs_dim = 197 in example", '"obs_dim": 197' in text)
    check("obs_dim formula grounding note", "n_features=15 -> obs_dim=965" in text)

    # Report
    print("=" * 70)
    print(f"FILE: {REPORT}")
    print(f"TOTAL LINES: {total} | NON-EMPTY: {len(nonempty)} | BYTES: {len(text.encode('utf-8'))}")
    print(f"CLAIM MARKERS: {counts}")
    print("=" * 70)
    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{status}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    print("=" * 70)
    print(f"RESULT: {'ALL PASS' if failed == 0 else str(failed) + ' FAILED'}  ({len(results)} checks)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
