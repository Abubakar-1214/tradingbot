"""Surgical fix #2 for frontend_dashboard_design.md (idempotent).

1. Add honest note in 6.3 that train_dreamer_mcts.py / train_meta.py do not exist.
2. Fix obs_dim 193 -> 197 in the 7.3 decision-feed JSON example (window 64 x
   n_features 3 + 5 = 197) and add a grounding note that real manifests follow
   the same formula (dreamer artifact: n_features 15 -> obs_dim 965).
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(os.path.dirname(ROOT), "frontend_dashboard_design.md")


def main():
    with io.open(REPORT, "r", encoding="utf-8") as f:
        text = f.read()
    changed = False

    # --- 1. Honest note in 6.3 -------------------------------------------
    anchor = "- `train/make_mcts_manifest.py` — assembles an MCTS-enabled model manifest for `dreamer_mcts` deployments. **[FACT]**"
    note = anchor + (
        "\n\n"
        "> **Honesty note (script inventory):** `train/train_dreamer_mcts.py` and `train/train_meta.py` "
        "**do not exist** in this repo (verified `dir /b train\\*.py`). They must never appear as launch "
        "targets. MCTS-enabled deployments are assembled from a standard Dreamer artifact via "
        "`train/make_mcts_manifest.py`; meta-training runs through `train/meta_train_dreamer.py`. "
        "**[FACT]**"
    )
    if anchor in text and "do not exist" not in text[text.find("### 6.3"):text.find("### 6.4")]:
        text = text.replace(anchor, note, 1)
        changed = True

    # --- 2. obs_dim 193 -> 197 -------------------------------------------
    old = '"inputs": {"obs_dim": 193, "window": 64, "n_features": 3, "account": [0, 0, 0, 0.01, 1.0]}'
    new = '"inputs": {"obs_dim": 197, "window": 64, "n_features": 3, "account": [0, 0, 0, 0.01, 1.0]}'
    if old in text:
        text = text.replace(old, new, 1)
        changed = True
    # Grounding note after the payload (idempotent)
    ground_anchor = '"fill": {"retcode": 10008, "retcode_name": "ORDER_PLACED", "price": 1944.12}, "equity": 100000.0}'
    ground_note = ground_anchor + (
        "\n\n"
        "Note: `obs_dim` always follows `window*n_features + 5` (account vector is 5 scalars). "
        "The example uses `64*3 + 5 = 197`; the real Dreamer artifact in this repo uses "
        "`n_features=15 -> obs_dim=965`, so the dashboard must read `obs_dim` from the active "
        "feature contract, not hard-code it. **[FACT `artifacts/models/dreamer_20261003T230304213562Z/manifest.json`]**"
    )
    if ground_anchor in text and "n_features=15" not in text:
        text = text.replace(ground_anchor, ground_note, 1)
        changed = True

    with io.open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"FIX2 {'APPLIED' if changed else 'NO CHANGE'}; lines={text.count(chr(10))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
