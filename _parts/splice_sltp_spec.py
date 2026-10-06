"""Subtask 10: idempotent splice of the SL/TP dual-mode spec into
frontend_dashboard_design.md.

Four insertions, each skipped if its marker already exists:
  A. 3.6 training hyperparameter table  -> `--sl-tp-model` row + dual-mode note
     (sl_tp_action checkbox + bucket presets {0.005,0.01,0.02,0.03,0.05}).
  B. 4.4 decision-feed table            -> stage-10 SL/TP source row.
  C. 4.4 decision-feed section          -> MODEL_DECIDED vs ATR_RULES badge note
     + honest guard note (no silent discard).
  D. 4.6 risk panel                     -> SL/TP fraction bounds clamp note.

CRLF preserved. Run with a file-I/O interpreter:
    E:\\python_3.11.9_installed\\python.exe _parts/splice_sltp_spec.py
"""
import io

TARGET = r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader\frontend_dashboard_design.md"

# --------------------------------------------------------------------------- #
# Content blocks (LF newlines; converted to the target's newline style)
# --------------------------------------------------------------------------- #

BLOCK_A = """| SL/TP action mode | `--sl-tp-model` | off -> legacy 3-action env (ATR rules SL/TP); on -> 75-action composite env (model decides SL/TP buckets) |

> **SL/TP dual-mode training spec [FACT `train/train_dreamer.py` + `env/dreamer_trading_env.py` + `core/model_artifacts.py` + `train/evaluate.py`]:**
> - **`sl_tp_action` checkbox** maps to the `--sl-tp-model` CLI flag. Checked -> the env is constructed with `sl_tp_action=True`, `action_dim=75` (composite: direction 3 x SL bucket 5 x TP bucket 5), and the artifact `manifest.json` records `sl_tp_mode: "model"`. Unchecked -> legacy path unchanged: `action_dim` is 3 (or 2 long-only) and the manifest records `sl_tp_mode: "rules"`.
> - **Bucket presets {0.005, 0.01, 0.02, 0.03, 0.05}** for both SL and TP — price-fraction levels matching the env's existing fraction semantics. Composite index decode: `idx = dir*25 + sl_idx*5 + tp_idx`.
> - **`evaluate_policy` infers the mode from the policy**: `action_dim > 3` -> the evaluation env is built with `sl_tp_action=True` [FACT `train/evaluate.py`], so a 75-action artifact is never evaluated against the 3-action env.
> - **Manifest guard**: `sl_tp_mode="model"` requires `action_dim == 75`; any mismatch is rejected at artifact load (`core/model_artifacts.py`) [FACT].
"""

BLOCK_B = """
| 10. SL/TP source | `MODEL_DECIDED` vs `ATR_RULES` badge | `live/model_signal.py` decode -> `Decision.sl_frac/tp_frac` (model mode); `live/trade_executor.py::entry_sl_tp` ATR path (rules mode) **[FACT]** |"""

BLOCK_C = """
**SL/TP source badge — MODEL_DECIDED vs ATR_RULES [FACT `live/model_signal.py` + `live/decision_engine.py` + `live/trade_executor.py`]:**

Every decision-feed item shows the SL/TP source of the active run:

- **`MODEL_DECIDED`** (`SLTP_MODE=model`, 75-action artifact): the model's composite action encodes SL/TP bucket fractions; `ModelSignalSource` decodes `idx = dir*25 + sl_idx*5 + tp_idx` into `sl_frac`/`tp_frac`; `Decision` carries them through `DecisionEngine.filter` untouched (no silent discard); `TradeExecutor` converts them to order prices `entry*(1-+frac)`, clamped to the configured bounds before attach.
- **`ATR_RULES`** (`SLTP_MODE=rules`, legacy 2/3-action artifact): the `Decision` carries `sl_frac=None`/`tp_frac=None` and the executor uses the existing ATR path (`entry -+ atr*sl_atr_mult` / `entry + atr*tp_atr_mult`) exactly as before. Explicit model fractions are ignored in rules mode.

The badge is driven by `cfg.behavior.sl_tp_mode` plus the active manifest's `sl_tp_mode` — never by a UI guess.

> **Honest guard note (non-negotiable):** the `SLTP_MODE` config + manifest `sl_tp_mode` guard (`core/config.py::enforce_sltp_compatibility`, called first from `live/live_trade_mt5.py::enforce_model_promotion_gate`) raises a clear `ValueError` at startup when a **model-trained artifact** (`sl_tp_mode="model"`) is configured with `SLTP_MODE=rules`, and also when `SLTP_MODE=model` is configured with a **rules-trained artifact**. A model's trained SL/TP output is **never silently discarded** — the system refuses to start rather than degrade. The dashboard must surface this state: if the mode badges disagree with the active artifact, the "start live" button stays disabled with the guard error message. **[FACT + ENGINEERING RECOMMENDATION]**
"""

BLOCK_D = """
**SL/TP fraction bounds clamp [FACT `core/config.py` + `live/trade_executor.py`]:**

In `SLTP_MODE=model` the executor clamps the model-decided fractions to the configured bounds before converting to prices: `sl_tp_min_frac <= sl_frac <= sl_tp_max_frac` and `tp_min_frac <= tp_frac <= tp_max_frac` (defaults 0.005 / 0.05 for both SL and TP). The risk panel shows the active bounds and the clamped result per order — e.g. a model SL request of 0.50 renders as **0.05 (clamped)** and a TP of 0.001 as **0.005 (clamped)**. Clamping happens on the *fraction*, then the order prices attach as `entry*(1-+frac)`. **[FACT]**
"""


def main() -> None:
    with io.open(TARGET, "r", encoding="utf-8", newline="") as f:
        raw = f.read()
    nl = "\r\n" if "\r\n" in raw else "\n"

    def conv(s: str) -> str:
        return s.replace("\n", nl)

    edits = [
        # (marker, anchor, insert_after_anchor, block)
        (
            "| SL/TP action mode |",
            "| Device | `--device` | auto |",
            BLOCK_A,
        ),
        (
            "| 10. SL/TP source |",
            "| 9. Equity / reward | equity after fill; reward signal | "
            "`mock_state.json` balance / `bot_state.json`; env info `equity` **[FACT]** |",
            BLOCK_B,
        ),
        (
            "SL/TP source badge — MODEL_DECIDED vs ATR_RULES",
            "### 4.5 Brain panel (per model type)",
            BLOCK_C,
        ),
        (
            "SL/TP fraction bounds clamp",
            "### 4.7 Economic calendar event feed",
            BLOCK_D,
        ),
    ]

    text = raw
    for marker, anchor, block in edits:
        marker_n = conv(marker)
        if marker_n in text:
            print(f"SKIP (already present): {marker}")
            continue
        anchor_n = conv(anchor)
        if anchor_n not in text:
            print(f"ERROR anchor not found: {anchor!r}")
            sys_exit = 2
            raise SystemExit(2)
        text = text.replace(anchor_n, anchor_n + conv(block), 1)
        print(f"INSERTED: {marker}")

    with io.open(TARGET, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"target now {text.count(nl) + 1} lines")


if __name__ == "__main__":
    main()
