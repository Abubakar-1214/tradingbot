
### 6.3 Hyperparameter preset library

The "Launch new run" form starts from a named preset. Each preset maps to a **real training script** (verified in `train/`), ships a resolved flag list, validation rules, the expected artifact layout, and the honest evaluation step (`train/evaluate.py` → `evaluation.json` → `passed` chip). **[FACT + ENGINEERING RECOMMENDATION]**

| Preset (exact name) | Real train script (verified) | Notable defaults / notes |
|---|---|---|
| `dreamer` | `train/train_dreamer.py` | World-model + actor-critic; `--steps 100000 --prefill 5000 --batch-size 16 --seq-len 64 --window 64 --embed-dim 256 --hidden-dim 512 --stoch-dim 32 --num-categories 32 --horizon 15` |
| `transformer` | `train/train_transformer.py` | Patched-transformer policy; train-step metrics `{loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction}` |
| `ppo` | `train/train_ppo.py` | SB3 PPO baseline; scalar reward curves |
| `ensemble` | `train/train_ensemble.py` | `num_models=5` bagging; agreement / consensus / uncertainty |
| `god-mode` | `train/train_god_mode.py` | Composite heavy pipeline preset (project-specific) |
| `ultimate-150` | `train/train_ultimate_150.py` | Long-horizon budgeted preset (project-specific) |
| `adversarial` | `train/train_adversarial.py` | Self-play trader vs market-maker; requires a base manifest (staged) |
| `meta-train` | `train/meta_train_dreamer.py` | MAML regime meta-training; requires a base manifest (staged) |

Supporting tools surfaced alongside the presets:

- `train/evaluate.py` — the promotion-gate evaluator that writes `evaluation.json` (thresholds `sharpe >= 0.5`, `max_dd_pct <= 20`, `trades >= 20`, `total_return_pct > 0`). **[FACT `train/evaluate.py`]**
- `train/make_mcts_manifest.py` — assembles an MCTS-enabled model manifest for `dreamer_mcts` deployments. **[FACT]**

### 6.4 Base-manifest / checkpoint selector for staged pipelines

Adversarial and meta-train pipelines are *staged*: they consume a previously trained model as their starting point. The form therefore includes:

- **Base-manifest selector:** a dropdown of `artifacts/models/<run>/manifest.json` entries, each rendered with its `evaluation.json` honesty chip (`passed: false` → NOT PROMOTABLE). The backend validates the base manifest before spawn: `MANIFEST_VERSION=1`, model file exists, contract file exists, **contract hash matches** `feature_contract.json`, and `obs_dim == window*n_features + 5`; a staged run without a compatible base is refused with a precise error. **[FACT `core/model_artifacts.py`]**
- **Checkpoint selector:** optional `--resume <checkpoint_path>` from the base run's `checkpoint_<step>.pt` files (e.g. `checkpoint_40000.pt`); the same dimension pre-check applies on resume. **[FACT `train/train_dreamer.py --resume`]**
- The selected base run's hyperparameters are shown read-only next to the selector so the user can align the staged run's flags (e.g. same `--window` / `--seq-len`). **[ENGINEERING RECOMMENDATION]**

### 6.5 Preflight checklist

Computed by the dashboard before every launch; the Launch button stays disabled until every item passes. **[ENGINEERING RECOMMENDATION]**

1. Data CSV exists and is non-empty; `--train-end` split lies inside the data range.
2. Feature contract exists for this window; contract hash matches when reusing an artifact.
3. All hyperparameter fields pass schema/range validation (backend re-validates on POST).
4. Base model selected when the preset is `adversarial` or `meta-train` (staged requirement).
5. Disk space sufficient for checkpoints + logs (supervisor reports free space, e.g. ≥ 2 GB).
6. Venv python resolves (`.venv\Scripts\python.exe` exists — verified present **[FACT]**).
7. No training job currently running (concurrency guard) and the artifact dir name is free.
8. TRADING_MODE/demo consistency: preparing data or training is allowed in demo; starting a live run additionally requires all live gates. **[FACT `core/config.py::check_live_gates`]**
