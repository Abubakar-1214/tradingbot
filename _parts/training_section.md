

---

## 3. What to show during training

Training is launched by the dashboard as a **child subprocess** (`python train/train_dreamer.py ...` etc. in the project venv). The dashboard does **not** run training in-process. **[ENGINEERING RECOMMENDATION]** All panels below read either (a) subprocess stdout/stderr, (b) files written by the training script (`artifacts/models/<run>/...`), or (c) the metrics the script would print — see the state-source mapping table in Section 6.6.

### 3.1 Training job list

| Column | Data source (verified) |
|---|---|
| Job id / run name | `--run-name` arg; artifact dir `artifacts/models/<run>/` |
| Train script + CLI | `train/train_dreamer.py`, `train/train_ppo.py`, `train/train_transformer.py`, `train/train_ensemble.py`, `train/train_adversarial.py`, `train/meta_train_dreamer.py`, `train/train_god_mode.py`, `train/train_ultimate_150.py` (exact `argparse` flags; see Section 5 presets) |
| Status (running/finished/failed/killed) | Supervisor-owned PID + exit code + sentinel files |
| Progress | For Dreamer: `steps` vs `--steps` (default 100,000), checkpoint files written every `--save-every` (default 10,000) as `checkpoint_<training_step>.pt` **[FACT `models/dreamer_agent.py` / `train/train_dreamer.py`]** |
| Started / duration | Supervisor record at spawn |

The list must render **at most one running job** (concurrency guard, Section 3.11). Historical jobs appear with their final state.

### 3.2 Live loss curves — the real `train_step` metrics dict

**Data source: `models/dreamer_agent.py` → `DreamerV3Agent.train_step()` return dict — exact keys [FACT]:**

```
{
  'world_model_loss': float,
  'recon_loss': float,
  'reward_loss': float,
  'kl_loss': float,
  'value_loss': float,
  'policy_loss': float,
  'entropy': float,
}
```

The dashboard must bind a line chart **one series per key** (7 series), x-axis = `training_step`. These are per-step values (not smoothed); the frontend should apply its own rolling mean for readability and keep raw values in the tooltip. **[ENGINEERING RECOMMENDATION]**

Additional sources per train script:

- `train/train_transformer.py` → `TransformerAgentWrapper.train_step()` returns `{loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction}` **[FACT `models/transformer_policy.py`]** — render these when the job is a transformer.
- PPO (`train/train_ppo.py`) uses SB3; expose its `rollout/ep_rew_mean` style scalar if the log exposes it (best-effort; the canonical curve for PPO is reward, not world-model loss). **[ENGINEERING RECOMMENDATION]**
- Adversarial (`train/train_adversarial.py`): `SelfPlayTrainer.history` has `trader_wins[]`, `mm_profits[]`, `epochs`, `trader_updates` **[FACT `models/adversarial_training.py`]** — show trader-reward vs market-maker-profit as twin series per epoch.
- Meta (`train/meta_train_dreamer.py`): `MAMLTrader.meta_train` returns `{meta_loss, support_loss}` per epoch **[FACT `models/meta_learning.py`]**.

### 3.3 Environment-interaction ticker

**Data source: `env/dreamer_trading_env.py` → `RealisticTradingEnv.step()` `info` dict — exact keys [FACT]:**

```
info = {
  "equity": self.equity,          # equity ratio (1.0 = start)
  "position": self.pos,           # 0 flat / 1 long / -1 short
  "pnl": self.equity / equity_before - 1.0,   # step P&L fraction
  "return": ret,                  # bar return (log close diff)
  "cost": cost / equity_before,   # fill cost fraction of pre-step equity
  "swap": swap,                   # overnight financing fraction
  "drawdown": self.drawdown,      # from peak equity
  "forced_close": forced_close,   # bool — SL/TP hit inside the bar
  "blown_up": blown_up,           # bool — episode ended on max-drawdown
}
```

Render as a scrolling ticker/table (latest first), one row per env `step`, with color coding (green equity-up / red equity-down; amber when `forced_close`; red flash when `blown_up`). **[ENGINEERING RECOMMENDATION]**

Also surface `RealisticTradingEnv.episode_stats()` after each episode ends **[FACT]**: `steps, equity, return_pct, max_drawdown_pct, trades, win_rate, sl_hits, tp_hits, costs_paid, swap_paid, reward_sharpe`.

### 3.4 Replay buffer usage

**Data source: `models/dreamer_agent.py` → `ReplayBuffer` [FACT]**: capacity `100_000`, `seq_len` default `64`. The buffer is a preallocated ring (`ptr`, `size`).

Panel: progress bar `size / capacity`, plus `seq_len`, plus a note when `sample()` returns `None` (insufficient valid sequences — training step is skipped, see `train_step` guard). Also render `prefill` progress separately: `train_dreamer.py` fills the buffer for `--prefill` (default 5000) random/agent steps before training starts **[FACT `train/train_dreamer.py`]**.

### 3.5 Regime labels

**Data source: `models/meta_learning.py` → `MarketRegimeGenerator` [FACT]**: labels are `trend_up`, `trend_down`, `range`, `high_vol`, `low_vol`, and `unknown` (before warm-up), computed causally (rolling stats shifted by 1). The companion `models/adversarial_training.py` self-play also produces manipulation events.

Panel: a timeline strip under the loss curves coloring bars by regime (5 colors + grey `unknown`); a per-regime bar chart of the current train split; during meta-training, show the task list `{name, label, start, end, support/query}` from `generate_regimes`. **[ENGINEERING RECOMMENDATION]**

### 3.6 Hyperparameter panel + "Launch new run" form

Show the resolved hyperparameters of the **running** job from its launch payload (Section 5), and the artifact's `manifest.json` `hyperparams` once written **[FACT `core/model_artifacts.py` ModelManifest.hyperparams]**.

The launch form (right side) lets the user compose a new run from presets (Section 5.3) with every field bound to a real CLI flag, validated client-side AND server-side before spawn (Section 4.1). Example bound fields for Dreamer **[FACT `train/train_dreamer.py` CLI]**:

| UI field | CLI flag | Default |
|---|---|---|
| Steps | `--steps` | 100000 |
| Prefill | `--prefill` | 5000 |
| Batch size | `--batch-size` | 16 |
| Train every | `--train-every` | 4 |
| Seq len | `--seq-len` | 64 |
| Window | `--window` | 64 |
| Embed dim | `--embed-dim` | 256 |
| Hidden dim | `--hidden-dim` | 512 |
| Stoch dim | `--stoch-dim` | 32 |
| Num categories | `--num-categories` | 32 |
| Horizon | `--horizon` | 15 |
| Train end | `--train-end` | 2022-01-01 |
| Data | `--data` | data/xauusd_h1.csv |
| Device | `--device` | auto |

### 3.7 Checkpoint timeline from `evaluation.json`

**Data source: `artifacts/models/<run>/evaluation.json` [FACT]** — schema verified:

```json
{
  "metrics": { "total_return_pct": ..., "cagr_pct": ..., "sharpe": ..., "max_dd_pct": ...,
               "trades": ..., "win_rate": ..., "exposure_pct": ..., "costs": ...,
               "buy_hold_return_pct": ..., "final_equity": ..., "bars": ..., "bars_per_year": ... },
  "thresholds": { "sharpe": 0.5, "max_dd_pct": 20.0, "trades": 20, "total_return_pct": 0.0 },
  "passed": false,
  "evaluated_at": "...", "test_start": "...", "test_end": "...", "contract_hash": "..."
}
```

Timeline panel: every artifact run (from `artifacts/models/`), each showing `evaluated_at`, sharpe / max_dd / trades / total_return vs thresholds, and a **`passed=false` chip → "NOT PROMOTABLE"** badge. **[FACT]** The promotion gate itself is `evaluate.promotion_gate()`: `sharpe >= 0.5 AND max_dd_pct <= 20.0 AND trades >= 20 AND total_return_pct > 0` **[FACT `train/evaluate.py`]**. Also show `manifest.json` `train_start`/`train_end`/`test_end` on the same timeline. Intra-run checkpoints (`checkpoint_<step>.pt`) are a secondary timeline under the run's artifact dir. **[ENGINEERING RECOMMENDATION]**

### 3.8 Logs tail

**Data source: the child subprocess's captured stdout/stderr** (the supervisor tees both to `logs/<job_id>.log` and streams the tail over `/ws/logs`). **[ENGINEERING RECOMMENDATION]** Show the last N lines with a "follow" toggle; preserve ANSI-free plain text. The dashboard never parses logs to fabricate metrics — log lines are display-only.

### 3.9 Stop / pause / resume / kill controls

- **Stop (clean):** supervisor writes a **sentinel file** (e.g. `logs/<job_id>.STOP`) that the training loop checks, then `terminate()` on the process group; the script's `save()` path persists a full checkpoint (`models/dreamer_agent.py` saves optimizers, RNG, return normalizer, training_step — P1 audit fix) **[FACT]**.
- **Pause:** same sentinel mechanism — the loop is told to stop stepping but keep the process alive; then **Resume** clears the sentinel. **[ENGINEERING RECOMMENDATION]** (Note: training scripts do not implement pause today; this is a recommended supervisor-level feature.)
- **Resume:** re-spawn with `--resume <checkpoint_path>`; the script validates dimensions (`agent.obs_dim != obs_dim or agent.action_dim != action_dim → ValueError`) **[FACT `train/train_dreamer.py`]**; the dashboard pre-validates via the manifest (`obs_dim == window*n_features + 5` **[FACT `core/model_artifacts.py`]**).
- **Kill:** hard kill of the process group (`taskkill /T /F /PID <pid>` on Windows), always as a last resort after stop.

### 3.10 Resource metrics

CPU % + RAM of the child PID (via `psutil` on the supervisor). **[FACT: no GPU available in this environment — `torch.cuda.is_available()` false; the dashboard must show CPU-only and *not* assume CUDA.]** Show device resolution (`--device auto` → cpu) from the job launch.

### 3.11 Concurrency guard

**Exactly one training job may run at a time.** The supervisor refuses a second `POST /api/jobs/start` while a PID is alive, returning a conflict. This is a hard invariant, not a UI hint. **[ENGINEERING RECOMMENDATION]** The UI disables the launch button while a job is running and shows the owning job id.
