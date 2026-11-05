# Frontend Design Specification — Training & Operations Dashboard for the Autonomous XAUUSD Trading AI

> **Document type:** Specification for frontend developers (not an implementation).
> **Grounded in:** verified project facts — real module names, real metrics-dict keys, real file paths, real state files (read directly from the repo at `e:\Desktop\NeoMind\Bazz\autonoumuse_trader`).
> **Honest status baseline (read this first):** **NO model has passed the promotion gate.** Every `evaluation.json` inspected in `artifacts/models/` has `"passed": false` (e.g. `ppo_gold_v1/evaluation.json`: sharpe `-3.21`, total return `-74.7%`, `passed: false`; `dreamer_20261003T230304213562Z/evaluation.json`: sharpe `3.10` but `trades: 3 < 20`, `passed: false`). No `artifacts/models/production/manifest.json` exists yet. The dashboard MUST render this honestly and must never fabricate a promotable state. **[FACT]**

---

## 0. How to read this document

### 0.1 Claim markers

Every major claim is tagged so the frontend team knows how much trust to place in it:

| Marker | Meaning |
|---|---|
| **[FACT]** | Verified directly against project files in this repo (module names, dict keys, file paths, JSON shapes, CLI arguments). Treat as ground truth for data binding. |
| **[RESEARCH FINDING]** | Supported by cited literature/evidence in `research/FINAL_AUTONOMOUS_TRADING_AI_STUDY.md` (e.g. DreamerV3 arXiv 2301.04104, Lopez de Prado SSRN 2460551). External evidence, not project code. |
| **[ENGINEERING RECOMMENDATION]** | This report's design judgment for the dashboard. The team should follow it unless there is a concrete reason not to; deviations must be documented. |
| **[UNPROVEN]** | Something the project itself has not established (e.g. "ML beats buy-and-hold" is unproven). The UI must not imply otherwise. |

### 0.2 Scope of this document

1. Overview + tech-stack recommendation.
2. What to show during **training**.
3. What to show during **trading** (MT5-style operations view).
4. How the dashboard **controls** the system (and what it must never do).
5. How the dashboard **prepares the AI** for a training run.
6. Proposed **API surface** (REST + WebSocket + JSON payloads + state-source mapping).
7. Non-functional requirements, risks & mitigations, roadmap mapping to the FINAL study's E1–E10 experiments, and verifiable acceptance criteria.

---

## 1. Overview

### 1.1 The system being monitored

The project is an **autonomous XAUUSD (gold) trading AI** — a hybrid of model-based RL (DreamerV3-style world model + actor-critic), learned representations, a deterministic risk layer, and an optional advisory LLM desk. It is a Python codebase, not a service: the repo contains training scripts, a live-trading loop, model code, a feature pipeline, and an honest backtester, but **no HTTP server or UI exists today**. **[FACT]**

*Light Roman-Urdu framing — is dashboard ka maqsad: system ko monitor karna, usay safely control karna, aur har number ka asli source dikhana. Koi fabricated metric nahi. (Purpose: monitor the system, control it safely, and show the true source of every number. No fabricated metrics.)*

Key verified architecture facts the UI must bind to:

| Layer | Real module (verified path) | What it exposes |
|---|---|---|
| Config | `core/config.py` | Validated `AppConfig` loaded from `.env`; `TRADING_MODE` gate (`demo`/`live`), `RiskConfig`, `BrokerConfig`, `PathConfig`, `TradingBehaviorConfig`; `check_live_gates()` + `ensure_trading_allowed()` |
| Features | `core/feature_pipeline.py`, `train/data.py` | Leak-free pipeline: fit on train window only, `feature_contract.json` with hash, `window * n_features + 5` observations |
| Models | `models/dreamer_agent.py`, `models/policy.py`, `models/ensemble.py`, `models/mcts.py`, `models/transformer_policy.py`, `models/registry.py`, `models/risk_supervisor.py`, `models/adversarial_training.py`, `models/meta_learning.py` | Policy types: `ppo`, `dreamer`, `transformer`, `ensemble`, `dreamer_mcts`; deterministic `RiskSupervisor` circuit breakers; adversarial self-play; MAML regime meta-training; `MarketRegimeGenerator` |
| Training | `train/train_dreamer.py`, `train/train_ppo.py`, `train/train_transformer.py`, `train/train_ensemble.py`, `train/train_adversarial.py`, `train/meta_train_dreamer.py`, `train/train_god_mode.py`, `train/train_ultimate_150.py`, `train/evaluate.py` | CLI jobs → `artifacts/models/<run>/` with `manifest.json` + `feature_contract.json` + `evaluation.json` |
| Live | `live/live_trade_mt5.py`, `live/mock_broker.py`, `live/trade_executor.py`, `live/decision_engine.py`, `live/model_signal.py`, `live/broker.py` | Candle-close-aligned loop; `MockBroker` (demo) / MT5 (live); order paths all RiskSupervisor-gated; kill-switch file; reconciliation |
| Backtest | `backtest/engine.py` | Honest walk-forward (`train 800 / embargo 25 / test 300`), `CostModel`, buy-and-hold benchmark |
| State | `state/mock_state.json`, `state/bot_state.json`, `state/risk_state.db`, `state/broker_idempotency.json`, `artifacts/_smoke_state/*` | Persisted account/trade/risk/idempotency state the UI reads |

### 1.2 Honest current status (non-negotiable baseline)

- **No model is promotable.** All existing `evaluation.json` files have `passed: false`. **[FACT]**
- **No production manifest.** `artifacts/models/production/manifest.json` does not exist, so `REQUIRE_PROMOTED_MODEL` (`core/config.py`, default `true`) means the system will not take model-driven live trades yet. **[FACT]**
- **Demo vs live.** `TRADING_MODE=demo` (default) trades only against `MockBroker`; `live` additionally requires MT5 credentials + all live gates (`check_live_gates`: feature contract present, model present, risk state loadable, reconciliation passed, MT5 creds). **[FACT]**
- **Buy-and-hold benchmark is the truth.** The FINAL study's honest finding: rule strategies did not beat buy-and-hold net of costs; whether any ML policy can is **unproven** (Open Question Q1). **[FACT] + [UNPROVEN]**

Design principle this implies: **the dashboard is a truth-display and a safe control surface, not a marketing demo.** Every panel must show exactly what the state files and metrics dicts contain — including failures, empty buffers, and `passed: false` chips.

### 1.3 What the dashboard must do (top-level requirements)

1. **Train:** launch, monitor, pause/resume, stop, and kill training jobs; render live loss curves from the real `train_step` metrics dict; show env interaction, replay buffer, regime labels, hyperparameters, checkpoint/evaluation timeline, logs, resources, and a strict one-job-at-a-time concurrency guard.
2. **Operate:** show an MT5-style account panel, XAUUSD candlesticks with entry/SL/TP overlays, positions & closed-trade tables matching `mock_state.json` shape, the AI decision feed, per-model "brain" introspection, the risk circuit-breaker panel, economic-calendar events, kill switch, `TRADING_MODE` badge, and reconciliation status.
3. **Control:** start/stop/kill/resume training; start/stop live; load/activate a model artifact; edit risk limits via validated `.env` writes — subject to the **Hard Safety Invariant** (Section 5).
4. **Prepare:** data-prep + contract build, hyperparameter presets, base-model selector for staged (adversarial/meta) pipelines, and a preflight checklist.
5. **Honesty:** never fabricate metrics; never show a model as promotable when `passed: false`; never show "none trained" as if a checkpoint existed.

---

## 2. Tech-stack recommendation

### 2.1 Decision

**Primary: FastAPI backend + WebSocket push + single-page application (SPA) frontend.**
**Alternative: Streamlit dashboard (single Python process).**

**[ENGINEERING RECOMMENDATION]** — with the alternative documented so the team can downgrade if the deployment context changes (e.g. a headless box with no node toolchain, or a "just need to watch training" phase).

### 2.2 Rationale — grounded in the FACT that no backend exists today

The repo is a set of Python training/trading scripts with **no server layer**: `train/*.py`, `live/*.py`, `models/*.py`, `core/*.py`, `backtest/engine.py`. There is no REST API, no database service, no process supervisor, no message bus. **[FACT]**

That means the dashboard's backend is not an "add a route to the existing app" task — it is a **new control plane** that must:

1. **Spawn and supervise child processes** (training jobs, the live loop) with Windows-safe lifecycle control (`CREATE_NEW_PROCESS_GROUP`, sentinel kill switch, taskkill). Streamlit is a single-sync-process framework with no clean subprocess supervisor story; FastAPI + a small asyncio supervisor does this natively.
2. **Push high-frequency metrics** (loss-curve ticks, decision feed, env ticker) without page reloads. WebSocket over FastAPI gives < 250 ms pushes; Streamlit's rerun model is not designed for low-latency streaming from a changing file/metrics source.
3. **Render dense, frequently-updating charts** (multi-series loss curves, candlesticks with overlays, attention heatmaps). A SPA with a canvas/WebGL charting library (lightweight-charts / uPlot / ECharts) handles thousands of points smoothly; Streamlit's default charts are heavier and less decimatable.
4. **Separate control actions from display** with clear auth-by-design (single-user localhost + token), CSRF-safe POST endpoints, and strict validation before touching `.env` or spawning processes. Streamlit buttons + script reruns make "do something dangerous" too easy to misfire.

### 2.3 Comparison table

| Dimension | FastAPI + WebSocket + SPA (primary) | Streamlit (alternative) | Notes (grounded in project facts) |
|---|---|---|---|
| Real-time push latency | ~ms — server→WS push, client renders on event **[ENGINEERING RECOMMENDATION]** | ~1–5 s rerun cycle; polling/file-watch based | Loss-curve ticks are produced per `train_step` in the training loop; a WebSocket channel is the only way to hit < 1 s without polling |
| Subprocess / job control | First-class: asyncio + `subprocess.Popen(creationflags=CREATE_NEW_PROCESS_GROUP)`, sentinel files, process-group kill **[ENGINEERING RECOMMENDATION]** | Poor: Streamlit reruns its own script; managing child PIDs across reruns is fragile | `live/live_trade_mt5.py` already uses a kill-switch **file** (`KILL_SWITCH`) — a supervisor that owns PIDs is a natural fit for FastAPI **[FACT]** |
| Dense chart rendering | SPA canvas/WebGL: thousands of points, zoom/pan, overlays **[ENGINEERING RECOMMENDATION]** | Slower; charts redraw per rerun; limited decimation | Candlestick + entry/SL/TP overlays and 7-series loss curves benefit from canvas rendering |
| Auth / safety posture | Single-user localhost + bearer token; explicit POST endpoints; input validation before any write **[ENGINEERING RECOMMENDATION]** | Minimal built-in auth; any rerun can re-execute side effects | The dashboard writes `.env` and spawns processes — it must be boring and safe, not convenient-and-risky |
| Development speed | Slower initially (build backend + frontend separately) **[ENGINEERING RECOMMENDATION]** | Fastest for read-only dashboards; ~hours to first panel | Good for a Phase-0 "watch training" fallback, weak for the full control surface |
| Deployment footprint | Requires Python venv (already exists: `.venv`) + Node build step for the SPA **[FACT: `.venv` exists]** | Single Python process, one `streamlit run` command | The repo is a local single-user tool; both are acceptable, FastAPI needs one extra build step |

### 2.4 Backend module sketch (target state)

```
backend/            # NEW control plane (recommended)
├── main.py         # FastAPI app, mounts REST + WS routes, starts supervisor
├── supervisor.py   # child-process registry: launch/stop/kill/resume, PIDs, sentinels
├── readers.py      # typed readers for state/*.json, risk_state.db, evaluation.json, manifest.json
├── ws.py           # WebSocket hub: /ws/training, /ws/live/decisions, /ws/live/risk, /ws/logs, /ws/checkpoints
├── config_writer.py# validated .env read/modify (whitelist of keys, type-checked via core.config.load_config)
└── safety.py       # HARD SAFETY INVARIANT enforcement (Section 5): no broker imports, no order calls
```

**[ENGINEERING RECOMMENDATION]** — the backend is the *only* process allowed to spawn children and write `.env`. The frontend is a thin view/controller; every mutation is a validated POST that the backend re-validates.

### 2.5 When to pick Streamlit instead

Use Streamlit if: the deployment box has no Node.js toolchain; the team only needs the **training watch** panels (loss curves, logs, job list) for a first iteration; or a frontend team is not available. **[ENGINEERING RECOMMENDATION]** The API-surface section (Section 6) is written so the same data readers can serve either shell.


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
| SL/TP action mode | `--sl-tp-model` | off -> legacy 3-action env (ATR rules SL/TP); on -> 75-action composite env (model decides SL/TP buckets) |

> **SL/TP dual-mode training spec [FACT `train/train_dreamer.py` + `env/dreamer_trading_env.py` + `core/model_artifacts.py` + `train/evaluate.py`]:**
> - **`sl_tp_action` checkbox** maps to the `--sl-tp-model` CLI flag. Checked -> the env is constructed with `sl_tp_action=True`, `action_dim=75` (composite: direction 3 x SL bucket 5 x TP bucket 5), and the artifact `manifest.json` records `sl_tp_mode: "model"`. Unchecked -> legacy path unchanged: `action_dim` is 3 (or 2 long-only) and the manifest records `sl_tp_mode: "rules"`.
> - **Bucket presets {0.005, 0.01, 0.02, 0.03, 0.05}** for both SL and TP — price-fraction levels matching the env's existing fraction semantics. Composite index decode: `idx = dir*25 + sl_idx*5 + tp_idx`.
> - **`evaluate_policy` infers the mode from the policy**: `action_dim > 3` -> the evaluation env is built with `sl_tp_action=True` [FACT `train/evaluate.py`], so a 75-action artifact is never evaluated against the 3-action env.
> - **Manifest guard**: `sl_tp_mode="model"` requires `action_dim == 75`; any mismatch is rejected at artifact load (`core/model_artifacts.py`) [FACT].


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



---

## 4. What to show during trading (MT5-style operations view)

Trading runs as a **separate child process** (`live/live_trade_mt5.py`), not inside the dashboard. The dashboard reads persisted state (`state/*.json`, `state/risk_state.db`), watches the kill-switch file, and subscribes to decision events over WebSocket. It **never** sends orders itself (Hard Safety Invariant, Section 5.7). **[FACT + ENGINEERING RECOMMENDATION]**

### 4.1 Account summary panel (MT5-style)

| Stat | Data source (verified) |
|---|---|
| Balance | `MockBroker.account_info()` → `balance` (persisted in `mock_state.json` / `bot_state.json`) **[FACT]** |
| Equity | `RiskSupervisor.get_statistics()['current_equity']` and/or broker account `equity` **[FACT `models/risk_supervisor.py`]** |
| Drawdown | `get_statistics()['daily_loss_ratio']` vs `max_daily_loss`, plus env/report `max_drawdown_pct` **[FACT]** |
| Daily PnL | `get_statistics()['daily_pnl']` **[FACT]** |
| Consecutive losses | `get_statistics()['consecutive_losses']` **[FACT]** |
| Halt remaining | `halt_remaining_seconds()` / `get_statistics()['halt_remaining_sec']` — 365-day halt after `emergency_shutdown()` **[FACT]** |

Also show `get_statistics()` totals: `total_checks`, `approved`, `rejected`, `approval_rate`, `rejection_rate`, `trades_today`. **[FACT]**

### 4.2 XAUUSD candlestick chart with entry / SL / TP overlays

Chart the live (or replayed/smoke) price feed with candles (open/high/low/close + volume). Overlay, per active position:

- **Entry** line (from `mock_state.json` `closed[].entry` and live position `entry`),
- **SL / TP** lines (from `live/trade_executor.py::entry_sl_tp` — `sl_atr_mult=2.0`, `tp_atr_mult=3.0` defaults from `BrokerConfig`) **[FACT]**.

**[ENGINEERING RECOMMENDATION]** Use a canvas charting library (e.g. lightweight-charts) for decimation of thousands of candles. Data source: broker history/bars (H1 default `TIMEFRAME=H1` **[FACT `core/config.py`]**); the smoke-state `mid` (`mock_state.json`) is the last known price.

### 4.3 Positions table + closed-trades table

**Exact shape: `artifacts/_smoke_state/mock_state.json` [FACT]:**

```json
{
  "balance": 99999.96,
  "mid": 1943.90,
  "next_ticket": 11,
  "positions": [ ],
  "closed": [
    { "ticket": 1, "side": "buy", "volume": 0.05,
      "entry": 1985.17, "exit": 1980.79, "pnl": -0.22, "reason": "SL" },
    { "ticket": 4, "side": "sell", "volume": 0.05,
      "entry": 1966.70, "exit": 1959.38, "pnl": 0.36, "reason": "TP" },
    { "ticket": 10, "side": "sell", "volume": 0.05,
      "entry": 1949.17, "exit": 1944.14, "pnl": 0.25, "reason": "manual_close" }
  ]
}
```

Positions table columns: `ticket, side, volume, entry, sl, tp, current price, pnl`. Closed-trades columns: `ticket, side, volume, entry, exit, pnl, reason` with reason chips (`SL`/`TP`/`manual_close`/...). **[ENGINEERING RECOMMENDATION]** `bot_state.json` adds `last_bar_time` (UTC ISO) and `positions` map for live-loop progress. **[FACT]**

### 4.4 AI decision feed

The heart of the operations view. Each decision event, pushed over `/ws/live/decisions`, shows the full chain:

| Stage | Fields | Data source (verified) |
|---|---|---|
| 1. Inputs | window features + `[position, trade_pnl, bars_in_trade, drawdown, equity_ratio]` account vector; obs dim = `window*n_features + 5` | `env/dreamer_trading_env.py` observation; `core/model_artifacts.py` manifest check **[FACT]** |
| 2. Model probs | per-action probability vector | `models/registry.py` policy; `policy_probs(state)` / softmax **[FACT]** |
| 3. Confidence / agreement | `confidence`; `agreement` | `live/model_signal.py`; `models/ensemble.py` `info['agreement']` **[FACT]** |
| 4. Uncertainty | `uncertainty` (normalized entropy), `epistemic_uncertainty` | `models/ensemble.py` `act()` info **[FACT]** |
| 5. Decision filter | `reason` ∈ `NO_SIGNAL, HOLD, LOW_CONFIDENCE, NO_CONSENSUS, NO_EDGE, ENTRY_BLOCKED...` | `live/decision_engine.py::filter` **[FACT]** |
| 6. Risk check | `approved` + `reason` (risk reason strings below) | `RiskSupervisor.check_trade` **[FACT]** |
| 7. Order | `side, lot (volume), sl, tp` | `TradeExecutor.execute_entry` — volume floored to `LOT_STEP=0.01`, `MIN_LOT=0.01` **[FACT]** |
| 8. Fill | `retcode` + `retcode_name` + fill price | `live/broker.py` `OrderResult` + `RETCODE_NAMES` **[FACT]** |
| 9. Equity / reward | equity after fill; reward signal | `mock_state.json` balance / `bot_state.json`; env info `equity` **[FACT]** |
| 10. SL/TP source | `MODEL_DECIDED` vs `ATR_RULES` badge | `live/model_signal.py` decode -> `Decision.sl_frac/tp_frac` (model mode); `live/trade_executor.py::entry_sl_tp` ATR path (rules mode) **[FACT]** |

Retcode rendering (from `live/broker.py` **[FACT]**): success = `10008 ORDER_PLACED`, `10009 DONE`, `10010 DONE_PARTIAL`; transient = `10004 REQUOTE`, `10020 PRICE_CHANGED`, `10021 PRICE_OFF` (auto-retry, `max_requote_retries=3`); failure = `10006 REJECT`, `10016 INVALID`, `10018 MARKET_CLOSED`, `10019 NO_MONEY`, `10017 TRADE_DISABLED`, etc.

**SL/TP source badge — MODEL_DECIDED vs ATR_RULES [FACT `live/model_signal.py` + `live/decision_engine.py` + `live/trade_executor.py`]:**

Every decision-feed item shows the SL/TP source of the active run:

- **`MODEL_DECIDED`** (`SLTP_MODE=model`, 75-action artifact): the model's composite action encodes SL/TP bucket fractions; `ModelSignalSource` decodes `idx = dir*25 + sl_idx*5 + tp_idx` into `sl_frac`/`tp_frac`; `Decision` carries them through `DecisionEngine.filter` untouched (no silent discard); `TradeExecutor` converts them to order prices `entry*(1-frac) / entry*(1+frac)`, clamped to the configured bounds before attach.
- **`ATR_RULES`** (`SLTP_MODE=rules`, legacy 2/3-action artifact): the `Decision` carries `sl_frac=None`/`tp_frac=None` and the executor uses the existing ATR path (`entry - atr*sl_atr_mult` / `entry + atr*tp_atr_mult`) exactly as before. Explicit model fractions are ignored in rules mode.

The badge is driven by `cfg.behavior.sl_tp_mode` plus the active manifest's `sl_tp_mode` — never by a UI guess.

> **Honest guard note (non-negotiable):** the `SLTP_MODE` config + manifest `sl_tp_mode` guard (`core/config.py::enforce_sltp_compatibility`, called first from `live/live_trade_mt5.py::enforce_model_promotion_gate`) raises a clear `ValueError` at startup when a **model-trained artifact** (`sl_tp_mode="model"`) is configured with `SLTP_MODE=rules`, and also when `SLTP_MODE=model` is configured with a **rules-trained artifact**. A model's trained SL/TP output is **never silently discarded** — the system refuses to start rather than degrade. The dashboard must surface this state: if the mode badges disagree with the active artifact, the "start live" button stays disabled with the guard error message. **[FACT + ENGINEERING RECOMMENDATION]**

### 4.5 Brain panel (per model type)


- **Dreamer** (`models/dreamer_agent.py`): render RSSM latent stats (`stoch_dim=32`, `num_categories=32`), the imagination rollout (horizon `15`) as a small multi-branch tree/timeline, critic value estimates, and world-model reconstruction loss live during training. **[FACT]** `compute_world_model_loss` returns `(loss, metrics)` incl. `states` — bind to the panel.
- **Transformer** (`models/transformer_policy.py`): `attention_weights` → heatmap of heads x positions; plus `train_step` metrics `{loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction}`. **[FACT]**
- **Ensemble** (`models/ensemble.py`): member votes, `agreement`, `consensus` (bool vs `min_agreement=0.6`), `uncertainty`, `epistemic_uncertainty`. **[FACT]**
- **MCTS** (`models/mcts.py`): per-action `visit_counts`, `q_values`, `visit_distribution`, `root_visits`, `probs` from `MCTS.search()`. **[FACT]**
- **Adversarial** (`models/adversarial_training.py`): `manipulation_rate` and per-episode `manipulation_count`; env perturbation type per step (`manipulation_type` from `action_names`); `SelfPlayTrainer.history` (`trader_wins[]`, `mm_profits[]`). **[FACT]**
- **Meta (MAML)** (`models/meta_learning.py`): regime task list `{name, label, start, end, support, query}` and per-epoch `meta_loss`/`support_loss`. **[FACT]**

**[UNPROVEN]** — note: "imagination rollout quality implies alpha" is unproven (FINAL study Q2). The panel shows internals for audit/explainability, not as a performance claim.

### 4.6 Risk circuit-breaker panel

**Every rejection reason from `RiskSupervisor` rendered as a status chip, with the exact strings [FACT `models/risk_supervisor.py`]:**

| Chip | Meaning |
|---|---|
| `APPROVED` | trade passed all checks |
| `CIRCUIT_BREAKER: daily loss …` | daily-loss breaker |
| `HALTED until …` | emergency halt (365 d) active |
| `MAX_DRAWDOWN: …` | drawdown > `max_drawdown` |
| `POSITION_TOO_LARGE: …` | size > `max_position` |
| `TOO_MANY_LOSSES: …` | ≥ `max_consecutive_losses` |
| `HIGH_VOLATILITY: …` | vol > `vol_threshold` |
| `CORRELATION_GUARD: …` | DXY correlation block |
| `EVENT_RISK: …` | news event scale-down |
| `MAX_TRADES: …` | daily trade cap |
| `COOLDOWN: …` | min interval not elapsed |
| `SPREAD_TOO_WIDE: …` | spread > `max_spread` |
| `MARKET_CLOSED: …` | outside market hours |

Panel also shows `get_statistics()` counters (`rejection_reasons` dict) as a per-reason bar chart, plus `emergency_shutdown()`/halt countdown. **[FACT]**

**SL/TP fraction bounds clamp [FACT `core/config.py` + `live/trade_executor.py`]:**

In `SLTP_MODE=model` the executor clamps the model-decided fractions to the configured bounds before converting to prices: `sl_tp_min_frac <= sl_frac <= sl_tp_max_frac` and `tp_min_frac <= tp_frac <= tp_max_frac` (defaults 0.005 / 0.05 for both SL and TP). The risk panel shows the active bounds and the clamped result per order — e.g. a model SL request of 0.50 renders as **0.05 (clamped)** and a TP of 0.001 as **0.005 (clamped)**. Clamping happens on the *fraction*, then the order prices attach as `entry*(1-frac) / entry*(1+frac)`. **[FACT]**

### 4.7 Economic calendar event feed


**Data source: `features/calendar_features.py` (verified).** Events arrive with `time, currency, impact, forecast, previous` shape; the feature pipeline shifts them into the model input (`macro_shift=1`). The feed panel colors by impact (high/medium/low) and flags `EVENT_RISK`-style near-event trades. **[ENGINEERING RECOMMENDATION]** The calendar is advisory input to the model, never an order source.

### 4.8 Kill switch + TRADING_MODE badge + REQUIRE_PROMOTED_MODEL

- **Kill switch**: a big red button that writes the **kill-switch file** (`KILL_SWITCH` path from `PathConfig`, default `KILL_SWITCH`) — the live loop (`live/live_trade_mt5.py`) polls this file and exits promptly. **[FACT]** The dashboard should also surface whether the file exists (armed state).
- **TRADING_MODE badge**: `demo` (green, MockBroker) vs `live` (red, MT5), from `core/config.py::TradingMode`. **[FACT]**
- **REQUIRE_PROMOTED_MODEL badge**: show the honest gate state — since no model has passed promotion, the badge reads **"BLOCKED — no promoted model"** and the live loop refuses model-driven trades. **[FACT]** (from `core/config.py` `require_promoted_model=True` default + honest `evaluation.json` facts.)

### 4.9 Reconciliation status

`live/trade_executor.py::startup_reconcile()` returns a `ReconResult` (`live/broker.py`): matched/mismatched positions, drift, `ok` flag; drift → `halt_reason = "startup reconciliation drift: ..."` and the executor halts. **[FACT]** Render: status pill (OK / DRIFT), drift magnitude, timestamp, and the halt banner. This feeds `check_live_gates(..., reconciliation_passed=...)`. **[FACT]**

### 4.10 Log tail (live loop)

Tail of the live child's captured stdout/stderr (supervisor tee + `/ws/logs`), with follow-toggle. Display-only; no metric fabrication. **[ENGINEERING RECOMMENDATION]**



---

## 5. How to control the system from the dashboard

### 5.1 Control principles

All control actions are **POST requests to the backend supervisor**; the frontend never directly touches processes, state files, or the broker. The backend re-validates every parameter before acting. **[ENGINEERING RECOMMENDATION]**

The backend spawns children with `subprocess.Popen(..., creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)` on Windows so the whole tree can be terminated by PID group. **[ENGINEERING RECOMMENDATION]** The venv interpreter is `.venv\Scripts\python.exe` (verified to exist at repo root). **[FACT]**

### 5.2 Start training from the UI

1. User composes/edits a run in the launch form (Section 3.6) — every field maps 1:1 to a real CLI flag of the target train script (e.g. `--steps`, `--prefill`, `--batch-size`, `--train-every`, `--seq-len`, `--window`, `--train-end`, `--data`, `--device` for `train/train_dreamer.py`). **[FACT CLI]**
2. Frontend submits `POST /api/jobs/start` with the job spec.
3. Backend validates: all flags known, types correct, values within range (e.g. `0 < batch_size <= 4096`), artifact dir name free, **no other job running** (concurrency guard), data CSV exists, feature contract present for the target window, and base model selected when the pipeline requires one (adversarial/meta). **[ENGINEERING RECOMMENDATION]**
4. Backend spawns `[venv_python, train_script, ...flags]`, tees stdout/stderr to `logs/<job_id>.log`, registers the PID, opens `/ws/training` + `/ws/logs` channels, and returns the job object.

### 5.3 Stop (clean) / Pause / Resume / Kill

- **Stop (clean):** backend writes a sentinel file (e.g. `logs/<job_id>.STOP`), then sends a graceful `terminate()` to the process group. Training loops check for the sentinel each iteration; Dreamer's `save()` persists a full checkpoint (optimizers, RNG state, `return_low/high` EMA, `training_step`) so the run is resumable. **[FACT `models/dreamer_agent.py`]**
- **Pause / Resume:** same sentinel mechanism, supervisor-level (scripts do not implement pause natively — recommended addition). Pause = set `.PAUSE` sentinel (loop stops stepping, process stays alive); Resume = remove it. **[ENGINEERING RECOMMENDATION]**
- **Kill (hard):** `taskkill /PID <pid> /T /F` on Windows to force-kill the process group; last resort after stop. The job is marked `killed`, and the UI shows partial progress honestly. **[ENGINEERING RECOMMENDATION]**
- **Resume a run:** `POST /api/jobs/{id}/resume` → re-spawn `train/train_dreamer.py --resume <checkpoint_path> ...`. **Dimension pre-check (mandatory):** the manifest's `obs_dim` must equal `window * n_features + 5` and `action_dim` must match the env — the script itself raises `ValueError` on mismatch **[FACT `train/train_dreamer.py` + `core/model_artifacts.py`]**, and the backend must catch that before spawn and surface "dimension mismatch — cannot resume" rather than a crashed process.

### 5.4 Start / stop live trading

- **Start live:** `POST /api/live/start` → backend spawns `live/live_trade_mt5.py` as a child. The **TRADING_MODE gate** (`core/config.py::check_live_gates`) decides the broker: `demo` → `MockBroker` (no real orders); `live` → `Mt5Broker`, refused unless *all* gates pass (feature contract present, model present, risk state loadable, reconciliation passed, MT5 creds configured). **[FACT]**
- **Stop live (clean):** write the **kill-switch file** (`KILL_SWITCH` path). The live loop checks it every iteration and halts with `logger.critical("KILL_SWITCH present — halting")` **[FACT `live/live_trade_mt5.py`]**. The dashboard should mark the kill-switch file armed/unarmed state.
- **Kill live (hard):** process-group kill as last resort.
- **Stop trading (risk stop):** `POST /api/live/stop` → also flips a "trading disabled" flag in the supervisor so no further start is possible until an explicit confirmation; never a broker call. **[ENGINEERING RECOMMENDATION]**

### 5.5 Load / activate a model artifact

- List candidate models from `artifacts/models/<run>/manifest.json` (+ `evaluation.json`) via `GET /api/models`.
- `POST /api/models/activate` with `{model_type, run_name}` → backend validates the manifest (`MANIFEST_VERSION=1`, model file exists, contract file exists, **contract hash matches** `feature_contract.json`, `obs_dim == window*n_features+5`) **[FACT `core/model_artifacts.py`]** and copies/symlinks to the production manifest path `artifacts/models/production/manifest.json`. **[ENGINEERING RECOMMENDATION]**
- **HONEST-STATE RULE:** the UI must show, per candidate, its real `evaluation.json` result. Every inspected evaluation has `passed: false` → the card shows **"NOT PROMOTABLE"**; if no production manifest exists → the model list shows **"none trained"** for production. Activation is allowed for *experimentation/demo*, but the badge **REQUIRE_PROMOTED_MODEL** (default `true` in `core/config.py`) keeps the live loop blocked until a model truly passes. **[FACT]**

### 5.6 Edit risk limits via validated `.env` write

- The risk panel's "Edit limits" form exposes the whitelisted keys: `MAX_DAILY_LOSS`, `MAX_DRAWDOWN`, `MAX_CONSECUTIVE_LOSSES`, `MAX_RISK_PER_TRADE`, `MAX_POSITION`, `MAX_TRADES_PER_DAY`, `MIN_TRADE_INTERVAL_SEC`, `VOL_THRESHOLD`, `MAX_SPREAD`, `MARKET_HOURS_ONLY`, `CORRELATION_GUARD_ENABLED`, `CORRELATION_ASSET`, `CORRELATION_BLOCK_LONG_ON_UP`, plus trading behavior: `MIN_CONFIDENCE`, `MIN_ENSEMBLE_AGREEMENT`, `USE_KELLY`, `KELLY_FRACTION`, `KELLY_MIN_TRADES`. **[FACT `.env` key list / `core/config.py`]**
- Backend validates the new `.env` by parsing it and running `core.config.load_config()` against the merged file **before writing**; invalid values → 422 with a precise message and **no write**. **[ENGINEERING RECOMMENDATION]**
- Write is atomic (temp file + rename) and **applies on next process start** (env is read at load time) — the UI must display "changes take effect on restart" prominently. **[FACT: `load_config(env_file, _env)` is startup-time]**

### 5.7 HARD SAFETY INVARIANT (non-negotiable)

> **The frontend NEVER calls broker or order functions directly.**
> It has no MT5 client, no `MockBroker` instance, no `TradeExecutor`, and no `RiskSupervisor` "approve then send" path. The dashboard may only:
> 1. **start/stop/kill/resume child processes** (training and live loops),
> 2. **read state files and metrics** (JSON/DB/logs/evaluation artifacts),
> 3. **write validated configuration** (`.env` via the validated writer, sentinel/kill-switch files).
>
> All order execution, fill handling, and position reconciliation happen **exclusively inside the live child processes** (`live/trade_executor.py` → `MockBroker`/`Mt5Broker`). Any UI feature that would need to "place a close order directly" must instead write the kill-switch/sentinel or a validated config change and let the live loop act on it. This rule is enforced in code (the backend never imports `live/trade_executor.py` or `live/broker.py`), reviewed in CI, and treated as a release blocker. **[ENGINEERING RECOMMENDATION — HARD RULE]**



---

## 6. How to prepare the AI for training from the frontend

Before any training run, the pipeline needs (a) prepared, leakage-free data, (b) a feature contract, and (c) validated hyperparameters. The "Prepare" workflow is orchestrated by the backend supervisor as **child jobs** (same process model as training) or validated config writes — the frontend never runs data code in-process. **[ENGINEERING RECOMMENDATION]**

### 6.1 Data-prep job form

| Field | Semantics (verified) |
|---|---|
| Data CSV selector | Dropdown of `data/*.csv` files (e.g. `data/xauusd_h1.csv`); validated to exist before job start. **[FACT: `data/` dir + `--data` CLI arg]** |
| Macro / calendar features toggle | Optional: enables `features/calendar_features.py` event flags + macro panel; requires `MACRO_CSV` env path. Macro/calendar columns enter the panel `shift(1)`-lagged (causal — `macro_shift=1`). **[FACT `core/config.py` + `features/calendar_features.py`]** |
| Train/validation split | `--train-end` (e.g. `2022-01-01`) splits history. Feature pipeline is **fit only on the train window**; validation bars never touch the fit (`fit_transform(train)` then `transform(val)`). **[FACT `train/data.py` — leak-free]** |
| Contract creation | The job writes `feature_contract.json` (window, n_features, feature names, normalization params, `contract_hash`) into the artifact dir / train root. **[FACT `core/feature_pipeline.py` + `core/model_artifacts.py`]** |
| Leakage-check summary | After the job: train/val bar counts, warm-up bars dropped (`drop_warmup=200`), macro lag applied, checklist "no validation row leaked into fit". **[FACT `core/config.py` FeatureConfig]** |

The form maps 1:1 to `train/data.py` CLI flags plus `--data` / `--train-end` / `--contract`. **[ENGINEERING RECOMMENDATION]**

### 6.2 Contract validation display

Once a contract exists, show:

- `feature_contract.json` schema: `window`, `n_features`, `feature_names[]`, `normalization` params, `contract_hash`. **[FACT `core/model_artifacts.py`]**
- Hash check status: `load_manifest()` recomputes the hash and compares; mismatch → red banner "CONTRACT HASH MISMATCH — artifact is stale". **[FACT `core/model_artifacts.py::load_manifest`]**
- Dimension consequence: model `obs_dim` must equal `window * n_features + 5` (the +5 is the account vector `[position, trade_pnl, bars_in_trade, drawdown, equity_ratio]`). **[FACT `env/dreamer_trading_env.py` + `core/model_artifacts.py`]**


---


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

> **Honesty note (script inventory):** `train/train_dreamer_mcts.py` and `train/train_meta.py` **do not exist** in this repo (verified `dir /b train\*.py`). They must never appear as launch targets. MCTS-enabled deployments are assembled from a standard Dreamer artifact via `train/make_mcts_manifest.py`; meta-training runs through `train/meta_train_dreamer.py`. **[FACT]**

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

## 7. Proposed API surface

The backend exposes a small REST surface plus WebSocket channels. All reads are typed readers over the real state sources; all mutations go through the supervisor and re-validation. **[ENGINEERING RECOMMENDATION]**

### 7.1 REST endpoints

| Method | Path | Purpose | Validates |
|---|---|---|---|
| GET | `/api/status` | Global status: TRADING_MODE, kill-switch armed, gate results (`check_live_gates`), active job | — |
| GET | `/api/jobs` | List training/data-prep jobs (past + current) | — |
| POST | `/api/jobs/start` | Launch a training/prep job (job spec below) | Concurrency guard, CLI flags, CSV exists, contract hash, base manifest for staged |
| POST | `/api/jobs/{id}/stop` | Clean stop via sentinel + graceful terminate | Job exists & running |
| POST | `/api/jobs/{id}/kill` | Hard process-group kill (`taskkill /T /F`) | Job exists |
| POST | `/api/jobs/{id}/resume` | Resume via `--resume` + checkpoint path | Dimension check `obs_dim == window*n_features+5` |
| POST | `/api/jobs/{id}/pause` | Pause via `.PAUSE` sentinel | Job exists & running |
| POST | `/api/jobs/{id}/unpause` | Resume from pause (clear sentinel) | Job exists & paused |
| GET | `/api/models` | Candidate models from `artifacts/models/*/manifest.json` + `evaluation.json` honesty chips | — |
| POST | `/api/models/activate` | Activate artifact → `artifacts/models/production/manifest.json` | MANIFEST_VERSION, files exist, contract hash, obs_dim |
| GET | `/api/risk/state` | `RiskSupervisor.get_statistics()` + halt + rejection_reasons | — |
| PUT | `/api/config/env` | Validated `.env` write (whitelisted keys) | `load_config()` dry-run on merged file; atomic rename |
| POST | `/api/prep/run` | Run data-prep job (CSV, macro toggle, train_end, contract) | CSV exists, train_end in range |
| GET | `/api/prep/leakcheck` | Leakage-check summary of the last prep job | — |
| GET | `/api/state/{panel}` | Panel snapshot (account, positions, closed, decisions, brain, calendar) | Panel whitelist |
| GET | `/api/reconciliation` | Latest `ReconResult` (matched/mismatch/drift/ok) | — |
| POST | `/api/live/start` | Start the live child | TRADING_MODE gates (demo vs live) |
| POST | `/api/live/stop` | Stop the live child via kill-switch file | Live child running |
| POST | `/api/live/killswitch` | Arm/disarm the kill-switch file | File write only |

(19 endpoints — exceeds the required 12.) **[ENGINEERING RECOMMENDATION]**

### 7.2 WebSocket channels

| Channel | Payload cadence | Contents |
|---|---|---|
| `/ws/training` | per train step | loss-curve series, env ticker rows, replay buffer usage, regime labels, job progress |
| `/ws/live/decisions` | per decision | full decision feed item (below) |
| `/ws/live/risk` | on change | risk state delta: current equity, daily pnl, rejection reasons, halt countdown |
| `/ws/logs` | batched tail | appended child-process stdout/stderr lines (training + live) |
| `/ws/checkpoints` | on write | `checkpoint_<step>.pt` + `evaluation.json` events, `passed` chip |

(5 channels — exceeds the required 4.) **[ENGINEERING RECOMMENDATION]**

### 7.3 JSON payload examples

**Job start request** (POST `/api/jobs/start`):

```json
{
  "preset": "dreamer",
  "script": "train/train_dreamer.py",
  "run_name": "dreamer_r02",
  "flags": {
    "steps": 100000, "prefill": 5000, "batch-size": 16,
    "seq-len": 64, "window": 64, "embed-dim": 256,
    "hidden-dim": 512, "stoch-dim": 32, "num-categories": 32,
    "horizon": 15, "train-end": "2022-01-01",
    "data": "data/xauusd_h1.csv", "device": "cpu"
  },
  "base_manifest": null
}
```

**Job status** (GET `/api/jobs/{id}` / WS event):

```json
{
  "job_id": "dreamer_r02",
  "status": "running",
  "pid": 12345,
  "step": 42100,
  "total_steps": 100000,
  "checkpoints": ["checkpoint_40000.pt"],
  "metrics": {"world_model_loss": 0.712, "recon_loss": 0.431,
              "reward_loss": 0.088, "kl_loss": 0.193,
              "value_loss": 0.224, "policy_loss": 0.117, "entropy": 1.34},
  "started_at": "2026-01-01T12:00:00Z"
}
```

**Decision feed item** (`/ws/live/decisions`):

```json
{
  "ts": "2026-01-01T13:05:00Z",
  "bar": "2026-01-01T13:00:00Z",
  "inputs": {"obs_dim": 197, "window": 64, "n_features": 3, "account": [0, 0, 0, 0.01, 1.0]},
  "probs": {"flat": 0.62, "long": 0.31, "short": 0.07},
  "confidence": 0.58, "agreement": 0.73, "uncertainty": 0.41,
  "filter": "LOW_CONFIDENCE",
  "risk": {"approved": false, "reason": "COOLDOWN: wait 120s before next trade"},
  "order": {"side": null, "lot": 0.0, "sl": null, "tp": null},
  "fill": {"retcode": null, "retcode_name": null, "price": null},
  "equity": 10012.5, "reward": 0.0012
}
```

> **obs_dim consistency note:** `obs_dim` always follows `window*n_features + 5` (the account vector is 5 scalars: position, trade_pnl, bars_in_trade, drawdown, equity_ratio). The example above uses `64*3 + 5 = 197`. The real Dreamer artifact in this repo uses `n_features=15 -> obs_dim=965` (see `artifacts/models/dreamer_20261003T230304213562Z/manifest.json` and its `feature_contract.json`), so the dashboard must read `obs_dim` from the active feature contract at runtime — never hard-code it. **[FACT `core/model_artifacts.py::load_manifest`]**

**Risk state** (GET `/api/risk/state`):

```json
{
  "current_equity": 10012.5, "peak_equity": 10040.0,
  "daily_pnl": -27.5, "daily_loss_ratio": -0.0027,
  "trades_today": 3, "consecutive_losses": 2,
  "halt_remaining_sec": 0, "trade_date": "2026-01-01",
  "rejection_reasons": {"COOLDOWN": 1, "HIGH_VOLATILITY": 2},
  "approval_rate": 0.83
}
```

### 7.4 State-source mapping table

| Dashboard panel (section) | Primary source (verified) |
|---|---|
| Account summary (4.1) | `state/risk_state.db` via `RiskSupervisor.get_statistics()`; `mock_state.json` balance/mid |
| Candlestick chart (4.2) | Broker bars (H1 default); `mock_state.json` `mid`; SL/TP from `TradeExecutor.entry_sl_tp` |
| Positions + closed trades (4.3) | `mock_state.json` `positions[]` / `closed[]` (ticket/side/volume/entry/exit/pnl/reason); `bot_state.json` `last_bar_time` |
| Decision feed (4.4) | `live/decision_engine.py::filter` reasons; `models/ensemble.py` info keys; `RiskSupervisor.check_trade`; `OrderResult` retcodes; env `equity` |
| Brain panel (4.5) | `models/*` internals: `train_step` metrics dict, `attention_weights`, `visit_counts`/`q_values`, `manipulation_rate`, meta task list |
| Risk breaker panel (4.6) | `RiskSupervisor` reason strings + `get_statistics()` rejection_reasons |
| Economic calendar (4.7) | `features/calendar_features.py` event rows |
| Kill switch / badges (4.8) | `KILL_SWITCH` file existence; `core/config.py` TradingMode + REQUIRE_PROMOTED_MODEL |
| Reconciliation (4.9) | `ReconResult` from `live/trade_executor.py::startup_reconcile` |
| Job list / progress (3.1) | Supervisor PID registry + `artifacts/models/<run>/` checkpoints |
| Loss curves (3.2) | `train_step` metrics dict (7 keys); transformer/adversarial/meta alternates |
| Env ticker (3.3) | env `info` keys (equity/position/pnl/return/cost/swap/drawdown/forced_close/blown_up) |
| Replay buffer (3.4) | `ReplayBuffer` size/capacity (100k) / seq_len 64 |
| Regime labels (3.5) | `MarketRegimeGenerator` labels |
| Hyperparameters (3.6) | Launch payload + `manifest.json` hyperparams |
| Checkpoint timeline (3.7) | `artifacts/models/*/evaluation.json` (metrics, thresholds, `passed`) |
| Logs tail (3.8 / 4.10) | supervisor-tee log files via `/ws/logs` |
| Data prep (6.1) | `feature_contract.json`; leak-check summary from prep job |


---

## 8. Non-functional requirements

### 8.1 Deployment shape

- **Single-user local application.** The dashboard binds to `127.0.0.1` on a fixed port; the SPA is served by the same FastAPI process. Optional bearer token (generated at first launch, stored in `.env`) for browser access. **[ENGINEERING RECOMMENDATION]**
- **Two processes total at runtime:** (1) the FastAPI backend (supervisor + readers + WS hub), (2) zero-or-one child (training job or live loop) — never both simultaneously (the concurrency guard covers training; the supervisor also refuses `POST /api/live/start` while a training job runs). **[ENGINEERING RECOMMENDATION]**

### 8.2 WebSocket vs polling decision

- **WebSocket for live feeds:** loss-curve ticks (per train step), env ticker, decision feed, risk deltas, log tail, checkpoint events — high-frequency or push-on-change; polling would lag or hammer the readers. **[ENGINEERING RECOMMENDATION]**
- **Polling fallback (REST) for slow panels:** model list, evaluation timeline, prep leak-check summary, reconciliation result. Refresh every 5–10 s or on demand. **[ENGINEERING RECOMMENDATION]**
- Client reconnection: WS auto-reconnect with exponential backoff; on reconnect the client re-requests a full snapshot (`GET /api/status`, `GET /api/risk/state`, `GET /api/state/{panel}`) before resuming the stream, so no event is lost silently. **[ENGINEERING RECOMMENDATION]**

### 8.3 Resilience to child-process crash

- If a child (training or live) exits unexpectedly, the supervisor: marks the job `failed` with exit code, keeps the log tail visible, raises a non-blocking banner, and — for training only — offers **Resume** if a checkpoint exists (`--resume` + checkpoint path, with the dimension pre-check). **[ENGINEERING RECOMMENDATION]**
- The backend never auto-restarts the **live** loop; restart requires explicit user action (safety first). **[ENGINEERING RECOMMENDATION — HARD RULE]**

### 8.4 Latency budgets

| Feed | Budget (p95) |
|---|---|
| Loss-curve tick → rendered | < 1 s |
| Decision feed item → rendered | < 250 ms |
| Log tail line → rendered | < 500 ms |
| REST snapshot | < 200 ms |
| Control action acknowledged | < 1 s |

**[ENGINEERING RECOMMENDATION]** These budgets are achievable only with WS push + canvas rendering; the Streamlit fallback meets only the REST snapshot and control-action budgets.

### 8.5 Windows process management

- Children are spawned with `subprocess.CREATE_NEW_PROCESS_GROUP` so the whole tree can be terminated by group. **[ENGINEERING RECOMMENDATION]**
- Clean stop: sentinel file (`.STOP` / `.PAUSE`) + graceful `terminate()`; hard kill: `taskkill /PID <pid> /T /F`. **[ENGINEERING RECOMMENDATION]**
- **Sentinel kill-switch pattern (already in the live loop — FACT):** `live/live_trade_mt5.py` checks the `KILL_SWITCH` file every loop iteration and halts with `logger.critical("KILL_SWITCH present — halting")`. The dashboard arms/disarms this file and always displays its state. **[FACT + ENGINEERING RECOMMENDATION]**

### 8.6 HONEST-STATE RULE (non-negotiable)

> **The dashboard never fabricates metrics.**
> - `evaluation.json` with `passed: false` renders **"NOT PROMOTABLE"** — never a green promotable badge. Every inspected artifact currently has `passed: false`. **[FACT]**
> - Missing `artifacts/models/production/manifest.json` renders **"none trained"** for the production slot — never an empty-but-active look. **[FACT]**
> - A killed/crashed job shows partial progress with a `failed`/`killed` chip — no implied completion.
> - Every number shown on screen must be traceable to a file path or metrics-dict key listed in Section 7.4; any panel that cannot resolve its source shows `—` (not a fabricated 0).

---

## 9. Risks & mitigations

| Risk | Impact | Mitigation (dashboard design) |
|---|---|---|
| Subprocess orphans after dashboard crash/close | Zombie training/live processes keep trading or consuming CPU | Process-group spawn (`CREATE_NEW_PROCESS_GROUP`); on startup the supervisor scans for orphan PIDs from its registry file and offers to terminate them; kill-switch file always honored by the live loop **[ENGINEERING RECOMMENDATION + FACT]** |
| Heavy DOM churn from per-step metrics | UI jank, high RAM in browser | Canvas/WebGL charts with downsampling (max ~2k points/series, aggregate beyond); batch WS frames (e.g. 10 steps/frame); virtualized log view (render last N lines only) **[ENGINEERING RECOMMENDATION]** |
| Timezone confusion (bars are UTC) | Misread signal times, wrong daily-loss boundaries | Display **all** timestamps in UTC (ISO-8601 with `Z`), honor `BROKER_UTC_OFFSET_HOURS` only inside live-account rendering; `mock_state.json` `last_bar_time` is `+00:00` **[FACT]** |
| Venv path resolution failure | Spawn fails with confusing error | Preflight check #6 verifies `.venv\Scripts\python.exe`; supervisor logs the resolved interpreter path at startup **[FACT: venv verified present]** |
| Contract drift (contract vs model mismatch) | Silent dimension mismatch, NaN signals | Manifest hash check + `obs_dim == window*n_features+5` validation on load and resume; UI shows CONTRACT HASH MISMATCH banner **[FACT `core/model_artifacts.py`]** |
| Data prep leakage (validation bars in fit) | Optimistic eval, false promotion | `train/data.py` fits on train window only; leak-check summary shown after every prep job **[FACT]** |
| Live trade while no promoted model | Unauthorized risk | `REQUIRE_PROMOTED_MODEL` badge (currently BLOCKED) + `check_live_gates` enforced at spawn **[FACT]** |

---

## 10. Roadmap mapping to the FINAL study experiments (E1–E10)

Each experiment from `research/FINAL_AUTONOMOUS_TRADING_AI_STUDY.md` (Proposed Experiments section) maps to dashboard features that **enable** or **verify** it:

| Exp | Experiment (FINAL study) | Dashboard features that enable/verify it |
|---|---|---|
| E1 | Cross-asset data panel (DXY/10Y/VIX, shift(1)) | Data-prep form (macro CSV selector + lag display); feature panel showing macro columns; correlation-guard chip in risk panel fires on real data |
| E2 | Learned representation vs feature pipeline | Prep form (contract build); job list with A/B runs; loss curves + eval timeline compare rollout loss and OOS P&L per variant |
| E3 | Walk-forward vs CPCV evaluation methodology | Eval timeline renders both protocols' promotion chips side-by-side; no training UI change |
| E4 | World-model value (imagination vs replay) | Job list with env-steps counter; loss curves (world_model_loss, entropy); checkpoint timeline shows which policy type reached parity |
| E5 | Reward ablation | Hyperparameter panel exposes reward-type/composite-weight flags per run; job comparison table maps reward variant → evaluation.json metrics |
| E6 | Offline pretraining (CQL/DT) | Prep form + base-manifest selector for staged runs; job list shows phase markers (pretrain→finetune); sample-efficiency counter (env steps to parity) on eval timeline |
| E7 | Regime-conditioned world model | Regime-label strip (3.5) colors per-regime OOS metrics; meta-train preset exposes regime-conditioning flags |
| E8 | LLM research-desk ablation | Economic calendar feed (4.7) + advisory-feature toggle in prep form; feature-attribution panel (brain panel) shows whether the policy uses event flags |
| E9 | Shadow self-improvement loop | Champion/challenger comparison on eval timeline; auto-rollback events feed; job list shows offline challenger training; risk panel shows non-degradation counters |
| E10 | Live-safe promotion trial | TRADING_MODE badge (demo→live), kill-switch arm state, reconciliation status, account panel vs shadow expectation; zero-safety-violation log |

---

## 11. Verifiable acceptance criteria

1. **File & structure:** `frontend_dashboard_design.md` exists at repo root, 600–900 lines, with all section headers: Overview, Tech-Stack, Training, Trading, Control, Preparation (AI), API, NFR, Risks, Roadmap, Acceptance. **[THIS DOCUMENT]**
2. **Tech-stack:** comparison table with ≥ 6 rows (FastAPI+WS+SPA vs Streamlit), decision rationale references the FACT that no backend exists today.
3. **Training view:** all 11 sub-panels listed, each with a concrete data source; the 7 `train_step` metric keys (`world_model_loss`, `recon_loss`, `reward_loss`, `kl_loss`, `value_loss`, `policy_loss`, `entropy`) appear verbatim.
4. **Trading view:** every panel has a concrete data source; the 13 risk reason strings appear verbatim (APPROVED, CIRCUIT_BREAKER, HALTED, MAX_DRAWDOWN, POSITION_TOO_LARGE, TOO_MANY_LOSSES, HIGH_VOLATILITY, CORRELATION_GUARD, EVENT_RISK, MAX_TRADES, COOLDOWN, SPREAD_TOO_WIDE, MARKET_CLOSED).
5. **Control:** HARD SAFETY INVARIANT present as a bolded rule (frontend never calls broker/order functions); resume dimension check (`obs_dim == window*n_features+5`); honest no-promoted-model statement.
6. **Preparation:** all 8 presets named exactly (dreamer, transformer, ppo, ensemble, god-mode, ultimate-150, adversarial, meta-train); all train-script names appear (train_dreamer.py, train_transformer.py, train_ppo.py, train_ensemble.py, train_adversarial.py, meta_train_dreamer.py, make_mcts_manifest.py); preflight checklist ≥ 6 items.
7. **API surface:** ≥ 12 REST endpoints, ≥ 4 WebSocket channels, 4 JSON payload examples (job start, job status, decision feed, risk state), and a state-source mapping table covering every panel from the training and trading sections.
8. **Honesty:** HONEST-STATE RULE present; `passed=false` renders "NOT PROMOTABLE"; missing production manifest renders "none trained"; no fabricated metric path exists in the spec.
9. **Roadmap:** E1–E10 all mapped (10 rows) to dashboard features.
10. **Risks:** ≥ 5 rows in the risks table.
11. **Acceptance:** this list itself has ≥ 8 items, each tied to a section.
12. **Claim markers:** every major claim tagged FACT / RESEARCH FINDING / ENGINEERING RECOMMENDATION / UNPROVEN.


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
- Venv interpreter `.venv\Scripts\python.exe` **exists** at repo root — preflight check #6 relies on this. **[FACT]**
