

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
