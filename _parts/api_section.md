
---

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
  "inputs": {"obs_dim": 193, "window": 64, "n_features": 3, "account": [0, 0, 0, 0.01, 1.0]},
  "probs": {"flat": 0.62, "long": 0.31, "short": 0.07},
  "confidence": 0.58, "agreement": 0.73, "uncertainty": 0.41,
  "filter": "LOW_CONFIDENCE",
  "risk": {"approved": false, "reason": "COOLDOWN: wait 120s before next trade"},
  "order": {"side": null, "lot": 0.0, "sl": null, "tp": null},
  "fill": {"retcode": null, "retcode_name": null, "price": null},
  "equity": 10012.5, "reward": 0.0012
}
```

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
