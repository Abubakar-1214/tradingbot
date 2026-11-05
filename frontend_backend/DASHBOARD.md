# XAUUSD Interlock dashboard

## Stack

- `frontend/`: React 19, TypeScript, Vite 8, React Router, Ant Design Icons.
- Charting: `@luxalgo/vela` 0.8.1 (`@luxalgo/vela/workspace`, `/ui`, `/plugin`). Charts use Vela's native canvas/WebGL renderer and locally registered historical CSV provider.
- `backend/`: FastAPI, served by Uvicorn from the repository virtual environment.

## Start and build

Development uses two PowerShell terminals (see [`frontend/README.md`](../frontend/README.md)). The frontend API/WebSocket requests are proxied to `127.0.0.1:8000`. Production `frontend/dist` is served by FastAPI.

## Screens

| Route | Purpose |
|---|---|
| `/` | Operations cockpit, model permission gate, historical Vela chart, source-backed demo/risk and empty runtime feeds |
| `/training` | Child-job supervision, raw Dreamer train-step losses, child log tail and evaluation history |
| `/prepare` | Data/contract preparation, Dreamer CLI fields, preflight and guarded child launch |
| `/models` | Evaluation thresholds, actual artifact rows, per-row gate failures and production slot |
| `/risk` | Resolved configuration, draft diff/review, validated `.env` writer, risk state and kill-switch file action |

## Data and safety

- `backend/readers.py` reads repository artifacts/state and uses a read-only SQLite connection for `state/risk_state.db`.
- Current configured `TRADING_MODE` is discovered through `core.config.load_config()`. The default in code is `demo`; this workspace's `.env` currently selects `live`.
- In `live` mode, `state/mock_state.json` is never presented as the account/positions feed. No MT5-backed live account reader or live decision/reconciliation snapshot currently exists.
- The production model manifest is absent and the three evaluated artifacts do not pass the promotion gate. Model-driven trading remains blocked.
- `/api/jobs/start` supports the Dreamer launch form only and enforces a one-child lock, known CLI flags, data path containment, parameter bounds, unique run name, disk-space check, and existing feature contract. Job snapshots are atomically journaled at `state/supervisor_jobs.json`, guarded by an OS-level single-writer lock. A replacement API process validates PID, process creation time, command line and a per-job inherited token before adopting a child; ambiguous identities remain `unverified` and block another launch. Jobs that no longer exist are retained as `interrupted`. API shutdown detaches surviving children so they can be adopted on restart. The current Dreamer script writes only metrics returned by `train_step()` to a JSONL file under `logs/` when launched by the dashboard.
- `/api/prep/run` launches `backend/prepare_contract.py` as a child, which calls the real `train.data.prepare_data()` pipeline and records its actual contract/report.
- Clean cooperative termination is not available because current training scripts do not poll STOP/PAUSE sentinels. The UI labels this accurately; hard kill remains a separate action.
- Kill-switch endpoint only creates/removes the path configured in `core.config.PathConfig`; it makes no broker/order call.
- Risk changes accept only the documented whitelist, validate the merged environment with `core.config.load_config()`, and replace `.env` atomically. Changes apply next process start.
- Risk statistics and state snapshots may be old; records without source timestamps do not get a fabricated “updated just now” label.

## Endpoint surface implemented

REST: `/api/status`, `/api/models`, `/api/risk/state`, `/api/state/{panel}`, `/api/market/bars`, `/api/reconciliation`, `/api/jobs`, `/api/jobs/start`, `/api/jobs/{id}/stop`, `/api/jobs/{id}/kill`, `/api/jobs/{id}/logs`, `/api/jobs/{id}/metrics`, `/api/prep/run`, `/api/prep/preflight`, `/api/prep/leakcheck`, `/api/config/env`, `/api/live/killswitch`.

WebSockets: `/ws/training`, `/ws/live/decisions`, `/ws/live/risk`, `/ws/logs`, `/ws/checkpoints`. Unimplemented runtime streams return unavailable envelopes rather than fabricated values.

## Vela attribution

Vela's built-in chart attribution remains enabled and a visible `Vela™` link remains in the chart footer. The app also mounts a Vela-hosted source/safety dialog using the UI kit (`Dialog`, `ensureUIHost`) and a Vela topbar action (`registerWidgetAction`) via the plugin SDK. Keep [`THIRD-PARTY-NOTICES.md`](../frontend/THIRD-PARTY-NOTICES.md) with redistributed builds.
