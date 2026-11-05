# XAUUSD Interlock dashboard

React + TypeScript SPA using Vite and LuxAlgo Vela. The dashboard is a display/control client for the FastAPI supervisor; it is not an order-entry terminal.

## Local development

From the repository root, in separate terminals:

```powershell
.venv\Scripts\python.exe -m uvicorn frontend_backend.backend.main:app --host 127.0.0.1 --port 8000
```

```powershell
Push-Location frontend_backend\frontend
npm install
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` and `/ws` to the local API.

## Production build

```powershell
Push-Location frontend_backend\frontend
npm run build
Pop-Location
.venv\Scripts\python.exe -m uvicorn frontend_backend.backend.main:app --host 127.0.0.1 --port 8000
```

FastAPI serves `frontend/dist` when that directory exists.

## Safety and source boundaries

- Vela charts use only the repository's XAUUSD CSV files. The provider is historical and has no subscription/live mode.
- Account and positions from `state/mock_state.json` are read only when `TRADING_MODE=demo`; they are hidden in live mode because no broker-backed reader exists.
- Evaluations come from `artifacts/models/*/evaluation.json`; failed artifacts are `NOT PROMOTABLE`.
- Control endpoints only launch/terminate validated child processes, write a validated `.env` diff, or touch the configured kill-switch sentinel. The frontend and backend do not call broker/order methods.
- The current backend implements Dreamer launch parameters and data-contract preparation. Other presets are visible but their launch schemas are not enabled. Live child-process launch, live decision/reconciliation readers and clean cooperative stopping are intentionally not implemented.

See [`../docs/DASHBOARD.md`](../docs/DASHBOARD.md) for endpoint/source details.
