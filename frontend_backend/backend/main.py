from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core.config import REPO_ROOT, load_config
from frontend_backend.backend.readers import (ROOT, dashboard_status, decisions, envelope, evaluations,
                             jobs_snapshot, market_bars, reconciliation, resumable_models, risk_state, state_file)
from frontend_backend.backend.schemas import JobStartRequest, KillSwitchRequest, PrepRunRequest, RiskConfigRequest
from frontend_backend.backend.preflight import csv_time_bounds
from frontend_backend.backend.supervisor import PRESET_SCRIPTS, ProcessSupervisor, SupervisorError

supervisor = ProcessSupervisor()

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        supervisor.shutdown()

app = FastAPI(title="XAUUSD Interlock Control Plane", version="0.1.0", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
                    allow_methods=["GET", "POST", "PUT"], allow_headers=["Content-Type", "Authorization"])
RISK_KEYS = {"MAX_DAILY_LOSS", "MAX_DRAWDOWN", "MAX_CONSECUTIVE_LOSSES", "MAX_RISK_PER_TRADE",
             "MAX_POSITION", "MAX_TRADES_PER_DAY", "MIN_TRADE_INTERVAL_SEC", "VOL_THRESHOLD",
             "MAX_SPREAD", "MARKET_HOURS_ONLY", "CORRELATION_GUARD_ENABLED", "CORRELATION_ASSET",
             "CORRELATION_BLOCK_LONG_ON_UP", "MIN_CONFIDENCE", "MIN_ENSEMBLE_AGREEMENT", "USE_KELLY",
             "KELLY_FRACTION", "KELLY_MIN_TRADES"}

@app.get("/api/status")
def status() -> dict[str, Any]:
    result = dashboard_status()
    if result.get("availability") == "available" and result.get("data"):
        result["data"]["supervisor"] = "available"
    return result

@app.get("/api/models")
def models() -> dict[str, Any]:
    return envelope(evaluations(), source="artifacts/models/*/evaluation.json")

@app.get("/api/models/resumable")
def models_resumable() -> dict[str, Any]:
    return envelope(resumable_models(), source="artifacts/models/*")

@app.get("/api/risk/state")
def risk() -> dict[str, Any]:
    return risk_state()

@app.get("/api/state/{panel}")
def read_state(panel: str) -> dict[str, Any]:
    if panel not in {"positions", "closed", "decisions", "calendar", "brain", "account"}:
        raise HTTPException(404, "Unknown state panel.")
    if panel in {"positions", "closed"}:
        return state_file(panel)
    if panel == "decisions": return decisions()
    if panel == "account":
        try:
            cfg = load_config()
        except Exception as exc:
            return envelope(availability="error", source="core/config.py", reason=str(exc))
        if cfg.trading_mode.value != "demo":
            return envelope(availability="unavailable", source="state/mock_state.json", reason="MockBroker state is hidden while TRADING_MODE is not demo; no broker-backed account reader is connected.")
        path = ROOT / "state" / "mock_state.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return envelope({"balance": data.get("balance"), "mid": data.get("mid")}, source="state/mock_state.json", reason="Persisted demo broker snapshot; not a live quote.")
        except (OSError, ValueError) as exc:
            return envelope(availability="unavailable", source="state/mock_state.json", reason=str(exc))
    if panel == "calendar":
        path = ROOT / "data" / "economic_events_2015_2025.json"
        try:
            events = json.loads(path.read_text(encoding="utf-8"))
            return envelope(events[-100:], source="data/economic_events_2015_2025.json", reason="Historical calendar fixture; not a current economic calendar.")
        except (OSError, ValueError) as exc:
            return envelope(availability="unavailable", source="data/economic_events_2015_2025.json", reason=str(exc))
    return envelope(availability="unavailable", source="model-specific runtime telemetry", reason="No model brain snapshot is available outside an active supervised process.")

@app.get("/api/market/bars")
def bars(symbol: str = "XAUUSD", timeframe: str = "H1", limit: int = Query(3000, ge=1, le=10000)) -> dict[str, Any]:
    return market_bars(symbol, timeframe, limit)

@app.get("/api/reconciliation")
def recon() -> dict[str, Any]:
    return reconciliation()

@app.get("/api/jobs")
def jobs() -> dict[str, Any]:
    return jobs_snapshot(supervisor)

@app.get("/api/jobs/{job_id}/logs")
def logs(job_id: str, limit: int = Query(200, ge=1, le=2000)) -> dict[str, Any]:
    try: return envelope(supervisor.log_tail(job_id, limit), source=f"logs/{job_id}.log")
    except SupervisorError as exc: raise HTTPException(404, str(exc)) from exc

@app.get("/api/jobs/{job_id}/metrics")
def job_metrics(job_id: str, limit: int = Query(600, ge=1, le=2000)) -> dict[str, Any]:
    try:
        result = supervisor.metric_samples(job_id, limit)
        samples = result["samples"]
        return envelope(result, source=result.get("source"),
                        reason=None if samples else "No training metric sample has been written yet.",
                        availability="available" if samples else "unavailable")
    except SupervisorError as exc: raise HTTPException(404, str(exc)) from exc

@app.post("/api/jobs/start")
def start_job(body: JobStartRequest) -> dict[str, Any]:
    if body.preset not in PRESET_SCRIPTS: raise HTTPException(422, "Unknown training preset.")
    try: return envelope(supervisor.start(body.model_dump()), source="backend/supervisor.py")
    except (SupervisorError, ValueError, OSError) as exc: raise HTTPException(409, str(exc)) from exc

@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict[str, Any]:
    try: return envelope(supervisor.terminate(job_id), source="backend/supervisor.py")
    except SupervisorError as exc: raise HTTPException(409, str(exc)) from exc

@app.post("/api/jobs/{job_id}/kill")
def kill_job(job_id: str) -> dict[str, Any]:
    try: return envelope(supervisor.terminate(job_id, hard=True), source="backend/supervisor.py")
    except SupervisorError as exc: raise HTTPException(409, str(exc)) from exc

@app.post("/api/prep/run")
def prep_run(body: PrepRunRequest) -> dict[str, Any]:
    try:
        record = supervisor.start_prep(data=body.data, train_end=body.train_end, window=body.window, macro=body.macro)
        return envelope(record, source="backend/supervisor.py · backend/prepare_contract.py")
    except (SupervisorError, ValueError, OSError) as exc:
        raise HTTPException(409, str(exc)) from exc

@app.get("/api/prep/leakcheck")
def leakcheck() -> dict[str, Any]:
    path = ROOT / "logs" / "prep_latest.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return envelope(result, source="logs/prep_latest.json")
    except (OSError, ValueError) as exc:
        return envelope(availability="unavailable", source="logs/prep_latest.json", reason=str(exc))

@app.get("/api/prep/preflight")
def preflight(preset: str = "dreamer", data: str = "data/xauusd_h1.csv", train_end: str = "2022-01-01", window: int = Query(64, ge=1, le=4096), macro: bool = False) -> dict[str, Any]:
    if preset not in PRESET_SCRIPTS: raise HTTPException(422, "Unknown preset.")
    candidate = (ROOT / data).resolve()
    try: candidate.relative_to((ROOT / "data").resolve())
    except ValueError: raise HTTPException(422, "Data path must be inside data/.")
    exists = candidate.is_file() and candidate.stat().st_size > 0
    try:
        split = datetime.fromisoformat(train_end)
        if split.tzinfo is None: split = split.replace(tzinfo=timezone.utc)
        first_time, last_time = csv_time_bounds(str(candidate), candidate.stat().st_mtime_ns, candidate.stat().st_size) if exists else (None, None)
        split_ok = bool(first_time and last_time and first_time < split < last_time)
    except (OSError, ValueError): split_ok = False
    try: load_config(); config_ok = True
    except Exception: config_ok = False
    contract_path = ROOT / "train" / "feature_contract.json"
    contract_ok = False
    contract_reason = "Feature contract has not been built for this window."
    macro_path = ROOT / "data" / "macro_daily.csv"
    macro_ok = (not macro) or (macro_path.is_file() and macro_path.stat().st_size > 0)
    if contract_path.is_file():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract_window = contract.get("pipeline_params", {}).get("window")
            report_path = ROOT / "logs" / "prep_latest.json"
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
            source_rel = candidate.relative_to(ROOT).as_posix()
            expected_macro = "data/macro_daily.csv" if macro else None
            contract_ok = bool(contract.get("hash")) and contract_window == window and len(contract.get("feature_names", [])) > 0 and report.get("contract_hash") == contract.get("hash") and report.get("window") == window and report.get("source") == source_rel and report.get("train_end") == train_end and report.get("macro_source") == expected_macro and macro_ok
            contract_reason = None if contract_ok else f"A matching prep report for this data/train-end/macro/window is required (requested window {window}, contract window {contract_window})."
        except (OSError, ValueError, TypeError):
            contract_reason = "Feature contract or source report could not be parsed."
    free = shutil.disk_usage(ROOT).free
    python_exists = (ROOT / ".venv" / "Scripts" / "python.exe").is_file()
    current_jobs = supervisor.list_jobs()
    has_running = any(job["status"] in {"running", "starting", "stopping", "unverified"} for job in current_jobs)
    checks = [
        {"id":"data_file", "label":"CSV exists, has OHLC rows, and train split falls inside its range", "availability":"available" if exists and split_ok else "error", "passed":bool(exists and split_ok), "reason":None if exists and split_ok else "Check data path and train-end against the source CSV."},
        {"id":"contract", "label":"Feature contract matches selected data, split, macro input and window", "availability":"available" if contract_ok else "not_checked", "passed":contract_ok, "reason":contract_reason},
        {"id":"macro", "label":"Requested macro CSV exists", "availability":"available" if macro_ok else "error", "passed":macro_ok, "reason":None if macro_ok else "data/macro_daily.csv is missing or empty."},
        {"id":"parameters", "label":"CLI parameters validated by preset schema", "availability":"available" if preset in {"dreamer", "ppo"} else "not_checked", "passed":preset in {"dreamer", "ppo"}, "reason":None if preset in {"dreamer", "ppo"} else "This preset launch schema is not wired yet."},
        {"id":"base_manifest", "label":"Compatible base manifest selected when required", "availability":"not_checked" if preset in {"adversarial", "meta-train"} else "available", "passed":preset not in {"adversarial", "meta-train"}, "reason":"Staged base-manifest selector is not wired yet." if preset in {"adversarial", "meta-train"} else None},
        {"id":"disk", "label":"At least 2 GB free disk for logs and checkpoints", "availability":"available" if free >= 2*1024**3 else "error", "passed":free >= 2*1024**3, "reason":None if free >= 2*1024**3 else "Insufficient free disk space."},
        {"id":"venv", "label":"Project virtual-environment interpreter resolves", "availability":"available" if python_exists else "unavailable", "passed":python_exists, "reason":None if python_exists else "The venv interpreter is not present; fallback launch is disabled."},
        {"id":"concurrency", "label":"No other supervised training process is running", "availability":"available" if not has_running else "error", "passed":not has_running, "reason":"Another child process owns the control plane." if has_running else None},
        {"id":"config", "label":"TRADING_MODE and config can be validated", "availability":"available" if config_ok else "error", "passed":config_ok, "reason":None if config_ok else "core.config.load_config() failed."},
    ]
    return envelope(checks, source="backend/main.py · repository checks")

@app.put("/api/config/env")
def write_env(body: RiskConfigRequest) -> dict[str, Any]:
    if not body.changes: raise HTTPException(422, "No configuration changes were provided.")
    unknown = set(body.changes) - RISK_KEYS
    if unknown: raise HTTPException(422, f"Keys are not whitelisted: {', '.join(sorted(unknown))}")
    env_path = ROOT / ".env"
    try:
        current = {key: str(value) for key, value in dotenv_values(env_path).items() if value is not None}
        current.update({key: value for key, value in os.environ.items()})
        changes = {key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in body.changes.items()}
        merged = {**current, **changes}
        load_config(_env=merged)
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        seen: set[str] = set()
        output: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in line:
                output.append(line); continue
            key = line.split("=", 1)[0].strip()
            if key in changes:
                output.append(f"{key}={changes[key]}"); seen.add(key)
            else: output.append(line)
        output.extend(f"{key}={value}" for key, value in changes.items() if key not in seen)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=env_path.parent, delete=False) as temp:
            temp.write("\n".join(output) + "\n"); temp_path = Path(temp.name)
        os.replace(temp_path, env_path)
        return envelope({"written": sorted(changes), "applies_on_next_start": True}, source=".env · validated with core.config.load_config()")
    except Exception as exc:
        raise HTTPException(422, f"Configuration was not written: {exc}") from exc

@app.post("/api/live/killswitch")
def killswitch(body: KillSwitchRequest) -> dict[str, Any]:
    try: cfg = load_config()
    except Exception as exc: raise HTTPException(422, f"Cannot resolve kill-switch path: {exc}") from exc
    path = cfg.paths.kill_switch_path
    try:
        if body.armed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        else:
            path.unlink(missing_ok=True)
        return envelope({"armed": path.is_file(), "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)}, source="KILL_SWITCH · sentinel file only")
    except OSError as exc: raise HTTPException(500, f"Sentinel file was not changed: {exc}") from exc

async def socket_loop(ws: WebSocket, channel: str) -> None:
    await ws.accept()
    try:
        while True:
            if channel == "training":
                job_rows = supervisor.list_jobs()
                active = next((job for job in job_rows if job["status"] in {"running", "starting", "stopping", "unverified"}), None)
                metric_data = supervisor.metric_samples(active["job_id"]) if active and active.get("metrics_path") else {"samples": [], "source": None}
                if active and metric_data["samples"]:
                    active["step"] = metric_data["samples"][-1]["training_step"]
                payload = envelope({"jobs": job_rows, "active_job": active, "metrics_samples": metric_data["samples"], "metrics_source": metric_data["source"]}, source="/ws/training", availability="available", reason=None)
            elif channel == "live/risk": payload = risk_state()
            elif channel == "live/decisions": payload = decisions()
            elif channel == "logs": payload = envelope(availability="unavailable", source="logs/<job_id>.log", reason="No selected running job log stream.")
            else: payload = envelope(availability="unavailable", source="artifacts/models/*", reason="No new checkpoint event has been observed.")
            await ws.send_json(payload)
            await asyncio.sleep(2 if channel != "logs" else 1)
    except WebSocketDisconnect:
        return

@app.websocket("/ws/training")
async def ws_training(ws: WebSocket) -> None: await socket_loop(ws, "training")
@app.websocket("/ws/live/decisions")
async def ws_decisions(ws: WebSocket) -> None: await socket_loop(ws, "live/decisions")
@app.websocket("/ws/live/risk")
async def ws_risk(ws: WebSocket) -> None: await socket_loop(ws, "live/risk")
@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket) -> None: await socket_loop(ws, "logs")
@app.websocket("/ws/checkpoints")
async def ws_checkpoints(ws: WebSocket) -> None: await socket_loop(ws, "checkpoints")

# Serve the built SPA from the same process when a production build exists. Vite remains
# the development server; its API and WebSocket requests are proxied to this app.
DIST = ROOT / "frontend_backend" / "frontend" / "dist"
if not DIST.is_dir():
    DIST = ROOT / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")
    @app.get("/{path:path}")
    def spa(path: str) -> Any:
        target = DIST / path
        if target.is_file(): return FileResponse(target)
        return FileResponse(DIST / "index.html")
