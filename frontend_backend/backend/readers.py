from __future__ import annotations

import csv
import json
import sqlite3
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import AppConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "artifacts" / "models"


def envelope(data: Any = None, *, availability: str = "available", source: str | None = None,
             reason: str | None = None, updated_at: str | None = None) -> dict[str, Any]:
    return {"availability": availability, "source": source, "updated_at": updated_at,
            "data": data, "reason": reason}


def evaluations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not MODEL_ROOT.is_dir():
        return rows
    for path in MODEL_ROOT.glob("*/evaluation.json"):
        if path.parent.name == "production":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            thresholds = payload["thresholds"]
            failures = []
            if metrics.get("sharpe", float("-inf")) < thresholds.get("sharpe", 0.5):
                failures.append("Sharpe below threshold")
            if metrics.get("max_dd_pct", float("inf")) > thresholds.get("max_dd_pct", 20.0):
                failures.append("Max drawdown above threshold")
            if metrics.get("trades", 0) < thresholds.get("trades", 20):
                failures.append("Trade count below threshold")
            if metrics.get("total_return_pct", float("-inf")) <= thresholds.get("total_return_pct", 0.0):
                failures.append("Total return does not exceed threshold")
            manifest_path = path.parent / "manifest.json"
            model_type = "unknown"
            if manifest_path.is_file():
                try:
                    model_type = json.loads(manifest_path.read_text(encoding="utf-8")).get("model_type", "unknown")
                except (OSError, ValueError):
                    pass
            rows.append({"run_name": path.parent.name, "model_type": model_type,
                         "source_path": path.relative_to(ROOT).as_posix(),
                         "evaluated_at": payload.get("evaluated_at", ""),
                         "test_start": payload.get("test_start"), "test_end": payload.get("test_end"),
                         "contract_hash": payload.get("contract_hash"), "metrics": metrics,
                         "thresholds": thresholds, "passed": bool(payload.get("passed", False)),
                         "failures": failures})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(rows, key=lambda row: row["evaluated_at"], reverse=True)


def _risk_database_loadable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=0.3) as con:
            con.execute("PRAGMA query_only=ON")
            row = con.execute("SELECT daily_pnl,current_equity FROM risk_state WHERE id=1").fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def dashboard_status(cfg: AppConfig | None = None) -> dict[str, Any]:
    try:
        cfg = cfg or load_config()
    except Exception as exc:
        return envelope(availability="error", source="core/config.py", reason=f"Configuration could not be loaded: {exc}")
    manifest = cfg.model.manifest_path if cfg.model.manifest_path.is_absolute() else ROOT / cfg.model.manifest_path
    feature_contract_path = cfg.paths.feature_contract_path if cfg.paths.feature_contract_path.is_absolute() else ROOT / cfg.paths.feature_contract_path
    risk_state_path = cfg.paths.risk_state_db if cfg.paths.risk_state_db.is_absolute() else ROOT / cfg.paths.risk_state_db
    eval_rows = evaluations()
    mode_source = "core/config.py · TRADING_MODE (.env override)" if "TRADING_MODE" in cfg.raw else "core/config.py · TRADING_MODE (default: demo)"
    production_required = cfg.model.signal_source == "model"
    live_checks = [
        ("Feature contract", feature_contract_path.is_file(), "feature contract present"),
        ("Production model for model signals", manifest.is_file() if production_required else None, "production manifest present" if production_required else "Not required by SIGNAL_SOURCE=rule"),
        ("Risk state", _risk_database_loadable(risk_state_path), "risk state loadable"),
        ("Reconciliation", None, "No trusted reconciliation result has been loaded"),
        ("MT5 credentials configured", bool(cfg.broker.mt5_login and cfg.broker.mt5_password and cfg.broker.mt5_server), "MT5 credentials configured"),
    ]
    runtime_gates = [{"name": name, "availability": "not_checked" if passed is None else "available" if passed else "unavailable",
                      "passed": passed, "reason": None if passed is True else reason}
                     for name, passed, reason in live_checks]
    return envelope({
        "trading_mode": cfg.trading_mode.value,
        "signal_source": cfg.model.signal_source,
        "trading_mode_source": mode_source,
        "require_promoted_model": cfg.behavior.require_promoted_model,
        "production_manifest": "available" if manifest.is_file() else "unavailable",
        "production_manifest_path": manifest.relative_to(ROOT).as_posix() if manifest.is_relative_to(ROOT) else str(manifest),
        "evaluations_passed": sum(1 for row in eval_rows if row["passed"]),
        "evaluations_inspected": len(eval_rows),
        "supervisor": "available",
        "kill_switch_armed": cfg.paths.kill_switch_path.is_file(),
        "runtime_gates": runtime_gates,
    }, source="core/config.py · artifacts/models/*/evaluation.json · KILL_SWITCH")


def risk_state(cfg: AppConfig | None = None) -> dict[str, Any]:
    try:
        cfg = cfg or load_config()
    except Exception as exc:
        return envelope(availability="error", source="core/config.py", reason=str(exc))
    risk = cfg.risk
    behavior = cfg.behavior
    config: dict[str, Any] = {
        "MAX_DAILY_LOSS": risk.max_daily_loss, "MAX_DRAWDOWN": risk.max_drawdown,
        "MAX_CONSECUTIVE_LOSSES": risk.max_consecutive_losses, "MAX_RISK_PER_TRADE": risk.risk_per_trade,
        "MAX_POSITION": risk.max_position, "MAX_TRADES_PER_DAY": risk.max_trades_per_day,
        "MIN_TRADE_INTERVAL_SEC": risk.min_trade_interval_sec, "VOL_THRESHOLD": risk.vol_threshold,
        "MAX_SPREAD": risk.max_spread, "MARKET_HOURS_ONLY": risk.market_hours_only,
        "CORRELATION_GUARD_ENABLED": risk.correlation_guard_enabled, "CORRELATION_ASSET": risk.correlation_asset,
        "CORRELATION_BLOCK_LONG_ON_UP": risk.correlation_block_long_on_up, "MIN_CONFIDENCE": behavior.min_confidence,
        "MIN_ENSEMBLE_AGREEMENT": behavior.min_ensemble_agreement, "USE_KELLY": behavior.use_kelly,
        "KELLY_FRACTION": behavior.kelly_fraction, "KELLY_MIN_TRADES": behavior.kelly_min_trades,
    }
    source = "core/config.py · .env + validated defaults"
    stats: dict[str, Any] = {"statistics_availability": "unavailable",
                             "statistics_reason": "risk_state.db is not present or has no singleton row.",
                             "config": config, "config_source": source}
    db_path = cfg.paths.risk_state_db if cfg.paths.risk_state_db.is_absolute() else ROOT / cfg.paths.risk_state_db
    if db_path.is_file():
        try:
            uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=0.3) as con:
                con.execute("PRAGMA query_only=ON")
                row = con.execute("SELECT daily_pnl,daily_start_equity,peak_equity,current_equity,trades_today,consecutive_losses,halt_until_ts,last_trade_ts,trade_date FROM risk_state WHERE id=1").fetchone()
            if row:
                daily_pnl, daily_start, peak, current, trades, losses, halt_ts, last_ts, trade_date = row
                stats.update({
                    "daily_pnl": daily_pnl,
                    "daily_start_equity": daily_start,
                    "peak_equity": peak,
                    "current_equity": current,
                    "trades_today": trades,
                    "consecutive_losses": losses,
                    "halt_remaining_sec": max(0.0, float(halt_ts) - datetime.now(timezone.utc).timestamp()) if halt_ts else 0.0,
                    "last_trade_at": datetime.fromtimestamp(last_ts, timezone.utc).isoformat() if last_ts else None,
                    "trade_date": trade_date,
                    "daily_loss_ratio": daily_pnl / daily_start if daily_start else None,
                    "statistics_availability": "available",
                    "statistics_reason": None,
                })
                source = db_path.relative_to(ROOT).as_posix()
        except (sqlite3.Error, OSError, ValueError) as exc:
            stats["statistics_reason"] = f"Risk database could not be read safely: {exc}"
    stats["config_source"] = source + (" · runtime + config" if stats["statistics_availability"] == "available" else " · resolved config")
    return envelope(stats, source=source)


def state_file(name: str) -> dict[str, Any]:
    if name not in {"positions", "closed"}:
        return envelope(availability="unavailable", source="state/mock_state.json", reason="This state panel is not backed by a persisted source.")
    try:
        cfg = load_config()
    except Exception as exc:
        return envelope(availability="error", source="core/config.py", reason=f"Cannot determine which account owns this state: {exc}")
    if cfg.trading_mode.value != "demo":
        return envelope(availability="unavailable", source="state/mock_state.json", reason="MockBroker state is hidden while TRADING_MODE is not demo; no broker-backed position reader is connected.")
    path = ROOT / "state" / "mock_state.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = "positions" if name == "positions" else "closed"
        return envelope(payload.get(key, []), source="state/mock_state.json", updated_at=None)
    except (OSError, ValueError) as exc:
        return envelope(availability="unavailable", source="state/mock_state.json", reason=str(exc))


def market_bars(symbol: str, timeframe: str, limit: int) -> dict[str, Any]:
    if symbol.upper() not in {"XAUUSD", "XAUUSDM"}:
        return envelope(availability="error", source="data/", reason="Only XAUUSD history is registered in the local CSV reader.")
    files = {"H1": "xauusd_h1.csv", "H4": "xauusd_h4.csv", "D1": "xauusd_d1.csv", "M1": "xauusd_m1.csv", "M5": "xauusd_m5.csv", "M15": "xauusd_m15.csv"}
    filename = files.get(timeframe.upper())
    if not filename:
        return envelope(availability="error", source="data/", reason="Unsupported historical timeframe.")
    path = ROOT / "data" / filename
    if not path.is_file():
        return envelope(availability="unavailable", source=f"data/{filename}", reason="Historical CSV file is absent.")
    try:
        rows: deque[dict[str, Any]] = deque(maxlen=min(max(limit, 1), 10_000))
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"time", "open", "high", "low", "close"}
            if not required.issubset(reader.fieldnames or []):
                return envelope(availability="error", source=f"data/{filename}", reason="CSV is missing required OHLC columns.")
            for row in reader:
                try:
                    stamp = datetime.fromisoformat(row["time"].strip())
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=timezone.utc)
                    bar = {"time": int(stamp.timestamp() * 1000),
                           "open": float(row["open"]), "high": float(row["high"]),
                           "low": float(row["low"]), "close": float(row["close"])}
                    if row.get("volume") not in (None, ""):
                        bar["volume"] = float(row["volume"])
                    rows.append(bar)
                except (ValueError, TypeError):
                    continue
        bars = list(rows)
        if not bars:
            return envelope(availability="unavailable", source=f"data/{filename}", reason="CSV has no valid OHLC rows.")
        last = datetime.fromtimestamp(bars[-1]["time"] / 1000, timezone.utc).isoformat()
        return envelope({"symbol": "XAUUSD", "timeframe": timeframe.upper(), "bars": bars,
                         "last_bar_time": last, "historical": True}, source=f"data/{filename}")
    except OSError as exc:
        return envelope(availability="error", source=f"data/{filename}", reason=str(exc))


def reconciliation() -> dict[str, Any]:
    return envelope({"availability": "unavailable", "reason": "No persisted reconciliation result is available; startup_reconcile runs inside the live child."},
                    availability="unavailable", source="live/trade_executor.py · startup_reconcile")


def decisions() -> dict[str, Any]:
    return envelope(availability="unavailable", source="/ws/live/decisions", reason="No supervisor decision event has been emitted.")


def jobs_snapshot(supervisor: Any) -> dict[str, Any]:
    return envelope(supervisor.list_jobs(), source="backend/supervisor.py")


def resumable_models() -> list[dict[str, Any]]:
    models_dir = ROOT / "artifacts" / "models"
    if not models_dir.is_dir():
        return []
    results: list[dict[str, Any]] = []
    for dir_path in sorted(models_dir.iterdir()):
        if not dir_path.is_dir():
            continue
        zip_files = list(dir_path.glob("*.zip"))
        if not zip_files:
            continue
        ckpts = list(dir_path.glob("checkpoint_*.zip"))
        latest_ckpt: Path | None = None
        latest_step = 0
        if ckpts:
            def step_val(p: Path) -> int:
                try:
                    return int(p.stem.split("_")[1])
                except (IndexError, ValueError):
                    return int(p.stat().st_mtime)
            ckpts_sorted = sorted(ckpts, key=step_val)
            latest_ckpt = ckpts_sorted[-1]
            try:
                latest_step = int(latest_ckpt.stem.split("_")[1])
            except (IndexError, ValueError):
                latest_step = 0
        elif (dir_path / "model.zip").exists():
            latest_ckpt = dir_path / "model.zip"

        if latest_ckpt is None:
            continue

        manifest_path = dir_path / "manifest.json"
        eval_path = dir_path / "evaluation.json"
        manifest_data: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass

        model_type = manifest_data.get("model_type")
        if not model_type:
            model_type = "ppo" if "ppo" in dir_path.name.lower() else "dreamer"

        results.append({
            "run_name": dir_path.name,
            "model_type": model_type,
            "latest_checkpoint": latest_ckpt.relative_to(ROOT).as_posix(),
            "latest_checkpoint_name": latest_ckpt.name,
            "latest_step": latest_step or manifest_data.get("hyperparams", {}).get("timesteps", 0),
            "window": manifest_data.get("window", 64),
            "n_features": manifest_data.get("n_features", 15),
            "has_macro": bool(manifest_data.get("n_features", 15) > 15),
            "train_end": manifest_data.get("train_end", "2024-01-01"),
            "allow_short": manifest_data.get("allow_short", True),
            "has_model_zip": (dir_path / "model.zip").is_file(),
            "has_evaluation": eval_path.is_file(),
            "checkpoints_count": len(ckpts),
        })
    return sorted(results, key=lambda x: x["latest_step"], reverse=True)

