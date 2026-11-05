from __future__ import annotations

import copy
import json
import os
import re
from collections import deque
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[2]
PRESET_SCRIPTS = {
    "dreamer": "train/train_dreamer.py", "transformer": "train/train_transformer.py",
    "ppo": "train/train_ppo.py", "ensemble": "train/train_ensemble.py",
    "god-mode": "train/train_god_mode.py", "ultimate-150": "train/train_ultimate_150.py",
    "adversarial": "train/train_adversarial.py", "meta-train": "train/meta_train_dreamer.py",
}
DREAMER_FLAGS = {"data", "macro", "train-end", "test-end", "run-name", "resume",
                 "steps", "prefill", "save-every", "train-every", "batch-size", "seq-len",
                 "window", "embed-dim", "hidden-dim", "stoch-dim", "num-categories", "horizon",
                 "seed", "device", "sl-tp-model", "allow-short", "long-only"}
PPO_FLAGS = {"data", "macro", "train-end", "test-end", "run-name", "resume",
             "timesteps", "steps", "n-envs", "subproc", "seed", "window",
             "n-steps", "batch-size", "chunk-steps", "max-episode-steps",
             "hidden-sizes", "allow-short", "long-only", "learning-rate", "gamma",
             "turnover-penalty", "hold-bonus", "min-hold-bars", "early-exit-penalty",
             "drawdown-penalty", "short-penalty"}

class SupervisorError(ValueError):
    pass

class ProcessSupervisor:
    REGISTRY_VERSION = 1
    ACTIVE_STATUSES = {"starting", "running", "stopping", "unverified"}

    def __init__(self, registry_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self.registry_path = (registry_path or ROOT / "state" / "supervisor_jobs.json").resolve()
        self.lock_path = self.registry_path.with_suffix(self.registry_path.suffix + ".lock")
        self._lock_handle: Any = None
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen[bytes] | psutil.Process] = {}
        self.log_dir = ROOT / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._acquire_registry_lock()
            try:
                self._load_registry()
                self._recover_jobs()
                self._persist_registry()
            except Exception:
                self._release_registry_lock()
                raise

    def _acquire_registry_lock(self) -> None:
        if self._lock_handle is not None:
            return
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise SupervisorError("Another dashboard supervisor owns the persistent job registry.") from exc
        self._lock_handle = handle

    def _release_registry_lock(self) -> None:
        handle, self._lock_handle = self._lock_handle, None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def _load_registry(self) -> None:
        if not self.registry_path.exists():
            self._jobs = {}
            return
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if payload.get("version") != self.REGISTRY_VERSION or not isinstance(payload.get("jobs"), list):
                raise ValueError("unsupported registry version or invalid jobs list")
            jobs: dict[str, dict[str, Any]] = {}
            for row in payload["jobs"]:
                if not isinstance(row, dict) or not isinstance(row.get("job_id"), str):
                    continue
                record = dict(row)
                record["log_path"] = self._safe_relative_path(record.get("log_path"), "log_path")
                if record.get("metrics_path"):
                    record["metrics_path"] = self._safe_relative_path(record["metrics_path"], "metrics_path")
                if record.get("pid") is not None and (isinstance(record["pid"], bool) or not isinstance(record["pid"], int) or record["pid"] <= 0):
                    record["status"] = "interrupted"
                    record["recovery_state"] = "invalid_persisted_pid"
                    record["pid"] = None
                jobs[record["job_id"]] = record
            self._jobs = jobs
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            raise SupervisorError(f"Persistent supervisor registry cannot be read safely: {exc}") from exc

    @staticmethod
    def _safe_relative_path(value: Any, field: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a relative repository path")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{field} escapes the repository root")
        return path.as_posix()

    def _persist_registry(self) -> None:
        payload = {"version": self.REGISTRY_VERSION,
                   "updated_at": datetime.now(timezone.utc).isoformat(),
                   "jobs": list(self._jobs.values())}
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline=None,
                                             dir=self.registry_path.parent, prefix=".jobs-", suffix=".tmp",
                                             delete=False) as handle:
                json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
                handle.write(chr(10))
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, self.registry_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _public_job(record: dict[str, Any]) -> dict[str, Any]:
        return {key: copy.deepcopy(value) for key, value in record.items() if not key.startswith("_")}

    def _find_spawned_process(self, record: dict[str, Any]) -> psutil.Process | None:
        token = record.get("_process_token")
        expected_hash = record.get("_process_cmdline_hash")
        if not isinstance(token, str) or not isinstance(expected_hash, str):
            return None
        matches: list[psutil.Process] = []
        for candidate in psutil.process_iter():
            try:
                if self._matches_persisted_process(candidate, record, require_create_time=False):
                    matches.append(candidate)
                    if len(matches) > 1:
                        return None
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
        return matches[0] if len(matches) == 1 else None

    def _recover_jobs(self) -> None:
        self._processes.clear()
        changed = False
        for job_id, record in self._jobs.items():
            if record.get("status") not in self.ACTIVE_STATUSES:
                continue
            pid = record.get("pid")
            try:
                process = psutil.Process(pid) if isinstance(pid, int) else self._find_spawned_process(record)
                if process is None:
                    status = "interrupted" if isinstance(pid, int) or not record.get("_process_token") else "unverified"
                    reason = "process_missing" if status == "interrupted" else "spawn_identity_unresolved"
                    record.update(status=status, recovery_state=reason, ended_at=datetime.now(timezone.utc).isoformat())
                    changed = True
                    continue
                if not isinstance(pid, int):
                    record["pid"] = process.pid
                    record["_process_create_time"] = process.create_time()
                if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                    record.update(status="interrupted", recovery_state="process_not_running", ended_at=datetime.now(timezone.utc).isoformat())
                    changed = True
                    continue
                if not self._matches_persisted_process(process, record):
                    # The PID is alive but cannot be proven to be ours. Keep the global
                    # single-child lock closed and never send signals to that process.
                    record.update(status="unverified", recovery_state="pid_identity_mismatch")
                    changed = True
                    continue
                self._processes[job_id] = process
                record.update(status="running", recovery_state="adopted")
            except psutil.NoSuchProcess:
                record.update(status="interrupted", recovery_state="process_missing", ended_at=datetime.now(timezone.utc).isoformat())
                changed = True
            except (psutil.AccessDenied, OSError):
                # A live PID that cannot be verified blocks another launch, but is never signalled.
                record.update(status="unverified", recovery_state="identity_unreadable")
                changed = True
        if changed:
            self._persist_registry()

    @staticmethod
    def _matches_persisted_process(process: psutil.Process, record: dict[str, Any], *, require_create_time: bool = True) -> bool:
        expected_created = record.get("_process_create_time")
        expected_token = record.get("_process_token")
        expected_cmdline = record.get("_process_cmdline_hash")
        if not isinstance(expected_token, str) or not isinstance(expected_cmdline, str):
            return False
        if require_create_time and not isinstance(expected_created, (int, float)):
            return False
        if isinstance(expected_created, (int, float)) and abs(process.create_time() - float(expected_created)) > 0.02:
            return False
        environment = process.environ()
        if environment.get("DASHBOARD_SUPERVISOR_TOKEN") != expected_token:
            return False
        cmdline = process.cmdline()
        actual_hash = __import__("hashlib").sha256(json.dumps(cmdline, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        return actual_hash == expected_cmdline

    @staticmethod
    def _cmdline_hash(argv: list[str]) -> str:
        import hashlib
        return hashlib.sha256(json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _refresh_processes(self) -> None:
        dirty = False
        for job_id, process in list(self._processes.items()):
            record = self._jobs[job_id]
            if isinstance(process, subprocess.Popen):
                code = process.poll()
                alive = code is None
            else:
                try:
                    alive = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                    code = None
                except psutil.NoSuchProcess:
                    alive, code = False, None
                except psutil.AccessDenied:
                    record.update(status="unverified", recovery_state="identity_unreadable")
                    self._processes.pop(job_id, None)
                    dirty = True
                    continue
            if not alive:
                requested = record.pop("_requested_terminal_status", None)
                if requested:
                    record["status"] = requested
                elif isinstance(process, psutil.Process):
                    record.update(status="interrupted", recovery_state="process_exited_after_adoption")
                else:
                    record["status"] = "finished" if code == 0 else "failed"
                    record["return_code"] = code
                record["ended_at"] = datetime.now(timezone.utc).isoformat()
                self._processes.pop(job_id, None)
                dirty = True
            elif record.get("status") not in {"stopping", "killed"}:
                if record.get("status") != "running":
                    record["status"] = "running"
                    dirty = True
        if dirty:
            self._persist_registry()

    def _ensure_registry(self) -> None:
        had_lock = self._lock_handle is not None
        self._acquire_registry_lock()
        if not had_lock:
            self._load_registry()
            self._recover_jobs()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_registry()
            self._refresh_processes()
            return [self._public_job(job) for job in reversed(list(self._jobs.values()))]

    def _spawn(self, job_id: str, argv: list[str], *, preset: str, script: str, total_steps: int | None, collect_metrics: bool = False) -> dict[str, Any]:
        with self._lock:
            self._ensure_registry()
            self._refresh_processes()
            if job_id in self._jobs:
                if self._jobs[job_id].get("status") in self.ACTIVE_STATUSES:
                    raise SupervisorError(f"Job id '{job_id}' is currently running or active.")
            if any(job.get("status") in self.ACTIVE_STATUSES for job in self._jobs.values()):
                raise SupervisorError("Another supervised child is running or unverified. Only one child process is allowed.")
            if shutil.disk_usage(ROOT).free < 2 * 1024**3:
                raise SupervisorError("At least 2 GB free disk space is required before launch.")
            log_path = self.log_dir / f"{job_id}.log"
            log_file = log_path.open("ab")
            kwargs: dict[str, Any] = {"cwd": ROOT, "stdout": log_file, "stderr": subprocess.STDOUT}
            metrics_path = self.log_dir / f"{job_id}.metrics.jsonl"
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(ROOT)
            process_token = uuid.uuid4().hex
            env["DASHBOARD_SUPERVISOR_TOKEN"] = process_token
            if collect_metrics:
                env["DASHBOARD_METRICS_PATH"] = str(metrics_path)
            kwargs["env"] = env
            if os.name == "nt": kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else: kwargs["start_new_session"] = True
            log_relative = log_path.relative_to(ROOT).as_posix()
            metrics_relative = metrics_path.relative_to(ROOT).as_posix() if collect_metrics else None
            started_at = datetime.now(timezone.utc).isoformat()
            record = {"job_id": job_id, "preset": preset, "script": script, "status": "starting",
                      "pid": None, "started_at": started_at, "step": None, "total_steps": total_steps,
                      "return_code": None, "log_path": log_relative, "metrics_path": metrics_relative,
                      "_process_token": process_token, "_process_cmdline_hash": self._cmdline_hash(argv),
                      "stop_capability": "process termination; training scripts do not implement a cooperative stop sentinel"}
            self._jobs[job_id] = record
            self._persist_registry()
            try:
                process = subprocess.Popen(argv, **kwargs)
            except OSError:
                log_file.close()
                self._jobs.pop(job_id, None)
                self._persist_registry()
                raise
            finally:
                if not log_file.closed:
                    log_file.close()
            try:
                created = psutil.Process(process.pid).create_time()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                created = None
            record.update(status="running", pid=process.pid, _process_create_time=created)
            self._processes[job_id] = process
            self._persist_registry()
            return self._public_job(record)

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        preset = payload.get("preset")
        script = PRESET_SCRIPTS.get(preset)
        if not script: raise SupervisorError("Unknown preset.")
        if preset not in {"dreamer", "ppo"}:
            raise SupervisorError("Only the Dreamer and PPO launch forms have a complete validated flag schema. Other preset scripts are listed but cannot launch from this version.")
        run_name = payload.get("run_name", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_name): raise SupervisorError("Run name must use letters, numbers, underscore, or hyphen (max 64 characters).")
        flags = payload.get("flags") or {}
        allowed_flags = DREAMER_FLAGS if preset == "dreamer" else PPO_FLAGS
        unknown = set(flags) - allowed_flags
        if unknown: raise SupervisorError(f"Unsupported {preset.upper()} flags: {', '.join(sorted(unknown))}")
        if flags.get("run-name") not in (None, run_name): raise SupervisorError("Run name flag must match the requested artifact directory.")
        data_path = Path(str(flags.get("data", "data/xauusd_h1.csv")))
        if not data_path.is_absolute(): data_path = ROOT / data_path
        try: data_path.resolve().relative_to((ROOT / "data").resolve())
        except ValueError as exc: raise SupervisorError("Data path must remain inside data/.") from exc
        if not data_path.is_file() or data_path.stat().st_size == 0: raise SupervisorError("Selected OHLC CSV is missing or empty.")
        macro = flags.get("macro")
        macro_path = None
        if macro:
            macro_path = Path(str(macro))
            if not macro_path.is_absolute(): macro_path = ROOT / macro_path
            try: macro_path.resolve().relative_to((ROOT / "data").resolve())
            except ValueError as exc: raise SupervisorError("Macro CSV must remain inside data/.") from exc
            if not macro_path.is_file() or macro_path.stat().st_size == 0: raise SupervisorError("Selected macro CSV is missing or empty.")
        numeric_ranges = {
            "steps": (1, 100_000_000), "timesteps": (1, 100_000_000),
            "prefill": (0, 10_000_000), "save-every": (1, 10_000_000),
            "train-every": (1, 100_000), "batch-size": (1, 4096), "seq-len": (1, 4096),
            "window": (1, 4096), "embed-dim": (8, 8192), "hidden-dim": (8, 8192),
            "stoch-dim": (1, 2048), "num-categories": (1, 1024), "horizon": (1, 1000), "seed": (0, 2**31 - 1),
            "n-envs": (1, 64), "n-steps": (1, 100_000), "chunk-steps": (1, 1_000_000),
            "max-episode-steps": (1, 100_000), "min-hold-bars": (0, 100),
        }
        for key, (minimum, maximum) in numeric_ranges.items():
            if key not in flags: continue
            value = flags[key]
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise SupervisorError(f"{key} must be an integer in [{minimum}, {maximum}].")
        float_ranges = {
            "turnover-penalty": (0.0, 0.1),
            "hold-bonus": (0.0, 0.01),
            "early-exit-penalty": (0.0, 0.1),
            "drawdown-penalty": (0.0, 10.0),
            "short-penalty": (0.0, 0.1),
            "learning-rate": (1e-7, 1.0),
            "gamma": (0.5, 0.9999),
        }
        for key, (minimum, maximum) in float_ranges.items():
            if key not in flags: continue
            value = flags[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum:
                raise SupervisorError(f"{key} must be a number in [{minimum}, {maximum}].")
        if "device" in flags and flags["device"] not in {"auto", "cpu"}:
            raise SupervisorError("This environment is verified CPU-only; select auto or cpu.")
        if preset == "ppo":
            steps = int(flags.get("timesteps") or flags.get("steps") or 1_000_000)
        else:
            steps = int(flags["steps"]) if "steps" in flags and flags["steps"] is not None else 100_000
        window = int(flags.get("window", 64))
        try:
            from datetime import date
            date.fromisoformat(str(flags.get("train-end", "2022-01-01")))
        except ValueError as exc:
            raise SupervisorError("train-end must be an ISO date (YYYY-MM-DD).") from exc
        train_end = str(flags.get("train-end", "2022-01-01"))
        try:
            from datetime import date
            date.fromisoformat(train_end)
            if flags.get("test-end"): date.fromisoformat(str(flags["test-end"]))
        except ValueError as exc:
            raise SupervisorError("train-end and test-end must be ISO dates (YYYY-MM-DD).") from exc
        resume = flags.get("resume")
        is_resume = bool(resume)
        if resume:
            resume_path = Path(str(resume))
            if not resume_path.is_absolute(): resume_path = ROOT / resume_path
            try: resume_path.resolve().relative_to((ROOT / "artifacts" / "models").resolve())
            except ValueError as exc: raise SupervisorError("Resume checkpoints must be inside artifacts/models/.") from exc
            if not resume_path.is_file(): raise SupervisorError("Resume checkpoint was not found.")
            manifest_path = resume_path.parent / "manifest.json"
            if manifest_path.is_file():
                try:
                    from core.model_artifacts import load_manifest
                    manifest = load_manifest(manifest_path)
                    if manifest.window != window or manifest.obs_dim != window * manifest.n_features + 5:
                        raise SupervisorError(f"Resume dimensions (window={manifest.window}, obs_dim={manifest.obs_dim}) do not match the selected observation window {window}.")
                    if manifest.n_features <= 15 and flags.get("macro"):
                        flags.pop("macro", None)
                    elif manifest.n_features > 15 and not flags.get("macro"):
                        flags["macro"] = "data/macro_daily.csv"
                except SupervisorError:
                    raise
                except Exception as exc:
                    raise SupervisorError(f"Resume artifact manifest is invalid: {exc}") from exc
        if not is_resume:
            report_path = ROOT / "logs" / "prep_latest.json"
            contract_path = ROOT / "train" / "feature_contract.json"
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise SupervisorError("A completed feature-contract preparation report is required before launch.") from exc
            macro_source = macro_path.relative_to(ROOT).as_posix() if macro_path else None
            expected_data = data_path.relative_to(ROOT).as_posix()
            if (report.get("source") != expected_data
                    or report.get("train_end") != train_end
                    or report.get("window") != int(flags.get("window", 64))
                    or report.get("macro_source") != macro_source
                    or report.get("contract_hash") != contract.get("hash")
                    or contract.get("pipeline_params", {}).get("window") != int(flags.get("window", 64))):
                raise SupervisorError("Active feature contract does not match the requested parameters.")
        out_dir = ROOT / "artifacts" / "models" / run_name
        if out_dir.exists() and not is_resume:
            raise SupervisorError("Artifact directory already exists.")
        script_path = ROOT / script
        if not script_path.is_file(): raise SupervisorError(f"Training script not found: {script}")
        python = ROOT / ".venv" / "Scripts" / "python.exe"
        if not python.is_file(): raise SupervisorError("The project venv interpreter was not found.")
        argv = [str(python), str(script_path)]
        for key, value in flags.items():
            if key in {"run-name", "artifact-root"}: continue
            if key in {"sl-tp-model", "allow-short", "long-only", "subproc"}:
                if value: argv.append(f"--{key}")
                continue
            if key == "hidden-sizes":
                if isinstance(value, list):
                    argv.extend(["--hidden-sizes", *[str(v) for v in value]])
                else:
                    argv.extend(["--hidden-sizes", *str(value).split()])
                continue
            if value is not None and value != "": argv.extend([f"--{key}", str(value)])
        argv.extend(["--run-name", run_name, "--artifact-root", str(ROOT / "artifacts" / "models")])
        return self._spawn(run_name, argv, preset=preset, script=script, total_steps=steps, collect_metrics=True)

    def start_prep(self, *, data: str, train_end: str, window: int, macro: str | None) -> dict[str, Any]:
        if not 1 <= window <= 4096: raise SupervisorError("Window must be between 1 and 4096.")
        data_path = Path(data)
        if not data_path.is_absolute(): data_path = ROOT / data_path
        try: data_path.resolve().relative_to((ROOT / "data").resolve())
        except ValueError as exc: raise SupervisorError("Data path must remain inside data/.") from exc
        if not data_path.is_file() or data_path.stat().st_size == 0: raise SupervisorError("Data file is missing or empty.")
        macro_path = None
        if macro:
            macro_path = Path(macro)
            if not macro_path.is_absolute(): macro_path = ROOT / macro_path
            try: macro_path.resolve().relative_to((ROOT / "data").resolve())
            except ValueError as exc: raise SupervisorError("Macro path must remain inside data/.") from exc
            if not macro_path.is_file() or macro_path.stat().st_size == 0: raise SupervisorError("Macro CSV is missing or empty.")
        script_path = Path(__file__).resolve().parent / "prepare_contract.py"
        python = ROOT / ".venv" / "Scripts" / "python.exe"
        if not python.is_file(): raise SupervisorError("The project venv interpreter was not found.")
        job_id = f"prep-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        argv = [str(python), str(script_path), "--data", str(data_path), "--train-end", train_end, "--window", str(window)]
        if macro_path: argv.extend(["--macro", str(macro_path)])
        return self._spawn(job_id, argv, preset="data-prep", script="backend/prepare_contract.py", total_steps=None)

    @staticmethod
    def _process_alive(process: subprocess.Popen[bytes] | psutil.Process) -> bool:
        if isinstance(process, subprocess.Popen):
            return process.poll() is None
        try:
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def terminate(self, job_id: str, *, hard: bool = False) -> dict[str, Any]:
        with self._lock:
            self._ensure_registry()
            self._refresh_processes()
            process = self._processes.get(job_id)
            if process is None or not self._process_alive(process):
                raise SupervisorError("No running supervised child has that job id.")
            record = self._jobs[job_id]
            if isinstance(process, psutil.Process) and not self._matches_persisted_process(process, record):
                record.update(status="interrupted", recovery_state="pid_identity_mismatch", ended_at=datetime.now(timezone.utc).isoformat())
                self._processes.pop(job_id, None)
                self._persist_registry()
                raise SupervisorError("Recovered process identity changed; refusing to signal an unverified PID.")
            record["status"] = "stopping"
            record["_requested_terminal_status"] = "killed" if hard else None
            self._persist_registry()
            try:
                if isinstance(process, subprocess.Popen):
                    if hard and os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                    elif hard and os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    elif os.name == "nt":
                        process.terminate()
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                else:
                    targets = [process, *process.children(recursive=True)]
                    for target in reversed(targets):
                        try:
                            target.kill() if hard else target.terminate()
                        except psutil.NoSuchProcess:
                            pass
            except (OSError, psutil.Error, subprocess.SubprocessError) as exc:
                record.update(status="unverified", recovery_state="termination_request_failed")
                self._persist_registry()
                raise SupervisorError(f"Could not signal the verified process: {exc}") from exc
            self._persist_registry()
            return self._public_job(record)

    def shutdown(self) -> None:
        """Persist active jobs and detach so a replacement API process can adopt them."""
        with self._lock:
            if self._lock_handle is None:
                return
            for job_id, process in list(self._processes.items()):
                if self._process_alive(process):
                    record = self._jobs[job_id]
                    if record.get("status") != "stopping":
                        record["status"] = "running"
                    self._processes.pop(job_id, None)
            self._persist_registry()
            self._release_registry_lock()

    def log_tail(self, job_id: str, limit: int = 200) -> dict[str, Any]:
        self.list_jobs()
        job = self._jobs.get(job_id)
        if not job: raise SupervisorError("Job not found.")
        path = ROOT / str(job["log_path"])
        if not path.is_file(): return {"lines": [], "source": str(job["log_path"])}
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = deque(handle, maxlen=max(1, min(limit, 2000)))
        return {"lines": [line.rstrip("\r\n") for line in lines], "source": str(job["log_path"])}

    def metric_samples(self, job_id: str, limit: int = 600) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job: raise SupervisorError("Job not found.")
        relative = job.get("metrics_path")
        if not relative: return {"samples": [], "source": None}
        path = ROOT / str(relative)
        if not path.is_file(): return {"samples": [], "source": str(relative)}
        samples: deque[dict[str, Any]] = deque(maxlen=max(1, min(limit, 2000)))
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                    if isinstance(payload, dict) and isinstance(payload.get("training_step"), int):
                        samples.append(payload)
                except ValueError:
                    continue
        return {"samples": list(samples), "source": str(relative)}
