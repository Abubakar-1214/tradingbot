from __future__ import annotations

import json
import os
import subprocess
import sys

import psutil
import pytest

from frontend_backend.backend.supervisor import ProcessSupervisor, SupervisorError


def _sleep_command() -> list[str]:
    return [sys.executable, "-c", "import time; time.sleep(60)"]


def test_supervisor_restarts_adopt_child_and_can_kill_it(tmp_path) -> None:
    registry = tmp_path / "jobs.json"
    first = ProcessSupervisor(registry_path=registry)
    record = first._spawn("recovery-test", _sleep_command(), preset="test", script="test-child", total_steps=None)
    pid = record["pid"]

    first.shutdown()
    assert psutil.pid_exists(pid), "API shutdown must detach the child, not kill it"

    recovered = ProcessSupervisor(registry_path=registry)
    try:
        job = next(row for row in recovered.list_jobs() if row["job_id"] == "recovery-test")
        assert job["status"] == "running"
        assert job["recovery_state"] == "adopted"
        assert job["pid"] == pid

        result = recovered.terminate("recovery-test", hard=True)
        assert result["status"] == "stopping"
        try:
            psutil.Process(pid).wait(timeout=5)
        except psutil.NoSuchProcess:
            pass
        final = next(row for row in recovered.list_jobs() if row["job_id"] == "recovery-test")
        assert final["status"] == "killed"
    finally:
        recovered.shutdown()
        try:
            psutil.Process(pid).kill()
        except psutil.NoSuchProcess:
            pass


def test_live_pid_without_matching_token_is_unverified_and_never_signalled(tmp_path) -> None:
    command = _sleep_command()
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             env={**os.environ, "DASHBOARD_SUPERVISOR_TOKEN": "different-job-token"})
    supervisor = None
    try:
        created = psutil.Process(child.pid).create_time()
        registry = tmp_path / "jobs.json"
        registry.write_text(json.dumps({
            "version": ProcessSupervisor.REGISTRY_VERSION,
            "jobs": [{
                "job_id": "possibly-reused-pid", "preset": "test", "script": "test-child",
                "status": "running", "pid": child.pid, "started_at": "test", "return_code": None,
                "log_path": "logs/test.log", "metrics_path": None,
                "_process_token": "expected-job-token",
                "_process_cmdline_hash": ProcessSupervisor._cmdline_hash(command),
                "_process_create_time": created,
            }],
        }), encoding="utf-8")

        supervisor = ProcessSupervisor(registry_path=registry)
        job = supervisor.list_jobs()[0]
        assert job["status"] == "unverified"
        assert job["recovery_state"] == "pid_identity_mismatch"
        with pytest.raises(SupervisorError, match="No running supervised child"):
            supervisor.terminate("possibly-reused-pid", hard=True)
        assert child.poll() is None
    finally:
        if supervisor is not None:
            supervisor.shutdown()
        child.kill()
        child.wait(timeout=5)


def test_registry_lock_allows_only_one_writer(tmp_path) -> None:
    registry = tmp_path / "jobs.json"
    first = ProcessSupervisor(registry_path=registry)
    try:
        with pytest.raises(SupervisorError, match="owns the persistent job registry"):
            ProcessSupervisor(registry_path=registry)
    finally:
        first.shutdown()

    replacement = ProcessSupervisor(registry_path=registry)
    replacement.shutdown()


def test_corrupt_registry_fails_closed_without_overwriting(tmp_path) -> None:
    registry = tmp_path / "jobs.json"
    original = "{not-json"
    registry.write_text(original, encoding="utf-8")

    with pytest.raises(SupervisorError, match="cannot be read safely"):
        ProcessSupervisor(registry_path=registry)

    assert registry.read_text(encoding="utf-8") == original
