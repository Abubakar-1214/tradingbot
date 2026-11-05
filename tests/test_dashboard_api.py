from __future__ import annotations

from fastapi.testclient import TestClient

from frontend_backend.backend.main import app
from frontend_backend.backend.readers import evaluations


def test_models_endpoint_reads_evaluations() -> None:
    with TestClient(app) as client:
        response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["availability"] == "available"
    rows = payload["data"]
    assert len(rows) >= 1


def test_models_resumable_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/models/resumable")
    assert response.status_code == 200
    payload = response.json()
    assert payload["availability"] == "available"
    items = payload["data"]
    assert any(m["run_name"] == "ppo_18m" for m in items)
    ppo_18m = next(m for m in items if m["run_name"] == "ppo_18m")
    assert "checkpoint_" in ppo_18m["latest_checkpoint"]
    assert ppo_18m["window"] == 70


def test_historical_bars_come_from_repository_csv() -> None:
    with TestClient(app) as client:
        response = client.get("/api/market/bars", params={"symbol": "XAUUSD", "timeframe": "H1", "limit": 3})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["historical"] is True
    assert len(data["bars"]) == 3
    assert data["bars"][-1]["time"] > data["bars"][-2]["time"]
