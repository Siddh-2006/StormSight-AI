from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_state_endpoint():
    resp = client.get("/api/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["grid"] == {"h": 64, "w": 64}
    assert set(body["channels"]) == {"radar_dbz", "satellite_ir_k", "lightning_density", "cape", "shear"}


def test_nowcast_endpoint_baseline():
    resp = client.get("/api/nowcast", params={"horizon": 4, "model": "baseline"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "baseline"
    assert len(body["lead_times_min"]) == 4
    assert len(body["radar_forecast"]) == 4


def test_nowcast_endpoint_convlstm():
    resp = client.get("/api/nowcast", params={"horizon": 3, "model": "convlstm"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "convlstm"
    assert len(body["radar_forecast"]) == 3


def test_alerts_endpoint():
    resp = client.get("/api/alerts", params={"threshold": 0.9})
    assert resp.status_code == 200
    body = resp.json()
    assert "alerts" in body


def test_simulate_step_and_reset():
    resp = client.post("/api/simulate/step")
    assert resp.status_code == 200
    resp = client.post("/api/simulate/reset", json={"seed": 5, "num_cells": 2})
    assert resp.status_code == 200
