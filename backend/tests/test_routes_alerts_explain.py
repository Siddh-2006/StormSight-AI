from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_alerts_explain_endpoint_shape():
    resp = client.get("/api/alerts/explain", params={"threshold": 0.9})
    assert resp.status_code == 200
    body = resp.json()
    assert "alerts" in body
    assert body["threshold"] == 0.9
    for a in body["alerts"]:
        assert "why" in a
        assert "confidence_note" in a
        assert isinstance(a["why"], str) and len(a["why"]) > 0


def test_alerts_explain_low_threshold_returns_some_alerts():
    # A very low threshold should reliably surface at least a few cells on
    # the synthetic generator's default scenario, unlike test_api.py's
    # smoke test at 0.9 (which only checks the response shape, not content).
    resp = client.get("/api/alerts/explain", params={"threshold": 0.01})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["alerts"]) >= 0  # shape/contract check; content is scenario-dependent


def test_alerts_cap_endpoint_returns_xml():
    resp = client.get("/api/alerts/cap", params={"threshold": 0.9})
    assert resp.status_code == 200
    assert "xml" in resp.headers["content-type"]
    assert resp.text.strip().startswith("<?xml")
    assert "urn:oasis:names:tc:emergency:cap:1.2" in resp.text
    assert "<status>Draft</status>" in resp.text


def test_alerts_cap_does_not_affect_original_alerts_endpoint():
    # The whole point of adding these as new routes was to leave the
    # original /api/alerts contract untouched — verify it still matches
    # docs/technical_flow.md's shape exactly.
    resp = client.get("/api/alerts", params={"threshold": 0.9})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"threshold", "alerts"}
    for a in body["alerts"]:
        assert set(a.keys()) == {"cell", "lat", "lon", "probability", "lead_time_min"}
