"""Tests for the serving + A/B endpoints (mock the models, test the wiring)."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def ab_client(monkeypatch):
    from src.inference import ab_compare_api

    monkeypatch.setattr(ab_compare_api, "_load", lambda cfg, adapter: object())
    monkeypatch.setattr(
        ab_compare_api,
        "_generate",
        lambda key, req: ("warm reply" if key == "finetuned" else "cold reply", 100.0),
    )
    with TestClient(ab_compare_api.app) as c:
        yield c


@pytest.mark.xfail(reason="stub: implement /compare")
def test_compare_returns_both_sides_and_deltas(ab_client):
    r = ab_client.post("/compare", json={"customer_message": "I'm furious", "draft_reply": "No refund."})
    assert r.status_code == 200
    body = r.json()
    assert {"base", "finetuned", "deltas", "latency_ms"} <= body.keys()


@pytest.mark.xfail(reason="stub: implement serve.py engine")
def test_healthz_reports_engine_state():
    from src.inference.serve import app

    with TestClient(app) as c:
        assert c.get("/healthz").json()["ok"] is True
