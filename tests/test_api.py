"""Smoke + contract tests for the FastAPI service.

Uses ``httpx.ASGITransport`` so we don't open a real port — the app is
exercised in-process. Fast and CI-friendly.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Manually trigger startup so the registry is loaded
        async with app.router.lifespan_context(app):
            yield ac


VALID_PAYLOAD = {
    "revenue": 65000,
    "dti_n": 18.5,
    "loan_amnt": 15000,
    "fico_n": 720,
    "experience_c": 1,
    "emp_length": 5,
    "purpose": "debt_consolidation",
    "home_ownership_n": "MORTGAGE",
    "addr_state": "CA",
    "zip_code": "900xx",
    "issue_d": "2017-06-01",
}


@pytest.mark.asyncio
async def test_health_responds_ok(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "credit-risk-api"


@pytest.mark.asyncio
async def test_models_info_exposes_metrics(client):
    r = await client.get("/v1/models/info")
    assert r.status_code == 200
    body = r.json()
    assert body["feature_count"] == 15
    assert "auroc_test_calibrated" in body["metrics"]


@pytest.mark.asyncio
async def test_predict_single_returns_calibrated(client):
    r = await client.post("/v1/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["pd_calibrated"] <= 1.0
    assert 0.0 <= body["pd_raw"] <= 1.0
    assert 0 <= body["score_0_1000"] <= 1000
    assert body["risk_band"] in {"low", "medium", "high", "very_high"}
    # Macro must be merged server-side
    for key in ("fed_funds_rate", "us_unemployment", "vix_close", "us_10y_treasury", "us_real_gdp_yoy"):
        assert key in body["macro_snapshot"]


@pytest.mark.asyncio
async def test_predict_batch_latency_reported(client):
    r = await client.post("/v1/predict/batch", json={"loans": [VALID_PAYLOAD, VALID_PAYLOAD]})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2
    assert body["latency_ms"] > 0
    assert len(body["predictions"]) == 2


@pytest.mark.asyncio
async def test_predict_rejects_invalid_fico(client):
    bad = {**VALID_PAYLOAD, "fico_n": 9999}
    r = await client.post("/v1/predict", json=bad)
    assert r.status_code == 422
    details = r.json()["detail"]
    assert any(d["loc"][-1] == "fico_n" for d in details)


@pytest.mark.asyncio
async def test_predict_rejects_unknown_purpose(client):
    bad = {**VALID_PAYLOAD, "purpose": "boat"}
    r = await client.post("/v1/predict", json=bad)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_explain_returns_15_contributions(client):
    r = await client.post("/v1/explain", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert len(body["contributions"]) == 15
    assert len(body["top_drivers"]) == 5
    # All directions valid
    assert all(c["direction"] in {"risk_up", "risk_down"} for c in body["contributions"])


@pytest.mark.asyncio
async def test_monitor_drift_offline(client):
    r = await client.get("/v1/monitor/drift")
    assert r.status_code == 200
    body = r.json()
    assert body["total_events"] > 0
    assert "ADWIN" in body["by_detector"]
    assert "KSWIN" in body["by_detector"]


@pytest.mark.asyncio
async def test_monitor_drift_live_state_visible(client):
    # Issue a couple of predictions to feed the live monitor
    for _ in range(3):
        await client.post("/v1/predict", json=VALID_PAYLOAD)
    r = await client.get("/v1/monitor/drift/live")
    assert r.status_code == 200
    body = r.json()
    assert body["samples_seen"] >= 3


@pytest.mark.asyncio
async def test_monitor_calibration_summary(client):
    r = await client.get("/v1/monitor/calibration")
    assert r.status_code == 200
    summary = r.json()["summary"]
    for key in ("auroc_rolling_mean", "auroc_sliding_mean", "brier_static_mean", "brier_sliding_mean"):
        assert key in summary


@pytest.mark.asyncio
async def test_recalibrate_accepts(client):
    r = await client.post("/v1/monitor/recalibrate", json={"trigger": "manual"})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "accepted"
    assert "job_id" in body


@pytest.mark.asyncio
async def test_recalibrate_rejects_unknown_trigger(client):
    r = await client.post("/v1/monitor/recalibrate", json={"trigger": "bogus"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_prometheus(client):
    r = await client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "credit_risk_requests_total" in body
    assert "credit_risk_predictions_total" in body


@pytest.mark.asyncio
async def test_champion_vs_challenger_returns_yearly(client):
    r = await client.get("/v1/monitor/champion-vs-challenger")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["yearly"]) >= 5
    row = body["yearly"][0]
    for key in ("year", "n_test", "auroc", "ks", "brier", "calib_slope"):
        assert key in row
    for key in ("auroc_mean", "ks_mean", "brier_mean", "years_covered"):
        assert key in body["summary"]
    assert "ARF" in body["note"]


@pytest.mark.asyncio
async def test_rolling_vs_frozen_overlap_summary(client):
    r = await client.get("/v1/monitor/rolling-vs-frozen")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rolling"], "rolling list must not be empty"
    assert body["frozen"], "frozen list must not be empty"
    for key in ("auroc_uplift_pp", "ks_uplift_pp", "brier_delta_pp", "years_overlap"):
        assert key in body["summary"]
    # Rolling beats frozen on the overlap: positive AUROC uplift expected
    assert body["summary"]["auroc_uplift_pp"] > 0
    assert body["summary"]["years_overlap"] >= 1


@pytest.mark.asyncio
async def test_adaptive_shap_surfaces_present(client):
    r = await client.get("/v1/explain/adaptive-shap")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["heatmap"], "heatmap must not be empty"
    assert body["by_decile"], "by_decile must not be empty"
    assert body["ridge_surrogate"], "ridge_surrogate must not be empty"
    assert 5 <= len(body["top_features"]) <= 12
    # Heatmap cells only reference top features
    feats = set(body["top_features"])
    assert {cell["feature"] for cell in body["heatmap"]}.issubset(feats)
    # Months are sorted
    months = body["months"]
    assert months == sorted(months)
    # Deciles are 0..9-ish
    assert all(0 <= d <= 9 for d in body["deciles"])
    # Surrogate rows carry coefficient dicts
    assert isinstance(body["ridge_surrogate"][0]["coefs"], dict)
    assert "fico_n" in body["ridge_surrogate"][0]["coefs"] or len(body["ridge_surrogate"][0]["coefs"]) > 0
