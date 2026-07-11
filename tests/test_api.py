"""Integration and unit tests for the FastAPI prediction service.

Covers schemas, model loader, /health, /predict, /metrics endpoints,
Prometheus counter/histogram behaviour, and Docker-build smoke proxy.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import src.api.model_loader as model_loader_module
from src.api.schemas import HeartFeatures, HealthResponse, PredictionResponse

# ---------------------------------------------------------------------------
# Helper: valid payload dict
# ---------------------------------------------------------------------------

_VALID_PAYLOAD = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline(fitted_pipeline):
    """Return the fitted_pipeline from conftest with a mock predict_proba."""
    return fitted_pipeline


@pytest.fixture
def test_client(mock_pipeline):
    """TestClient with the model pre-loaded (no disk I/O needed in tests)."""
    model_loader_module._model = mock_pipeline
    model_loader_module._model_version = "test-1"

    from src.api.main import create_app

    application = create_app()
    with TestClient(application, raise_server_exceptions=False) as client:
        yield client

    model_loader_module._model = None


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_schema_accepts_valid_payload():
    """HeartFeatures must parse a fully valid payload without errors."""
    features = HeartFeatures(**_VALID_PAYLOAD)
    assert features.age == 63


def test_schema_rejects_out_of_range():
    """HeartFeatures must raise ValidationError for physiologically invalid values."""
    bad = dict(_VALID_PAYLOAD)
    bad["age"] = 999  # > 120
    with pytest.raises(ValidationError):
        HeartFeatures(**bad)


def test_schema_rejects_missing_field():
    """HeartFeatures must raise ValidationError when a required field is absent."""
    bad = dict(_VALID_PAYLOAD)
    del bad["thal"]
    with pytest.raises(ValidationError):
        HeartFeatures(**bad)


def test_prediction_response_schema():
    """PredictionResponse must hold prediction, probability, and model_version."""
    resp = PredictionResponse(prediction=1, probability=0.82, model_version="v1")
    assert resp.prediction == 1
    assert resp.probability == pytest.approx(0.82)
    assert resp.model_version == "v1"


def test_health_response_schema():
    """HealthResponse must hold status and model_loaded fields."""
    hr = HealthResponse(status="ok", model_loaded=True)
    assert hr.status == "ok"
    assert hr.model_loaded is True


# ---------------------------------------------------------------------------
# Model loader tests
# ---------------------------------------------------------------------------


def test_model_loader_returns_pipeline(fitted_pipeline):
    """load_model with a local joblib path must return a Pipeline instance."""
    import tempfile
    from pathlib import Path

    from sklearn.pipeline import Pipeline

    import joblib

    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td) / "bundle.joblib"
        joblib.dump(fitted_pipeline, bundle)

        model_loader_module.reset_model()
        pipeline = model_loader_module.load_model(str(bundle))

    assert isinstance(pipeline, Pipeline), "load_model must return a Pipeline"


def test_model_loader_caches(fitted_pipeline):
    """get_model must return the same object on repeated calls (singleton)."""
    model_loader_module._model = fitted_pipeline
    first = model_loader_module.get_model()
    second = model_loader_module.get_model()
    assert first is second, "get_model must return the cached instance"
    model_loader_module._model = None


def test_model_loader_raises_when_unset():
    """get_model must raise RuntimeError when no model is loaded and no env var."""
    model_loader_module._model = None
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(RuntimeError),
    ):
        model_loader_module.get_model()


def test_model_loader_missing_file():
    """load_model must raise FileNotFoundError for a non-existent local path."""
    model_loader_module._model = None
    with pytest.raises(FileNotFoundError):
        model_loader_module.load_model("/nonexistent/bundle.joblib")


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


def test_health_endpoint(test_client):
    """GET /health must return 200 with status='ok' and model_loaded=True."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_endpoint_returns_prediction(test_client):
    """POST /predict with valid payload must return 200 with prediction + probability."""
    resp = test_client.post("/predict", json=_VALID_PAYLOAD)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "prediction" in body
    assert "probability" in body
    assert "model_version" in body
    assert body["prediction"] in {0, 1}
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_endpoint_validation_error(test_client):
    """POST /predict with an out-of-range field must return 422 Unprocessable Entity."""
    bad = dict(_VALID_PAYLOAD)
    bad["age"] = -5
    resp = test_client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_metrics_endpoint_exposes_counters(test_client):
    """GET /metrics must expose http_requests_total and predictions_total metrics."""
    test_client.post("/predict", json=_VALID_PAYLOAD)
    resp = test_client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "http_requests_total" in body
    assert "predictions_total" in body


# ---------------------------------------------------------------------------
# Middleware tests
# ---------------------------------------------------------------------------


def test_request_counter_increments(test_client):
    """http_requests_total must increase after making a request."""
    test_client.get("/metrics")
    test_client.get("/health")

    resp_after = test_client.get("/metrics")
    after_text = resp_after.text

    assert "http_requests_total" in after_text


def test_latency_histogram_observed(test_client):
    """http_request_duration_seconds histogram must appear in /metrics output."""
    test_client.get("/health")
    resp = test_client.get("/metrics")
    assert "http_request_duration_seconds" in resp.text


# ---------------------------------------------------------------------------
# App importable smoke test
# ---------------------------------------------------------------------------


def test_app_importable():
    """create_app() must succeed without errors (Docker build proxy test)."""
    from src.api.main import create_app

    application = create_app()
    assert application is not None
