"""FastAPI application: /predict, /health, /metrics endpoints.

Build and run::

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Environment variables
---------------------
MODEL_BUNDLE_PATH
    Path to the serialised ``.joblib`` model bundle.
MODEL_URI
    MLflow model URI (used if MODEL_BUNDLE_PATH is unset).
MODEL_VERSION
    Version string exposed in every PredictionResponse.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.middleware import PREDICTIONS_TOTAL, setup_metrics
from src.api.model_loader import get_model, get_model_version, load_model
from src.api.schemas import HeartFeatures, HealthResponse, PredictionResponse
from src import config

import os
import pandas as pd

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the FastAPI application with routes, middleware, and startup logic.

    Attaches Prometheus + request-logging middleware, registers ``/health``,
    ``/predict``, and ``/metrics`` routes, and wires a startup event that loads
    the model from the configured URI.

    Returns
    -------
    fastapi.FastAPI
        The fully configured ASGI application.
    """
    application = FastAPI(
        title="Heart Disease Risk Prediction API",
        description=(
            "Binary classification endpoint for UCI Heart Disease dataset. "
            "POST /predict returns prediction, probability, and model version."
        ),
        version="1.0.0",
    )

    setup_metrics(application)

    @application.on_event("startup")
    async def _startup() -> None:
        """Load the model on application startup (skip if already loaded)."""
        from src.api.model_loader import _model as current_model

        if current_model is not None:
            logger.info("Model already loaded; skipping startup load.")
            return

        bundle_path = os.getenv("MODEL_BUNDLE_PATH")
        model_uri = os.getenv("MODEL_URI")
        if bundle_path:
            load_model(bundle_path)
        elif model_uri:
            load_model(model_uri)
        else:
            logger.warning(
                "MODEL_BUNDLE_PATH and MODEL_URI not set; "
                "model will be loaded lazily on first request."
            )

    @application.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        """Return the service liveness and model-load status.

        Returns
        -------
        HealthResponse
            ``{"status": "ok", "model_loaded": bool}``
        """
        from src.api.model_loader import _model

        return HealthResponse(status="ok", model_loaded=_model is not None)

    @application.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    async def predict(payload: HeartFeatures) -> PredictionResponse:
        """Predict heart disease risk for one patient.

        Parameters
        ----------
        payload:
            Validated :class:`~src.api.schemas.HeartFeatures` object.

        Returns
        -------
        PredictionResponse
            Prediction (0/1), probability, and model version.

        Raises
        ------
        HTTPException
            500 if the model is unavailable or prediction fails.
        """
        try:
            pipeline = get_model()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        try:
            features = payload.model_dump()
            X = pd.DataFrame([features])[config.FEATURE_COLS]
            prediction = int(pipeline.predict(X)[0])
            probability = float(pipeline.predict_proba(X)[0, 1])
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

        PREDICTIONS_TOTAL.labels(class_label=str(prediction)).inc()

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_version=get_model_version(),
        )

    @application.get("/metrics", tags=["ops"])
    async def metrics() -> Response:
        """Expose Prometheus metrics in the text exposition format.

        Returns
        -------
        starlette.responses.Response
            Plain-text Prometheus metrics page.
        """
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return application


app = create_app()
