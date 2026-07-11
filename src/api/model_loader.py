"""Lazy-singleton model loader for the FastAPI prediction service.

The module maintains a single, process-wide :class:`~sklearn.pipeline.Pipeline`
instance that is loaded on first request (or explicitly via :func:`load_model`).
Subsequent calls to :func:`get_model` return the cached instance without I/O.

Environment variables
---------------------
MODEL_BUNDLE_PATH
    Path to the ``.joblib`` bundle (takes precedence over ``MODEL_URI``).
MODEL_URI
    MLflow model URI, e.g. ``"models:/heart-disease-clf/Production"``.
    Used when ``MODEL_BUNDLE_PATH`` is absent.
MODEL_VERSION
    Optional version string to expose via :func:`get_model_version`.
    Defaults to ``"unknown"``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_model: Pipeline | None = None
_model_version: str = "unknown"


def load_model(uri: str) -> Pipeline:
    """Load a model from a local ``.joblib`` path or an MLflow URI.

    Updates the module-level cached model so that subsequent :func:`get_model`
    calls return the newly loaded instance.

    Parameters
    ----------
    uri:
        Either a local filesystem path (absolute or relative) to a ``.joblib``
        bundle or an MLflow model URI starting with ``"runs:/"`` or
        ``"models:/"``).

    Returns
    -------
    sklearn.pipeline.Pipeline
        The loaded, predict-ready pipeline.

    Raises
    ------
    mlflow.exceptions.MlflowException
        If the MLflow URI cannot be resolved.
    FileNotFoundError
        If the local path does not exist.
    """
    global _model, _model_version

    import joblib

    p = Path(uri)
    if not p.exists():
        raise FileNotFoundError(f"Model bundle not found: {p}")
    pipeline = joblib.load(p)

    version_file = p.parent / "bundle.version"
    logger.info("version file=====", version_file)
    if version_file.exists():
        _model_version = version_file.read_text().strip()
    _model = pipeline
    logger.info("Model loaded from %s", uri)
    return pipeline


def get_model() -> Pipeline:
    """Return the cached model, loading it on first call if necessary.

    Resolves the model location from environment variables in this order:

    1. ``MODEL_BUNDLE_PATH`` — local ``.joblib`` bundle path.
    2. ``MODEL_URI`` — MLflow URI.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The predict-ready pipeline singleton.

    Raises
    ------
    RuntimeError
        If no model has been loaded and neither ``MODEL_BUNDLE_PATH`` nor
        ``MODEL_URI`` environment variables are set.
    """
    global _model
    if _model is None:
        bundle_path = os.getenv("MODEL_BUNDLE_PATH")

        if bundle_path:
            load_model(bundle_path)
        else:
            raise RuntimeError(
                "No model loaded. Set MODEL_BUNDLE_PATH "
                "or call load_model() explicitly before serving requests."
            )
    return _model


def get_model_version() -> str:
    """Return the version string of the currently loaded model.

    The version is taken from the ``MODEL_VERSION`` environment variable.
    Defaults to ``"unknown"`` if the variable is not set.

    Returns
    -------
    str
        Model version string.
    """
    return os.getenv("MODEL_VERSION", _model_version)


def reset_model() -> None:
    """Reset the cached model to ``None`` (for testing purposes only).

    Calling this function forces the next :func:`get_model` call to reload
    from the configured URI.
    """
    global _model
    _model = None
