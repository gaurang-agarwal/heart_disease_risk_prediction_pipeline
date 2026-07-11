"""Standalone inference module: bundle persistence and prediction utilities.

Provides serialisation helpers for a complete scikit-learn ``Pipeline`` (model +
preprocessor) and convenience wrappers for single-record and batch prediction.

A minimal CLI is exposed via ``python -m src.models.predict`` so the bundle can
be exercised outside of any web framework::

    # single JSON record
    python -m src.models.predict --input sample.json

    # batch CSV
    python -m src.models.predict --input records.csv

Environment variables recognised at runtime
-------------------------------------------
MODEL_BUNDLE_PATH
    Path to the ``.joblib`` bundle produced by :func:`save_bundle`.
    Defaults to ``models/bundle.joblib`` relative to the project root.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config

logger = logging.getLogger(__name__)

# Default on-disk location for the persisted bundle.
_DEFAULT_BUNDLE_PATH: Path = config.PROJECT_ROOT / "models" / "bundle.joblib"

# The module-level singleton; populated by :func:`load_bundle`.
_pipeline: Pipeline | None = None
_bundle_path_used: str | None = None


# --------------------------------------------------------------------------- #
# Bundle persistence
# --------------------------------------------------------------------------- #


def save_bundle(pipeline: Pipeline, path: Path) -> Path:
    """Serialize the full pipeline (preprocessor + model) to disk via joblib.

    Parameters
    ----------
    pipeline:
        A *fitted* scikit-learn :class:`~sklearn.pipeline.Pipeline` whose
        first step is the ``ColumnTransformer`` preprocessor and whose last
        step is a fitted estimator.
    path:
        Destination ``.joblib`` file path.  Parent directories are created
        automatically.

    Returns
    -------
    Path
        Resolved absolute path where the bundle was written.

    Raises
    ------
    IOError
        If the file cannot be written (e.g. permission denied).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Bundle saved to %s", path)
    return path.resolve()


def load_bundle(path_or_uri: str) -> Pipeline:
    """Load a serialised pipeline bundle from a local path or an MLflow URI.

    Supports two URI formats:

    * **Local path** — anything not starting with ``"runs:/"`` or ``"models:/"``
      is treated as a filesystem path and loaded with ``joblib.load``.
    * **MLflow URI** — ``"runs:/<run_id>/model"`` or ``"models:/<name>/<stage>"``
      are passed to :func:`mlflow.sklearn.load_model`.

    Parameters
    ----------
    path_or_uri:
        Either a filesystem path (absolute or relative) or an MLflow model URI.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The reloaded, predict-ready pipeline.

    Raises
    ------
    FileNotFoundError
        If *path_or_uri* is a local path that does not exist.
    mlflow.exceptions.MlflowException
        If the MLflow URI cannot be resolved.
    """
    global _pipeline, _bundle_path_used

    if path_or_uri.startswith("runs:/") or path_or_uri.startswith("models:/"):
        import mlflow.sklearn

        pipeline = mlflow.sklearn.load_model(path_or_uri)
    else:
        p = Path(path_or_uri)
        if not p.exists():
            raise FileNotFoundError(f"Bundle not found: {p}")
        pipeline = joblib.load(p)

    _pipeline = pipeline
    _bundle_path_used = path_or_uri
    logger.info("Bundle loaded from %s", path_or_uri)
    return pipeline


def _get_pipeline() -> Pipeline:
    """Return the module-level cached pipeline, loading the default if needed."""
    global _pipeline
    if _pipeline is None:
        bundle_path = os.getenv("MODEL_BUNDLE_PATH", str(_DEFAULT_BUNDLE_PATH))
        _pipeline = load_bundle(bundle_path)
    return _pipeline


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #


def predict_one(features: dict) -> dict:
    """Return a prediction for a single patient feature dictionary.

    Parameters
    ----------
    features:
        Dictionary mapping feature names to their values.  Must contain all
        columns defined in :attr:`src.config.FEATURE_COLS`.

    Returns
    -------
    dict
        ``{"prediction": int, "probability": float}`` where *prediction* is
        ``0`` (no disease) or ``1`` (disease) and *probability* is the
        model's estimated probability for the positive class.

    Raises
    ------
    ValueError
        If *features* is missing one or more required columns.
    """
    missing = set(config.FEATURE_COLS) - set(features.keys())
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    df = pd.DataFrame([features])[config.FEATURE_COLS]
    pipeline = _get_pipeline()
    prediction = int(pipeline.predict(df)[0])
    probability = float(pipeline.predict_proba(df)[0, 1])
    return {"prediction": prediction, "probability": probability}


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``prediction`` and ``probability`` columns to a feature DataFrame.

    Parameters
    ----------
    df:
        Feature DataFrame. Must contain at least all columns in
        :attr:`src.config.FEATURE_COLS`; extra columns are ignored.

    Returns
    -------
    pandas.DataFrame
        A copy of *df* with two additional columns appended:
        ``prediction`` (int) and ``probability`` (float, positive class).

    Raises
    ------
    ValueError
        If *df* is missing one or more required feature columns.
    """
    missing = set(config.FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    pipeline = _get_pipeline()
    X = df[config.FEATURE_COLS]
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]

    result = df.copy()
    result["prediction"] = predictions.astype(int)
    result["probability"] = probabilities
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    """CLI entry point: read JSON or CSV from ``--input``, print predictions.

    Usage examples::

        python -m src.models.predict --input sample.json
        python -m src.models.predict --input records.csv
        python -m src.models.predict --input records.csv --bundle models/bundle.joblib
    """
    parser = argparse.ArgumentParser(
        description="Heart Disease risk prediction inference CLI",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSON file (single record or list) or CSV file.",
    )
    parser.add_argument(
        "--bundle",
        default=os.getenv("MODEL_BUNDLE_PATH", str(_DEFAULT_BUNDLE_PATH)),
        help="Path to the serialised model bundle (.joblib).",
    )
    args = parser.parse_args()

    load_bundle(args.bundle)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".json":
        with input_path.open() as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            result = predict_one(payload)
            print(json.dumps(result, indent=2))
        else:
            for record in payload:
                print(json.dumps(predict_one(record), indent=2))
    else:
        df = pd.read_csv(input_path)
        out = predict_batch(df)
        print(out[["prediction", "probability"]].to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
