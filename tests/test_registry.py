"""Tests for src/models/registry.py.

Covers:
  - test_register_model_creates_version
  - test_promote_sets_stage
  - test_get_production_uri_format
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src import config
from src.features.pipeline import build_preprocessor
from src.models.registry import get_production_model_uri, promote_to_stage, register_model
from src.models.train import build_models, train_model


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _log_model_and_get_run_id(valid_df, tmp_mlflow_uri: str) -> str:
    """Train a tiny pipeline, log it to MLflow, and return the run_id."""
    # tmp_mlflow_uri is already set by the fixture; re-affirm for clarity
    mlflow.set_experiment("registry-test")

    X = valid_df.drop(columns=["target"])
    y = valid_df["target"]
    preprocessor = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    pipeline = train_model(
        build_models(42)["logistic_regression"],
        preprocessor,
        X,
        y,
    )
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(pipeline, "model")
        return run.info.run_id


# --------------------------------------------------------------------------- #
# test_register_model_creates_version
# --------------------------------------------------------------------------- #


def test_register_model_creates_version(tmp_mlflow_uri, valid_df):
    """register_model() must create a ModelVersion with version >= 1."""
    run_id = _log_model_and_get_run_id(valid_df, tmp_mlflow_uri)

    mv = register_model(run_id, model_name="test-clf")

    assert mv is not None
    assert int(mv.version) >= 1


def test_register_model_returns_model_version_type(tmp_mlflow_uri, valid_df):
    """register_model() must return an mlflow ModelVersion object."""
    from mlflow.entities.model_registry import ModelVersion

    run_id = _log_model_and_get_run_id(valid_df, tmp_mlflow_uri)
    mv = register_model(run_id, model_name="test-clf-type")

    assert isinstance(mv, ModelVersion)


def test_register_model_custom_name(tmp_mlflow_uri, valid_df):
    """register_model() must use the supplied model_name."""
    run_id = _log_model_and_get_run_id(valid_df, tmp_mlflow_uri)
    mv = register_model(run_id, model_name="custom-model-name")

    assert mv.name == "custom-model-name"


# --------------------------------------------------------------------------- #
# test_promote_sets_stage
# --------------------------------------------------------------------------- #


def test_promote_sets_stage(tmp_mlflow_uri, valid_df):
    """promote_to_stage() must set the model version's stage to 'Staging'."""
    run_id = _log_model_and_get_run_id(valid_df, tmp_mlflow_uri)
    mv = register_model(run_id, model_name="test-clf-stage")

    promote_to_stage("test-clf-stage", int(mv.version), stage="Staging")

    client = MlflowClient(tracking_uri=tmp_mlflow_uri)
    mv_updated = client.get_model_version("test-clf-stage", mv.version)
    assert mv_updated.current_stage == "Staging"


def test_promote_to_production(tmp_mlflow_uri, valid_df):
    """promote_to_stage() must also handle 'Production' as target stage."""
    run_id = _log_model_and_get_run_id(valid_df, tmp_mlflow_uri)
    mv = register_model(run_id, model_name="test-clf-prod")

    promote_to_stage("test-clf-prod", int(mv.version), stage="Production")

    client = MlflowClient(tracking_uri=tmp_mlflow_uri)
    mv_updated = client.get_model_version("test-clf-prod", mv.version)
    assert mv_updated.current_stage == "Production"


# --------------------------------------------------------------------------- #
# test_get_production_uri_format
# --------------------------------------------------------------------------- #


def test_get_production_uri_format():
    """get_production_model_uri() must return a valid models:/ URI string."""
    uri = get_production_model_uri("heart-disease-clf")

    assert uri.startswith("models:/"), f"URI must start with 'models:/': {uri}"
    assert "Production" in uri, f"URI must contain 'Production': {uri}"
    assert "heart-disease-clf" in uri


def test_get_production_uri_custom_name():
    """get_production_model_uri() must embed the given model name in the URI."""
    uri = get_production_model_uri("my-custom-model")

    assert "my-custom-model" in uri
    assert uri == "models:/my-custom-model/Production"
