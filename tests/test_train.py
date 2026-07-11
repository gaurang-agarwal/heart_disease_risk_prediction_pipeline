"""Tests for src/models/train.py.

Covers:
  - test_build_models_keys
  - test_train_model_returns_fitted_pipeline
  - test_pipeline_predict_shape
  - test_cross_validate_returns_metrics
  - test_cv_scores_within_bounds
  - test_select_best_picks_highest_auc
  - test_select_best_tie_break_f1
  - test_log_experiment_creates_run
  - test_logged_metrics_present
  - test_model_artifact_logged
  - test_plots_logged_as_artifacts
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from src import config
from src.features.pipeline import build_preprocessor
from src.models.train import (
    build_models,
    cross_validate_model,
    log_experiment,
    select_best,
    train_model,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_NUMERIC = config.NUMERIC_COLS
_CATEGORICAL = config.CATEGORICAL_COLS


def _make_xy(valid_df: pd.DataFrame):
    """Split valid_df into X and y."""
    X = valid_df.drop(columns=["target"])
    y = valid_df["target"]
    return X, y


# --------------------------------------------------------------------------- #
# Training entrypoint & model factory
# --------------------------------------------------------------------------- #


def test_build_models_keys():
    """build_models() must return exactly the two expected keys."""
    models = build_models(42)
    assert set(models.keys()) == {"logistic_regression", "random_forest"}


def test_train_model_returns_fitted_pipeline(valid_df):
    """train_model() must return a fitted sklearn Pipeline."""
    X, y = _make_xy(valid_df)
    preprocessor = build_preprocessor(_NUMERIC, _CATEGORICAL)
    models = build_models(42)
    pipeline = train_model(models["logistic_regression"], preprocessor, X, y)

    assert isinstance(pipeline, Pipeline)
    # A fitted pipeline can call predict without raising NotFittedError
    preds = pipeline.predict(X)
    assert len(preds) == len(y)


def test_pipeline_predict_shape(valid_df):
    """Pipeline predictions must have the same length as input rows."""
    X, y = _make_xy(valid_df)
    preprocessor = build_preprocessor(_NUMERIC, _CATEGORICAL)
    pipeline = train_model(build_models(42)["random_forest"], preprocessor, X, y)

    preds = pipeline.predict(X)
    assert preds.shape == (len(X),)


def test_pipeline_has_named_steps(valid_df):
    """The trained Pipeline must expose 'preprocessor' and 'classifier' steps."""
    X, y = _make_xy(valid_df)
    preprocessor = build_preprocessor(_NUMERIC, _CATEGORICAL)
    pipeline = train_model(build_models(42)["logistic_regression"], preprocessor, X, y)

    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #


def test_cross_validate_returns_metrics(valid_df):
    """cross_validate_model() must return all five expected metric keys."""
    X, y = _make_xy(valid_df)
    preprocessor = build_preprocessor(_NUMERIC, _CATEGORICAL)
    pipeline = train_model(build_models(42)["logistic_regression"], preprocessor, X, y)

    cv_scores = cross_validate_model(pipeline, X, y, cv=2)

    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert set(cv_scores.keys()) == expected_keys


def test_cv_scores_within_bounds(valid_df):
    """All cross-validation scores must be floats in [0, 1]."""
    X, y = _make_xy(valid_df)
    preprocessor = build_preprocessor(_NUMERIC, _CATEGORICAL)
    pipeline = train_model(build_models(42)["logistic_regression"], preprocessor, X, y)

    cv_scores = cross_validate_model(pipeline, X, y, cv=2)

    for metric, score in cv_scores.items():
        assert isinstance(score, float), f"{metric} is not float"
        assert 0.0 <= score <= 1.0, f"{metric}={score} out of [0,1]"


def test_cross_validate_custom_scoring(valid_df):
    """cross_validate_model() must respect a custom scoring list."""
    X, y = _make_xy(valid_df)
    preprocessor = build_preprocessor(_NUMERIC, _CATEGORICAL)
    pipeline = train_model(build_models(42)["logistic_regression"], preprocessor, X, y)

    scores = cross_validate_model(pipeline, X, y, cv=2, scoring=["accuracy"])
    assert set(scores.keys()) == {"accuracy"}


# --------------------------------------------------------------------------- #
# Model comparison & best-model selection
# --------------------------------------------------------------------------- #


def _fake_results(auc_lr=0.80, f1_lr=0.75, auc_rf=0.85, f1_rf=0.80):
    """Build a minimal results dict for select_best tests."""
    mock_pipe = MagicMock(spec=Pipeline)
    return {
        "logistic_regression": {
            "pipeline": mock_pipe,
            "metrics": {"roc_auc": auc_lr, "f1": f1_lr},
        },
        "random_forest": {
            "pipeline": mock_pipe,
            "metrics": {"roc_auc": auc_rf, "f1": f1_rf},
        },
    }


def test_select_best_picks_highest_auc():
    """select_best() must select the model with the highest roc_auc."""
    results = _fake_results(auc_lr=0.80, auc_rf=0.90)
    best_name, _, _ = select_best(results)
    assert best_name == "random_forest"


def test_select_best_tie_break_f1():
    """select_best() must use f1 as a tie-breaker when roc_auc is equal."""
    results = _fake_results(auc_lr=0.85, f1_lr=0.70, auc_rf=0.85, f1_rf=0.80)
    best_name, _, _ = select_best(results)
    assert best_name == "random_forest"


def test_select_best_returns_pipeline_and_metrics():
    """select_best() must return (name, Pipeline, metrics_dict)."""
    results = _fake_results()
    best_name, best_pipeline, best_metrics = select_best(results)
    assert isinstance(best_name, str)
    assert isinstance(best_metrics, dict)
    assert "roc_auc" in best_metrics


def test_select_best_raises_on_empty():
    """select_best() must raise ValueError for an empty results dict."""
    with pytest.raises(ValueError, match="empty"):
        select_best({})


# --------------------------------------------------------------------------- #
# MLflow tracking integration
# --------------------------------------------------------------------------- #


def test_log_experiment_creates_run(tmp_mlflow_uri):
    """log_experiment() must return a non-empty run_id when inside start_run."""
    mlflow.set_experiment("test-log-exp")
    with mlflow.start_run():
        run_id = log_experiment("lr_test", {"param_a": "1"}, {"accuracy": 0.9}, [])

    assert run_id is not None
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_logged_metrics_present(tmp_mlflow_uri):
    """Metrics logged via log_experiment() must be retrievable from the tracking store."""
    mlflow.set_experiment("test-metrics")
    metrics_to_log = {"accuracy": 0.88, "roc_auc": 0.92}

    with mlflow.start_run() as run:
        log_experiment("lr_test", {}, metrics_to_log, [])
        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=tmp_mlflow_uri)
    run_data = client.get_run(run_id).data

    for key, expected_val in metrics_to_log.items():
        assert key in run_data.metrics, f"Metric '{key}' not found in logged run."
        assert run_data.metrics[key] == pytest.approx(expected_val, rel=1e-6)


def test_log_experiment_logs_params(tmp_mlflow_uri):
    """Parameters passed to log_experiment() must be stored in the run."""
    mlflow.set_experiment("test-params")
    params = {"model_type": "lr", "test_size": "0.2"}

    with mlflow.start_run() as run:
        log_experiment("lr", params, {}, [])
        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=tmp_mlflow_uri)
    stored = client.get_run(run_id).data.params
    assert stored["model_name"] == "lr"
    for k, v in params.items():
        assert stored[k] == v


# --------------------------------------------------------------------------- #
# Artifact & plot logging
# --------------------------------------------------------------------------- #


def test_model_artifact_logged(tmp_mlflow_uri, valid_df):
    """The sklearn model must be logged as a named artifact under each run."""
    mlflow.set_experiment("test-model-artifact")

    X, y = _make_xy(valid_df)
    preprocessor = build_preprocessor(_NUMERIC, _CATEGORICAL)
    pipeline = train_model(build_models(42)["logistic_regression"], preprocessor, X, y)

    with mlflow.start_run() as run:
        log_experiment("lr", {}, {}, [])
        mlflow.sklearn.log_model(pipeline, "model")
        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=tmp_mlflow_uri)

    # MLflow 3.x logs models as LoggedModels (not in list_artifacts).
    # Check via search_logged_models if available, else fall back to list_artifacts.
    if hasattr(client, "search_logged_models"):
        logged = client.search_logged_models(
            experiment_ids=[mlflow.get_experiment_by_name("test-model-artifact").experiment_id]
        )
        model_run_ids = [m.source_run_id for m in logged]
        assert (
            run_id in model_run_ids
        ), f"No logged model found for run {run_id}. Logged model run_ids: {model_run_ids}"
    else:
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]
        assert (
            "model" in artifact_names
        ), f"'model' artifact not found in run artifacts: {artifact_names}"


def test_plots_logged_as_artifacts(tmp_path, tmp_mlflow_uri):
    """At least three plot artifacts must be logged in a single run."""
    mlflow.set_experiment("test-plots")

    # Create dummy PNG files to simulate evaluation plots
    plot_files: list[Path] = []
    for name in ["confusion_matrix.png", "roc_curve.png", "pr_curve.png"]:
        p = tmp_path / name
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # minimal PNG header
        plot_files.append(p)

    with mlflow.start_run():
        run_id = log_experiment("rf", {}, {"accuracy": 0.85}, plot_files)

    client = MlflowClient(tracking_uri=tmp_mlflow_uri)
    artifacts = client.list_artifacts(run_id)
    assert (
        len(artifacts) >= 3
    ), f"Expected ≥3 artifacts, got {len(artifacts)}: {[a.path for a in artifacts]}"


def test_missing_artifact_files_skipped(tmp_path, tmp_mlflow_uri):
    """Artifact paths that don't exist must be silently skipped."""
    mlflow.set_experiment("test-missing-artifacts")
    nonexistent = tmp_path / "does_not_exist.png"

    with mlflow.start_run():
        run_id = log_experiment("lr", {}, {}, [nonexistent])

    # No exception should be raised; run should still be created
    assert run_id is not None
