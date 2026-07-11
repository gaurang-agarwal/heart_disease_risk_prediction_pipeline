"""Tests for src/models/predict.py — bundle persistence and inference."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.predict import (
    load_bundle,
    predict_batch,
    predict_one,
    save_bundle,
)


@pytest.fixture
def bundle_pipeline(fitted_pipeline, tmp_path):
    """Save the fitted_pipeline fixture to a temp bundle and reload it."""
    bundle_path = tmp_path / "bundle.joblib"
    save_bundle(fitted_pipeline, bundle_path)
    return bundle_path


def test_save_load_bundle_roundtrip(fitted_pipeline, tmp_path, valid_df):
    """Reloaded bundle must produce identical predictions to the original pipeline."""
    bundle_path = tmp_path / "bundle.joblib"
    saved_path = save_bundle(fitted_pipeline, bundle_path)
    assert saved_path.exists(), "Bundle file was not written"

    reloaded = load_bundle(str(bundle_path))
    X = valid_df.drop(columns=["target"])
    original_preds = fitted_pipeline.predict(X)
    reloaded_preds = reloaded.predict(X)
    assert list(original_preds) == list(reloaded_preds), "Predictions differ after reload"


def test_predict_one_contract(bundle_pipeline, valid_df):
    """predict_one must return dict with 'prediction' (int) and 'probability' (float)."""
    import src.models.predict as pred_module

    pred_module._pipeline = None
    load_bundle(str(bundle_pipeline))

    row = valid_df.drop(columns=["target"]).iloc[0].to_dict()
    result = predict_one(row)

    assert "prediction" in result, "Missing 'prediction' key"
    assert "probability" in result, "Missing 'probability' key"
    assert isinstance(result["prediction"], int), "prediction must be int"
    assert isinstance(result["probability"], float), "probability must be float"
    assert result["prediction"] in {0, 1}, "prediction must be 0 or 1"


def test_predict_batch_shape(bundle_pipeline, valid_df):
    """predict_batch must return DataFrame with same row count and added columns."""
    import src.models.predict as pred_module

    pred_module._pipeline = None
    load_bundle(str(bundle_pipeline))

    X = valid_df.drop(columns=["target"])
    result = predict_batch(X)

    assert len(result) == len(X), "Row count must be preserved"
    assert "prediction" in result.columns, "Missing 'prediction' column"
    assert "probability" in result.columns, "Missing 'probability' column"


def test_predict_probability_range(bundle_pipeline, valid_df):
    """All predicted probabilities must be in [0, 1]."""
    import src.models.predict as pred_module

    pred_module._pipeline = None
    load_bundle(str(bundle_pipeline))

    X = valid_df.drop(columns=["target"])
    result = predict_batch(X)

    assert (result["probability"] >= 0).all(), "Some probabilities < 0"
    assert (result["probability"] <= 1).all(), "Some probabilities > 1"


def test_predict_one_missing_features(bundle_pipeline):
    """predict_one must raise ValueError when required features are absent."""
    import src.models.predict as pred_module

    pred_module._pipeline = None
    load_bundle(str(bundle_pipeline))

    with pytest.raises(ValueError, match="Missing required features"):
        predict_one({"age": 55})


def test_load_bundle_missing_file():
    """load_bundle must raise FileNotFoundError for a non-existent local path."""
    with pytest.raises(FileNotFoundError):
        load_bundle("/nonexistent/path/bundle.joblib")


def test_save_bundle_returns_resolved_path(fitted_pipeline, tmp_path):
    """save_bundle must return the resolved absolute Path of the written file."""
    bundle_path = tmp_path / "sub" / "bundle.joblib"
    result = save_bundle(fitted_pipeline, bundle_path)
    assert isinstance(result, Path), "save_bundle must return a Path"
    assert result.is_absolute(), "Returned path must be absolute"
    assert result.exists(), "Bundle file must exist after save"
