"""Tests for src/models/evaluate.py.

Covers:
  - test_evaluate_returns_expected_keys
  - test_metrics_match_sklearn
  - test_plot_confusion_matrix_saves_file
"""

from __future__ import annotations

import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.models.evaluate import evaluate, plot_confusion_matrix, plot_pr_curve, plot_roc_curve


# --------------------------------------------------------------------------- #
# test_evaluate_returns_expected_keys
# --------------------------------------------------------------------------- #


def test_evaluate_returns_expected_keys(fitted_pipeline, valid_df):
    """evaluate() must return all five expected metric keys."""
    X = valid_df.drop(columns=["target"])
    y = valid_df["target"]

    metrics = evaluate(fitted_pipeline, X, y)

    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert (
        set(metrics.keys()) == expected_keys
    ), f"Missing keys: {expected_keys - set(metrics.keys())}"


# --------------------------------------------------------------------------- #
# test_metrics_match_sklearn
# --------------------------------------------------------------------------- #


def test_metrics_match_sklearn(fitted_pipeline, valid_df):
    """evaluate() values must match direct sklearn computations."""
    X = valid_df.drop(columns=["target"])
    y = valid_df["target"]

    metrics = evaluate(fitted_pipeline, X, y)

    y_pred = fitted_pipeline.predict(X)
    y_score = fitted_pipeline.predict_proba(X)[:, 1]

    assert metrics["accuracy"] == pytest.approx(accuracy_score(y, y_pred), rel=1e-6)
    assert metrics["f1"] == pytest.approx(f1_score(y, y_pred, zero_division=0), rel=1e-6)
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y, y_score), rel=1e-6)


# --------------------------------------------------------------------------- #
# test_plot_confusion_matrix_saves_file
# --------------------------------------------------------------------------- #


def test_plot_confusion_matrix_saves_file(tmp_path, valid_df, fitted_pipeline):
    """plot_confusion_matrix() must save a non-empty PNG to the given path."""
    X = valid_df.drop(columns=["target"])
    y = valid_df["target"]
    y_pred = fitted_pipeline.predict(X)

    out_path = tmp_path / "cm.png"
    returned_path = plot_confusion_matrix(y, y_pred, out_path)

    assert out_path.exists(), "PNG file was not created."
    assert out_path.stat().st_size > 0, "PNG file is empty."
    assert returned_path == out_path.resolve()


# --------------------------------------------------------------------------- #
# Additional plot tests
# --------------------------------------------------------------------------- #


def test_plot_roc_curve_saves_file(tmp_path, valid_df, fitted_pipeline):
    """plot_roc_curve() must save a non-empty PNG."""
    X = valid_df.drop(columns=["target"])
    y = valid_df["target"]
    y_score = fitted_pipeline.predict_proba(X)[:, 1]

    out_path = tmp_path / "roc.png"
    returned_path = plot_roc_curve(y, y_score, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert returned_path == out_path.resolve()


def test_plot_pr_curve_saves_file(tmp_path, valid_df, fitted_pipeline):
    """plot_pr_curve() must save a non-empty PNG."""
    X = valid_df.drop(columns=["target"])
    y = valid_df["target"]
    y_score = fitted_pipeline.predict_proba(X)[:, 1]

    out_path = tmp_path / "pr.png"
    returned_path = plot_pr_curve(y, y_score, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert returned_path == out_path.resolve()


def test_plot_creates_parent_dirs(tmp_path, valid_df, fitted_pipeline):
    """Plot functions must create parent directories if they don't exist."""
    X = valid_df.drop(columns=["target"])
    y = valid_df["target"]
    y_pred = fitted_pipeline.predict(X)

    deep_path = tmp_path / "a" / "b" / "c" / "cm.png"
    plot_confusion_matrix(y, y_pred, deep_path)

    assert deep_path.exists()


def test_metrics_all_floats(fitted_pipeline, valid_df):
    """All metric values must be Python floats in [0, 1]."""
    X = valid_df.drop(columns=["target"])
    y = valid_df["target"]

    metrics = evaluate(fitted_pipeline, X, y)

    for key, val in metrics.items():
        assert isinstance(val, float), f"{key} is not a float"
        assert 0.0 <= val <= 1.0, f"{key}={val} is outside [0, 1]"
