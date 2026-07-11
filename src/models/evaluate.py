"""Model evaluation: metrics and diagnostic plots.

Provides metric computation and plot-saving utilities that are called both
during the training pipeline and in the model-experiments notebook.

Typical usage::

    from src.models.evaluate import evaluate, plot_confusion_matrix

    metrics = evaluate(fitted_pipeline, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, Path("screenshots/cm.png"))
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")  # non-interactive backend; set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402


def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute classification metrics on a held-out test set.

    Calls ``.predict()`` for threshold-based metrics and ``.predict_proba()``
    for the ROC-AUC score, so the pipeline must expose both methods.

    Parameters
    ----------
    pipeline:
        A *fitted* scikit-learn ``Pipeline`` with a probabilistic classifier.
    X_test:
        Test feature DataFrame aligned with the pipeline's expected columns.
    y_test:
        True binary labels for the test set.

    Returns
    -------
    dict[str, float]
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``, ``roc_auc``.
    """
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
    }


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    path: Path,
) -> Path:
    """Save a confusion matrix plot as a PNG file.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Hard predictions from the classifier.
    path:
        Destination file path.  Parent directories are created automatically.

    Returns
    -------
    Path
        Resolved absolute path where the PNG was written.

    Raises
    ------
    IOError
        If the file cannot be written (e.g. permission denied).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path.resolve()


def plot_roc_curve(
    y_true: pd.Series,
    y_score: np.ndarray,
    path: Path,
) -> Path:
    """Save an ROC curve plot as a PNG file.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_score:
        Predicted probability scores for the positive class.
    path:
        Destination file path.  Parent directories are created automatically.

    Returns
    -------
    Path
        Resolved absolute path where the PNG was written.

    Raises
    ------
    IOError
        If the file cannot be written (e.g. permission denied).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path.resolve()


def plot_pr_curve(
    y_true: pd.Series,
    y_score: np.ndarray,
    path: Path,
) -> Path:
    """Save a precision-recall curve plot as a PNG file.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_score:
        Predicted probability scores for the positive class.
    path:
        Destination file path.  Parent directories are created automatically.

    Returns
    -------
    Path
        Resolved absolute path where the PNG was written.

    Raises
    ------
    IOError
        If the file cannot be written (e.g. permission denied).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"PR (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path.resolve()
