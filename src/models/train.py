"""Model training entrypoint, cross-validation, best-model selection, and MLflow tracking.

Provides the model factory, cross-validation, best-model selection, and MLflow
tracking with artifact and plot logging.

Run end-to-end from the command line::

    python -m src.models.train

This reads ``data/raw/heart.csv``, trains LogisticRegression and RandomForest,
logs both runs to MLflow, selects the best model by ROC-AUC, saves the best model at ``models/bundle.joblib``, registers it in ML flow model Registry at Staging, and prints a
comparison table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src import config
from src.features.pipeline import build_preprocessor, save_preprocessor
from src.models.evaluate import (
    evaluate,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)

from src.models.predict import save_bundle
from src.models.registry import promote_to_stage, register_model

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #


def build_models(random_state: int) -> dict[str, BaseEstimator]:
    """Return a dict mapping identifier strings to unfitted estimator instances.

    Parameters
    ----------
    random_state:
        Integer seed applied to all stochastic estimators for reproducibility.

    Returns
    -------
    dict[str, BaseEstimator]
        Keys ``"logistic_regression"`` and ``"random_forest"``, each mapped to
        an *unfitted* scikit-learn estimator ready for ``Pipeline`` assembly.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
    }


def train_model(
    model: BaseEstimator,
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Assemble a preprocessor + model ``Pipeline`` and fit it on training data.

    The pipeline has two named steps:

    * ``"preprocessor"`` — the supplied ``ColumnTransformer``.
    * ``"classifier"`` — the supplied estimator.

    Parameters
    ----------
    model:
        An unfitted scikit-learn estimator (e.g. ``LogisticRegression``).
    preprocessor:
        An unfitted ``ColumnTransformer`` produced by :func:`build_preprocessor`.
    X_train:
        Feature DataFrame for training.
    y_train:
        Target series aligned with *X_train*.

    Returns
    -------
    Pipeline
        A fitted two-step sklearn ``Pipeline``.

    Raises
    ------
    ValueError
        If *X_train* and *y_train* have incompatible lengths.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: list[str] | None = None,
) -> dict[str, float]:
    """Return mean stratified k-fold cross-validation scores per metric.

    sklearn's :func:`~sklearn.model_selection.cross_validate` internally clones
    the pipeline for each fold, so the supplied *pipeline* may be fitted or
    unfitted.

    Parameters
    ----------
    pipeline:
        Fitted or unfitted sklearn ``Pipeline`` to evaluate.
    X:
        Feature DataFrame (training data only — never include the holdout set
        to avoid leakage).
    y:
        Target series aligned with *X*.
    cv:
        Number of stratified folds (default ``5``).
    scoring:
        List of sklearn scorer strings.  Defaults to ``["accuracy",
        "precision", "recall", "f1", "roc_auc"]``.

    Returns
    -------
    dict[str, float]
        Mean score across folds per metric, keyed by metric name.
    """
    if scoring is None:
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    raw = cross_validate(
        pipeline,
        X,
        y,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=False,
    )
    return {metric: float(raw[f"test_{metric}"].mean()) for metric in scoring}


# --------------------------------------------------------------------------- #
# Best-model selection
# --------------------------------------------------------------------------- #


def select_best(
    results: dict[str, dict],
) -> tuple[str, Pipeline, dict]:
    """Pick the best model by ROC-AUC score, using F1 as a tie-breaker.

    Parameters
    ----------
    results:
        Mapping of ``model_name → {"pipeline": Pipeline, "metrics": dict}``.
        The ``"metrics"`` sub-dict must contain keys ``"roc_auc"`` and ``"f1"``.

    Returns
    -------
    tuple[str, Pipeline, dict]
        ``(best_name, best_pipeline, best_metrics)`` — the name of the winning
        model, its fitted ``Pipeline``, and its metrics dict.

    Raises
    ------
    ValueError
        If *results* is empty.
    """
    if not results:
        raise ValueError("results dict is empty; cannot select best model.")

    best_name = max(
        results,
        key=lambda name: (
            results[name]["metrics"]["roc_auc"],
            results[name]["metrics"]["f1"],
        ),
    )
    return best_name, results[best_name]["pipeline"], results[best_name]["metrics"]


# --------------------------------------------------------------------------- #
# MLflow logging
# --------------------------------------------------------------------------- #


def log_experiment(
    model_name: str,
    params: dict,
    metrics: dict,
    artifacts: list[Path],
) -> str:
    """Log params, metrics, and artifacts to the currently active MLflow run.

    This function must be called inside a ``with mlflow.start_run():`` context
    managed by the caller.  It does **not** start or end a run itself.

    Parameters
    ----------
    model_name:
        Human-readable label for the model (logged as a ``"model_name"``
        parameter).
    params:
        Hyper-parameter dict passed to ``mlflow.log_params``.
    metrics:
        Metric dict passed to ``mlflow.log_metrics``.
    artifacts:
        Local ``Path`` objects logged via ``mlflow.log_artifact``.  Missing
        files are silently skipped.

    Returns
    -------
    str
        The ``run_id`` of the active run.

    Raises
    ------
    mlflow.exceptions.MlflowException
        If there is no active run or MLflow encounters a tracking error.
    """
    mlflow.log_param("model_name", model_name)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    for path in artifacts:
        if Path(path).exists():
            mlflow.log_artifact(str(path))
    return mlflow.active_run().info.run_id


# --------------------------------------------------------------------------- #
# Orchestration — main()
# --------------------------------------------------------------------------- #


def main(cfg=None) -> dict:
    """Orchestrate train → CV → evaluate → MLflow-log → select best.

    Reads raw data from ``cfg.RAW_DATA_PATH``, cleans and splits it, trains
    LogisticRegression and RandomForest, logs every run to MLflow (params,
    metrics, confusion matrix, ROC curve, PR curve, and the serialised model),
    then selects and prints the best model.

    Parameters
    ----------
    cfg:
        Config module or namespace with attributes matching :mod:`src.config`.
        Defaults to :mod:`src.config` when ``None``.

    Returns
    -------
    dict
        Keys ``"best_name"`` (str), ``"best_metrics"`` (dict), and
        ``"results"`` (per-model dict with ``pipeline``, ``metrics``,
        ``cv_metrics``, and ``run_id``).
    """
    if cfg is None:
        cfg = config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("Starting training pipeline…")

    # ------------------------------------------------------------------
    # 1. Load & prepare data
    # ------------------------------------------------------------------
    from src.data.ingestion import load_raw
    from src.data.preprocessing import clean, split as data_split

    df = load_raw(cfg.RAW_DATA_PATH)
    df_clean = clean(df)
    X_train, X_test, y_train, y_test = data_split(
        df_clean,
        target_col=cfg.TARGET_COL,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # 2. Output directories
    # ------------------------------------------------------------------
    screenshots_dir = Path(cfg.PROJECT_ROOT) / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    models_dir = Path(cfg.PROJECT_ROOT) / "models"
    models_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Setup MLflow experiment
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.EXPERIMENT_NAME)

    models_dict = build_models(cfg.RANDOM_STATE)
    results: dict[str, dict] = {}

    for model_name, model_estimator in models_dict.items():
        logger.info("Training %s…", model_name)

        fresh_preprocessor = clone(build_preprocessor(cfg.NUMERIC_COLS, cfg.CATEGORICAL_COLS))
        pipeline = train_model(model_estimator, fresh_preprocessor, X_train, y_train)

        # Save the fitted preprocessor extracted from the pipeline
        fitted_prep = pipeline.named_steps["preprocessor"]
        prep_path = models_dir / f"{model_name}_preprocessor.joblib"
        save_preprocessor(fitted_prep, prep_path)

        # Cross-validation on training data only (no leakage from test set)
        cv_metrics = cross_validate_model(
            pipeline,
            X_train,
            y_train,
            cv=5,
        )

        # Holdout evaluation
        metrics = evaluate(pipeline, X_test, y_test)

        # Diagnostic plots
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        cm_path = screenshots_dir / f"{model_name}_confusion_matrix.png"
        roc_path = screenshots_dir / f"{model_name}_roc_curve.png"
        pr_path = screenshots_dir / f"{model_name}_pr_curve.png"

        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_roc_curve(y_test, y_score, roc_path)
        plot_pr_curve(y_test, y_score, pr_path)

        # Hyper-parameters to log
        clf = pipeline.named_steps["classifier"]
        clf_params = {k: str(v) for k, v in clf.get_params().items()}
        log_params = {
            "model_type": model_name,
            "random_state": str(cfg.RANDOM_STATE),
            "test_size": str(cfg.TEST_SIZE),
            **clf_params,
        }

        artifact_paths: list[Path] = [cm_path, roc_path, pr_path, prep_path]

        with mlflow.start_run(run_name=model_name):
            run_id = log_experiment(model_name, log_params, metrics, artifact_paths)
            mlflow.sklearn.log_model(pipeline, "model")

        results[model_name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "cv_metrics": cv_metrics,
            "run_id": run_id,
        }

        logger.info(
            "%-25s  accuracy=%.4f  roc_auc=%.4f",
            model_name,
            metrics["accuracy"],
            metrics["roc_auc"],
        )

    # ------------------------------------------------------------------
    # 4. Select and report best model
    # ------------------------------------------------------------------
    best_name, best_pipeline, best_metrics = select_best(results)
    best_run_id = results[best_name]["run_id"]
    logger.info(
        "Best model: %s Run id: %s (roc_auc=%.4f)", best_name, best_run_id, best_metrics["roc_auc"]
    )

    # Package best model for serving
    bundle_path = models_dir / "bundle.joblib"
    save_bundle(best_pipeline, bundle_path)
    logger.info("Saved best model bundle to %s", bundle_path)

    model_name_mlflow = "heart-disease-clf"
    mv = register_model(best_run_id, model_name=model_name_mlflow)
    (models_dir / "bundle.version").write_text(str(mv.version))
    promote_to_stage(model_name_mlflow, int(mv.version), stage="Staging")
    logger.info(
        "Registered %s version %s at Staging (run_id = %s)",
        model_name_mlflow,
        mv.version,
        best_run_id,
    )

    print("\n=== Model Comparison ===")
    cols = f"{'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
    header = f"{'Model':<25} {cols}"
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        m = res["metrics"]
        marker = " ← BEST" if name == best_name else ""
        print(
            f"{name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['roc_auc']:>10.4f}{marker}"
        )
    print(f"\nBest model: {best_name}  (roc_auc={best_metrics['roc_auc']:.4f})")

    return {
        "best_name": best_name,
        "best_metrics": best_metrics,
        "results": results,
        "best_run_id": best_run_id,
        "bundle_path": bundle_path,
        "model_version": str(mv.version),
    }


if __name__ == "__main__":
    main()
