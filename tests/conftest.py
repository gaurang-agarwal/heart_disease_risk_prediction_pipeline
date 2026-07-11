"""Shared pytest fixtures for the Heart Disease MLOps test suite."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest

# A tiny, schema-correct sample of the Heart Disease dataset. Column order
# matches the raw UCI CSV: 13 features + the ``target`` column.
_RAW_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

_SAMPLE_ROWS = [
    (63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 1),
    (37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2, 1),
    (41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2, 1),
    (56, 1, 1, 120, 236, 0, 1, 178, 0, 0.8, 2, 0, 2, 1),
    (57, 0, 0, 120, 354, 0, 1, 163, 1, 0.6, 2, 0, 2, 1),
    (67, 1, 0, 160, 286, 0, 0, 108, 1, 1.5, 1, 3, 2, 0),
    (67, 1, 0, 120, 229, 0, 0, 129, 1, 2.6, 1, 2, 3, 0),
    (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2, 0),
]


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """A small, valid Heart Disease DataFrame with all expected columns."""
    return pd.DataFrame(_SAMPLE_ROWS, columns=_RAW_COLUMNS)


@pytest.fixture
def sample_csv(tmp_path: Path, valid_df: pd.DataFrame) -> Path:
    """Write ``valid_df`` to a temp CSV and return its path."""
    csv_path = tmp_path / "heart_sample.csv"
    valid_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def invalid_df(valid_df: pd.DataFrame) -> pd.DataFrame:
    """A DataFrame missing a required column (``chol``)."""
    return valid_df.drop(columns=["chol"])


@pytest.fixture
def out_of_range_df(valid_df: pd.DataFrame) -> pd.DataFrame:
    """A DataFrame with values outside the allowed numeric/categorical domains."""
    df = valid_df.copy()
    # Impossible physiological value for a continuous feature.
    df.loc[df.index[0], "age"] = 999
    # Value outside the categorical domain for a discrete feature.
    df.loc[df.index[1], "cp"] = 7
    return df


@pytest.fixture
def dup_df(valid_df: pd.DataFrame) -> pd.DataFrame:
    """A DataFrame containing an exact duplicate row."""
    return pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)


@pytest.fixture
def missing_df(valid_df: pd.DataFrame) -> pd.DataFrame:
    """A DataFrame with NaN values introduced in several columns."""
    df = valid_df.copy()
    df.loc[df.index[0], "age"] = float("nan")
    df.loc[df.index[1], "chol"] = float("nan")
    df.loc[df.index[2], "cp"] = float("nan")
    return df


@pytest.fixture
def tmp_mlflow_uri(tmp_path: Path) -> str:
    """Temporary SQLite MLflow tracking URI, isolated per test."""
    db_path = tmp_path / "mlflow.db"
    uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(uri)
    yield uri
    # Reset to avoid polluting other tests
    mlflow.set_tracking_uri("")


@pytest.fixture
def fitted_pipeline(valid_df: pd.DataFrame):
    """A fitted LogisticRegression Pipeline on the sample DataFrame."""
    from src import config
    from src.features.pipeline import build_preprocessor
    from src.models.train import build_models, train_model

    X = valid_df.drop(columns=["target"])
    y = valid_df["target"]
    preprocessor = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    models = build_models(42)
    return train_model(models["logistic_regression"], preprocessor, X, y)
