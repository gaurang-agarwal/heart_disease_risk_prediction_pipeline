"""Central configuration for the Heart Disease MLOps pipeline.

Exposes module-level constants for paths, column groupings, split
parameters, and MLflow settings. All values can be overridden via
environment variables so that no path is hardcoded to a specific
machine. Paths are resolved relative to the project root.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Project root is two levels up from this file: <root>/src/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_DATA_PATH: Path = Path(os.getenv("RAW_DATA_PATH", str(DATA_DIR / "raw" / "heart.csv")))
PROCESSED_DIR: Path = Path(os.getenv("PROCESSED_DIR", str(DATA_DIR / "processed")))

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #
TARGET_COL: str = os.getenv("TARGET_COL", "target")

# Continuous / numeric features (StandardScaler).
NUMERIC_COLS: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Categorical features (OneHotEncoder).
CATEGORICAL_COLS: list[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

# All model input features (order preserved), excluding the target.
FEATURE_COLS: list[str] = NUMERIC_COLS + CATEGORICAL_COLS

# --------------------------------------------------------------------------- #
# Split / reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))

# --------------------------------------------------------------------------- #
# Dataset source
# --------------------------------------------------------------------------- #
DATA_URL: str = os.getenv(
    "DATA_URL",
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv",
)

# --------------------------------------------------------------------------- #
# MLflow
# --------------------------------------------------------------------------- #
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{PROJECT_ROOT / 'mlruns' / 'mlflow.db'}",
)
EXPERIMENT_NAME: str = os.getenv("EXPERIMENT_NAME", "heart-disease")
