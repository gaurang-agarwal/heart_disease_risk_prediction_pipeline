"""Tests for ``src.config``."""

from pathlib import Path

from src import config


def test_config_paths_defined():
    """Path constants exist and are ``pathlib.Path`` instances."""
    for attr in ("RAW_DATA_PATH", "PROCESSED_DIR", "PROJECT_ROOT", "DATA_DIR"):
        assert hasattr(config, attr), f"config missing {attr}"
        assert isinstance(getattr(config, attr), Path)


def test_column_lists_disjoint():
    """Numeric and categorical feature lists must not overlap."""
    numeric = set(config.NUMERIC_COLS)
    categorical = set(config.CATEGORICAL_COLS)

    assert numeric, "NUMERIC_COLS must be non-empty"
    assert categorical, "CATEGORICAL_COLS must be non-empty"
    assert numeric & categorical == set()
    assert config.TARGET_COL not in numeric | categorical


def test_scalar_constants_types():
    """Split/seed constants and MLflow settings have the expected types."""
    assert config.TARGET_COL == "target"
    assert config.RANDOM_STATE == 42
    assert isinstance(config.RANDOM_STATE, int)
    assert config.TEST_SIZE == 0.2
    assert isinstance(config.TEST_SIZE, float)
    assert isinstance(config.MLFLOW_TRACKING_URI, str)
    assert isinstance(config.EXPERIMENT_NAME, str)
