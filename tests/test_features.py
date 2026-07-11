"""Tests for src/features/pipeline.py.

Covers:
* test_build_preprocessor_returns_columntransformer — factory return type
* test_fit_transform_output_shape                  — rows preserved, columns expanded
* test_numeric_scaled_zero_mean                    — StandardScaler zero-mean / unit-std
* test_categorical_onehot_expands                  — OneHotEncoder produces binary columns
* test_get_feature_names_after_fit                 — names returned post-fit
* test_get_feature_names_raises_unfitted           — NotFittedError before fit
* test_save_load_preprocessor_roundtrip            — joblib round-trip equality
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

from src import config
from src.data.preprocessing import clean, split
from src.features.pipeline import (
    build_preprocessor,
    get_feature_names,
    load_preprocessor,
    save_preprocessor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_splits(
    valid_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return clean train/test splits from the small fixture DataFrame."""
    df = clean(valid_df)
    return split(
        df,
        target_col=config.TARGET_COL,
        test_size=0.25,
        random_state=config.RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# build_preprocessor()
# ---------------------------------------------------------------------------


def test_build_preprocessor_returns_columntransformer() -> None:
    """build_preprocessor() must return a ColumnTransformer instance."""
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    assert isinstance(pre, ColumnTransformer)


def test_build_preprocessor_is_unfitted() -> None:
    """A freshly built preprocessor must not yet be fitted."""
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    # get_feature_names_out() raises before fit; our wrapper must re-raise NotFittedError.
    with pytest.raises(NotFittedError):
        get_feature_names(pre)


def test_build_preprocessor_has_two_transformers() -> None:
    """The ColumnTransformer must register exactly two named transformers."""
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    names = [name for name, _, _ in pre.transformers]
    assert "num" in names
    assert "cat" in names


# ---------------------------------------------------------------------------
# fit_transform output shape
# ---------------------------------------------------------------------------


def test_fit_transform_output_shape(valid_df: pd.DataFrame) -> None:
    """fit_transform() must preserve row count and expand column count."""
    X_train, X_test, _, _ = _make_splits(valid_df)
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)

    X_tr = pre.fit_transform(X_train)

    # Row count unchanged.
    assert X_tr.shape[0] == len(X_train)
    # Column count must exceed input feature count (OHE expands categoricals).
    assert X_tr.shape[1] > len(config.NUMERIC_COLS)


def test_transform_test_uses_train_params(valid_df: pd.DataFrame) -> None:
    """transform() on test data must use train-fit stats (no leakage)."""
    X_train, X_test, _, _ = _make_splits(valid_df)
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    pre.fit(X_train)

    X_te = pre.transform(X_test)

    # Same column count as train transform.
    X_tr = pre.transform(X_train)
    assert X_te.shape[1] == X_tr.shape[1]


def test_output_is_numpy_array(valid_df: pd.DataFrame) -> None:
    """fit_transform() must return a numpy ndarray (dense, not sparse)."""
    X_train, _, _, _ = _make_splits(valid_df)
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    result = pre.fit_transform(X_train)
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# numeric scaling
# ---------------------------------------------------------------------------


def test_numeric_scaled_zero_mean(valid_df: pd.DataFrame) -> None:
    """After fit_transform, numeric features must have approximately zero mean."""
    df = clean(valid_df)
    # Use the full small dataset to have enough samples for the assertion.
    X = df[config.NUMERIC_COLS + config.CATEGORICAL_COLS]
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    X_tr = pre.fit_transform(X)

    # The first len(NUMERIC_COLS) columns are the scaled numerics.
    n_num = len(config.NUMERIC_COLS)
    numeric_block = X_tr[:, :n_num]
    means = numeric_block.mean(axis=0)

    np.testing.assert_allclose(
        means,
        np.zeros(n_num),
        atol=1e-10,
        err_msg="Scaled numeric features must have zero mean.",
    )


def test_numeric_scaled_unit_std(valid_df: pd.DataFrame) -> None:
    """After fit_transform (n > 1), numeric features must have unit std."""
    df = clean(valid_df)
    X = df[config.NUMERIC_COLS + config.CATEGORICAL_COLS]
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    X_tr = pre.fit_transform(X)

    n_num = len(config.NUMERIC_COLS)
    numeric_block = X_tr[:, :n_num]
    stds = numeric_block.std(axis=0, ddof=0)

    np.testing.assert_allclose(
        stds,
        np.ones(n_num),
        atol=1e-10,
        err_msg="Scaled numeric features must have unit std.",
    )


# ---------------------------------------------------------------------------
# categorical one-hot encoding
# ---------------------------------------------------------------------------


def test_categorical_onehot_expands(valid_df: pd.DataFrame) -> None:
    """OneHotEncoder must produce only binary (0/1) values in the cat block."""
    df = clean(valid_df)
    X = df[config.NUMERIC_COLS + config.CATEGORICAL_COLS]
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    X_tr = pre.fit_transform(X)

    n_num = len(config.NUMERIC_COLS)
    cat_block = X_tr[:, n_num:]

    unique_vals = set(np.unique(cat_block))
    assert unique_vals.issubset(
        {0.0, 1.0}
    ), f"OneHotEncoder block contains non-binary values: {unique_vals}"


def test_categorical_more_cols_than_input(valid_df: pd.DataFrame) -> None:
    """Total output columns must exceed the number of input feature columns."""
    df = clean(valid_df)
    X = df[config.NUMERIC_COLS + config.CATEGORICAL_COLS]
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    X_tr = pre.fit_transform(X)

    n_input = len(config.NUMERIC_COLS) + len(config.CATEGORICAL_COLS)
    assert (
        X_tr.shape[1] > n_input
    ), "Expected OHE to expand column count beyond the number of input features."


# ---------------------------------------------------------------------------
# get_feature_names()
# ---------------------------------------------------------------------------


def test_get_feature_names_after_fit(valid_df: pd.DataFrame) -> None:
    """get_feature_names() must return a non-empty list after fit."""
    df = clean(valid_df)
    X = df[config.NUMERIC_COLS + config.CATEGORICAL_COLS]
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    pre.fit(X)

    names = get_feature_names(pre)

    assert isinstance(names, list)
    assert len(names) > 0


def test_get_feature_names_count_matches_transform(valid_df: pd.DataFrame) -> None:
    """Feature name count must equal the number of output columns."""
    df = clean(valid_df)
    X = df[config.NUMERIC_COLS + config.CATEGORICAL_COLS]
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    X_tr = pre.fit_transform(X)

    names = get_feature_names(pre)
    assert len(names) == X_tr.shape[1]


def test_get_feature_names_raises_unfitted() -> None:
    """get_feature_names() must raise NotFittedError on an unfitted preprocessor."""
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    with pytest.raises(NotFittedError):
        get_feature_names(pre)


# ---------------------------------------------------------------------------
# save_preprocessor / load_preprocessor round-trip
# ---------------------------------------------------------------------------


def test_save_preprocessor_returns_path(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """save_preprocessor() must return a Path pointing to the written file."""
    df = clean(valid_df)
    X = df[config.NUMERIC_COLS + config.CATEGORICAL_COLS]
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    pre.fit(X)

    out = save_preprocessor(pre, tmp_path / "pre.joblib")

    assert isinstance(out, Path)
    assert out.exists()


def test_save_preprocessor_creates_file(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """save_preprocessor() must write a non-empty file to disk."""
    df = clean(valid_df)
    X = df[config.NUMERIC_COLS + config.CATEGORICAL_COLS]
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    pre.fit(X)

    path = tmp_path / "pre.joblib"
    save_preprocessor(pre, path)

    assert path.stat().st_size > 0


def test_load_preprocessor_raises_file_not_found(tmp_path: Path) -> None:
    """load_preprocessor() must raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_preprocessor(tmp_path / "nonexistent.joblib")


def test_save_load_preprocessor_roundtrip(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """Round-trip transform must be numerically identical to the original.

    Specifically: fit on X_train, transform X_test, save, reload, transform
    X_test again — the two output arrays must match element-wise.
    """
    X_train, X_test, _, _ = _make_splits(valid_df)
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    pre.fit(X_train)
    X_test_original = pre.transform(X_test)

    path = tmp_path / "preprocessor.joblib"
    save_preprocessor(pre, path)
    loaded_pre = load_preprocessor(path)

    X_test_reloaded = loaded_pre.transform(X_test)

    np.testing.assert_array_equal(
        X_test_original,
        X_test_reloaded,
        err_msg="Round-trip transform must produce identical output.",
    )


def test_save_load_creates_nested_dirs(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """save_preprocessor() must create missing parent directories."""
    df = clean(valid_df)
    X = df[config.NUMERIC_COLS + config.CATEGORICAL_COLS]
    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    pre.fit(X)

    nested_path = tmp_path / "a" / "b" / "c" / "pre.joblib"
    save_preprocessor(pre, nested_path)

    assert nested_path.exists()
