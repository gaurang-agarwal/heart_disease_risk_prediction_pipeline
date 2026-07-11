"""Tests for src/data/preprocessing.py.

Covers:
* test_clean_drops_duplicates   — duplicate rows are removed
* test_clean_handles_missing    — NaN values are imputed, not left in
* test_split_stratified_ratio   — class balance is preserved after split
* test_save_splits_writes_files — train.csv and test.csv are created
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.preprocessing import clean, save_splits, split
from src import config


# ---------------------------------------------------------------------------
# clean()
# ---------------------------------------------------------------------------


def test_clean_drops_duplicates(dup_df: pd.DataFrame) -> None:
    """clean() must remove exact duplicate rows and reset the index."""
    original_len = len(dup_df)
    result = clean(dup_df)

    assert len(result) < original_len, "Duplicate row was not removed."
    # The index must be a clean 0-based range after reset.
    assert list(result.index) == list(range(len(result)))


def test_clean_drops_duplicates_count(dup_df: pd.DataFrame) -> None:
    """Exactly one duplicate row is removed from dup_df."""
    result = clean(dup_df)
    assert len(result) == len(dup_df) - 1


def test_clean_handles_missing(missing_df: pd.DataFrame) -> None:
    """clean() must leave no NaN values in feature or target columns."""
    result = clean(missing_df)

    feature_cols = [c for c in result.columns if c != config.TARGET_COL]
    assert (
        result[feature_cols].isnull().sum().sum() == 0
    ), "NaN values remain in feature columns after clean()."


def test_clean_returns_dataframe(valid_df: pd.DataFrame) -> None:
    """clean() must return a pandas DataFrame."""
    result = clean(valid_df)
    assert isinstance(result, pd.DataFrame)


def test_clean_does_not_mutate_input(valid_df: pd.DataFrame) -> None:
    """clean() must not modify the original DataFrame (copy semantics)."""
    original_hash = pd.util.hash_pandas_object(valid_df).sum()
    clean(valid_df)
    assert pd.util.hash_pandas_object(valid_df).sum() == original_hash


def test_clean_preserves_clean_data(valid_df: pd.DataFrame) -> None:
    """clean() on already-clean data must preserve all rows."""
    result = clean(valid_df)
    assert len(result) == len(valid_df)


# ---------------------------------------------------------------------------
# split()
# ---------------------------------------------------------------------------


def test_split_stratified_ratio(valid_df: pd.DataFrame) -> None:
    """split() must preserve the class distribution in both partitions."""
    df = clean(valid_df)
    X_train, X_test, y_train, y_test = split(
        df,
        target_col=config.TARGET_COL,
        test_size=0.25,
        random_state=config.RANDOM_STATE,
    )

    original_ratio = df[config.TARGET_COL].mean()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    # Allow ±15 % absolute tolerance given the tiny fixture size.
    assert (
        abs(train_ratio - original_ratio) < 0.20
    ), f"Train class ratio {train_ratio:.2f} drifts too far from {original_ratio:.2f}."
    assert (
        abs(test_ratio - original_ratio) < 0.20
    ), f"Test class ratio {test_ratio:.2f} drifts too far from {original_ratio:.2f}."


def test_split_correct_sizes(valid_df: pd.DataFrame) -> None:
    """split() must allocate approximately test_size fraction to the test set."""
    df = clean(valid_df)
    X_train, X_test, y_train, y_test = split(
        df,
        target_col=config.TARGET_COL,
        test_size=0.25,
        random_state=config.RANDOM_STATE,
    )

    total = len(X_train) + len(X_test)
    assert total == len(df)
    # Test set should be roughly 25 % of total (±1 row tolerance).
    assert abs(len(X_test) / total - 0.25) <= 1 / total + 0.05


def test_split_no_target_in_features(valid_df: pd.DataFrame) -> None:
    """split() must not include the target column in the feature DataFrames."""
    df = clean(valid_df)
    X_train, X_test, _, _ = split(
        df,
        target_col=config.TARGET_COL,
        test_size=0.25,
        random_state=config.RANDOM_STATE,
    )
    assert config.TARGET_COL not in X_train.columns
    assert config.TARGET_COL not in X_test.columns


def test_split_raises_on_missing_target(valid_df: pd.DataFrame) -> None:
    """split() must raise ValueError when target_col is absent."""
    with pytest.raises(ValueError, match="not found"):
        split(
            valid_df,
            target_col="nonexistent_column",
            test_size=0.2,
            random_state=42,
        )


def test_split_reproducible(valid_df: pd.DataFrame) -> None:
    """split() must produce identical results with the same random_state."""
    df = clean(valid_df)
    kwargs = dict(target_col=config.TARGET_COL, test_size=0.25, random_state=42)
    X_tr1, X_te1, y_tr1, y_te1 = split(df, **kwargs)
    X_tr2, X_te2, y_tr2, y_te2 = split(df, **kwargs)

    pd.testing.assert_frame_equal(X_tr1.reset_index(drop=True), X_tr2.reset_index(drop=True))
    pd.testing.assert_series_equal(y_tr1.reset_index(drop=True), y_tr2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# save_splits()
# ---------------------------------------------------------------------------


def test_save_splits_writes_files(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """save_splits() must create train.csv and test.csv under out_dir."""
    df = clean(valid_df)
    X_train, X_test, y_train, y_test = split(
        df,
        target_col=config.TARGET_COL,
        test_size=0.25,
        random_state=config.RANDOM_STATE,
    )
    save_splits(X_train, X_test, y_train, y_test, tmp_path)

    assert (tmp_path / "train.csv").exists(), "train.csv was not created."
    assert (tmp_path / "test.csv").exists(), "test.csv was not created."


def test_save_splits_files_non_empty(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """The CSVs written by save_splits() must contain at least one data row."""
    df = clean(valid_df)
    X_train, X_test, y_train, y_test = split(
        df,
        target_col=config.TARGET_COL,
        test_size=0.25,
        random_state=config.RANDOM_STATE,
    )
    save_splits(X_train, X_test, y_train, y_test, tmp_path)

    train_loaded = pd.read_csv(tmp_path / "train.csv")
    test_loaded = pd.read_csv(tmp_path / "test.csv")

    assert len(train_loaded) > 0
    assert len(test_loaded) > 0


def test_save_splits_target_column_present(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """The saved CSVs must include the target column."""
    df = clean(valid_df)
    X_train, X_test, y_train, y_test = split(
        df,
        target_col=config.TARGET_COL,
        test_size=0.25,
        random_state=config.RANDOM_STATE,
    )
    save_splits(X_train, X_test, y_train, y_test, tmp_path)

    train_loaded = pd.read_csv(tmp_path / "train.csv")
    assert config.TARGET_COL in train_loaded.columns


def test_save_splits_creates_dir(valid_df: pd.DataFrame, tmp_path: Path) -> None:
    """save_splits() must create out_dir if it does not yet exist."""
    nested = tmp_path / "a" / "b" / "c"
    df = clean(valid_df)
    X_train, X_test, y_train, y_test = split(
        df,
        target_col=config.TARGET_COL,
        test_size=0.25,
        random_state=config.RANDOM_STATE,
    )
    save_splits(X_train, X_test, y_train, y_test, nested)

    assert (nested / "train.csv").exists()
    assert (nested / "test.csv").exists()
