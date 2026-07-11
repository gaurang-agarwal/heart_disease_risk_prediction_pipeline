"""Data cleaning, stratified splitting, and split persistence.

Provides three functions that form the preprocessing backbone of the pipeline:

* :func:`clean` — removes duplicates, coerces dtypes, and median/mode-imputes
  missing values so that downstream feature engineering never sees NaNs.
* :func:`split` — performs a stratified train/test split using the label
  distribution, ensuring both partitions reflect the full class balance.
* :func:`save_splits` — writes the four arrays as two CSVs to ``data/processed/``
  so they can be loaded reproducibly by later pipeline stages.

Run as a module to execute the full preprocessing flow::

    python -m src.data.preprocessing    # writes data/processed/train.csv + test.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, coerce dtypes, and impute missing values.

    Cleaning steps applied in order:

    1. Drop exact-duplicate rows and reset the integer index.
    2. Coerce every column to numeric (non-parseable values become NaN).
    3. Impute missing values in numeric features with the column median.
    4. Impute missing values in categorical features with the column mode.
    5. Drop any remaining rows whose *target* value is NaN (cannot train without a label).

    Parameters
    ----------
    df:
        Raw Heart Disease DataFrame, as returned by
        :func:`src.data.ingestion.load_raw`.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of *df* with no duplicate rows and no NaNs in feature or
        target columns.
    """
    df = df.copy()

    df = df.drop_duplicates().reset_index(drop=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in config.NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in config.CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])

    if config.TARGET_COL in df.columns:
        df = df.dropna(subset=[config.TARGET_COL]).reset_index(drop=True)

    return df


def split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split → (X_train, X_test, y_train, y_test).

    Uses :func:`sklearn.model_selection.train_test_split` with
    ``stratify=y`` so that the class distribution of the target column is
    preserved in both partitions.

    Parameters
    ----------
    df:
        Cleaned Heart Disease DataFrame produced by :func:`clean`.
    target_col:
        Name of the binary target column (``"target"`` by default).
    test_size:
        Fraction of rows to allocate to the test set (e.g. ``0.2`` for 20 %).
    random_state:
        Integer seed passed to :func:`train_test_split` for reproducibility.

    Returns
    -------
    tuple[DataFrame, DataFrame, Series, Series]
        ``(X_train, X_test, y_train, y_test)`` — feature DataFrames and label
        Series with aligned indices.

    Raises
    ------
    ValueError
        If *target_col* is not a column of *df*.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def save_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    out_dir: Path,
) -> None:
    """Persist train and test splits as CSV files under *out_dir*.

    Each output CSV contains all feature columns **plus** the target column
    appended as the last column, matching the layout of the original raw file.
    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    X_train:
        Training feature DataFrame.
    X_test:
        Test feature DataFrame.
    y_train:
        Training labels (Series whose ``.name`` attribute is used as the target
        column name; falls back to ``"target"`` if ``None``).
    y_test:
        Test labels.
    out_dir:
        Directory path where ``train.csv`` and ``test.csv`` are written.

    Raises
    ------
    IOError
        If the directory cannot be created or the CSVs cannot be written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_name: str = y_train.name if y_train.name is not None else "target"

    train_df = X_train.copy()
    train_df[target_name] = y_train.values
    train_df.to_csv(out_dir / "train.csv", index=False)

    test_df = X_test.copy()
    test_df[target_name] = y_test.values
    test_df.to_csv(out_dir / "test.csv", index=False)


def main() -> None:
    """CLI: clean raw data, split, and write processed CSVs.

    Reads ``config.RAW_DATA_PATH``, cleans, splits (stratified), then persists
    ``train.csv`` and ``test.csv`` to ``config.PROCESSED_DIR``.
    """
    from src.data.ingestion import load_raw

    df = load_raw(config.RAW_DATA_PATH)
    df_clean = clean(df)

    X_train, X_test, y_train, y_test = split(
        df_clean,
        target_col=config.TARGET_COL,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    save_splits(X_train, X_test, y_train, y_test, config.PROCESSED_DIR)
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"Splits written to: {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
