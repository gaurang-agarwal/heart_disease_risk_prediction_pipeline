"""Schema and data-quality validation.

Provides schema checks (required columns and numeric dtypes), range/domain
checks for each feature, and an aggregate :func:`run_validation` entrypoint
that returns a structured :class:`ValidationReport`. Expected columns and the
target name are sourced from :mod:`src.config` so nothing is hardcoded.

Run as a module to validate the raw dataset::

    python -m src.data.validation            # validate RAW_DATA_PATH
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src import config


class SchemaError(Exception):
    """Raised when a DataFrame does not conform to the expected schema."""


@dataclass
class ValidationReport:
    """Structured validation outcome.

    Attributes
    ----------
    passed:
        ``True`` when no schema errors, range violations, or nulls were found.
    errors:
        Human-readable descriptions of every problem detected.
    n_rows:
        Number of rows in the validated DataFrame.
    n_nulls:
        Mapping of column name to its null count (only columns with at least
        one null are included).
    """

    passed: bool
    errors: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_nulls: dict[str, int] = field(default_factory=dict)


# Inclusive numeric ranges for continuous features.
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (1.0, 120.0),
    "trestbps": (80.0, 220.0),
    "chol": (100.0, 600.0),
    "thalach": (60.0, 250.0),
    "oldpeak": (0.0, 10.0),
}

# Allowed discrete value sets for categorical features (and the target).
CATEGORICAL_ALLOWED: dict[str, set[int]] = {
    "sex": {0, 1},
    "cp": {0, 1, 2, 3},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {0, 1, 2},
    "ca": {0, 1, 2, 3, 4},
    "thal": {0, 1, 2, 3},
    config.TARGET_COL: {0, 1},
}

# Every column the raw dataset is expected to contain.
REQUIRED_COLUMNS: list[str] = config.FEATURE_COLS + [config.TARGET_COL]


def validate_schema(df: pd.DataFrame) -> None:
    """Assert that all required columns are present with numeric dtypes.

    Parameters
    ----------
    df:
        DataFrame to check against the expected Heart Disease schema.

    Raises
    ------
    SchemaError
        If any required column is missing or has a non-numeric dtype.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

    non_numeric = [col for col in REQUIRED_COLUMNS if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise SchemaError(f"Non-numeric columns: {non_numeric}")


def validate_ranges(df: pd.DataFrame) -> dict[str, list]:
    """Return per-column row indices whose values fall outside allowed ranges.

    Parameters
    ----------
    df:
        DataFrame to inspect. Missing columns are skipped.

    Returns
    -------
    dict[str, list]
        Mapping of column name to the list of row-index labels that violate
        the column's numeric range or categorical domain. Only columns with at
        least one violation are included.
    """
    flags: dict[str, list] = {}

    for col, (low, high) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        mask = (df[col] < low) | (df[col] > high)
        offending = df.index[mask].tolist()
        if offending:
            flags[col] = offending

    for col, allowed in CATEGORICAL_ALLOWED.items():
        if col not in df.columns:
            continue
        mask = ~df[col].isin(allowed)
        offending = df.index[mask].tolist()
        if offending:
            flags[col] = offending

    return flags


def run_validation(df: pd.DataFrame) -> ValidationReport:
    """Run all schema and quality checks and produce a report.

    Parameters
    ----------
    df:
        DataFrame to validate.

    Returns
    -------
    ValidationReport
        Aggregated outcome with pass/fail, error messages, row count, and
        per-column null counts.
    """
    errors: list[str] = []

    try:
        validate_schema(df)
    except SchemaError as exc:
        errors.append(str(exc))

    null_counts = df.isnull().sum()
    n_nulls = {col: int(count) for col, count in null_counts.items() if count > 0}
    if n_nulls:
        errors.append(f"Null values found: {n_nulls}")

    range_flags = validate_ranges(df)
    if range_flags:
        summary = {col: len(idx) for col, idx in range_flags.items()}
        errors.append(f"Out-of-range values found: {summary}")

    return ValidationReport(
        passed=not errors,
        errors=errors,
        n_rows=int(len(df)),
        n_nulls=n_nulls,
    )


def main() -> None:
    """CLI: validate the raw dataset at ``config.RAW_DATA_PATH`` and print report."""
    from src.data.ingestion import load_raw

    df = load_raw(config.RAW_DATA_PATH)
    report = run_validation(df)
    print(f"Rows: {report.n_rows}")
    print(f"Nulls: {report.n_nulls}")
    print(f"Passed: {report.passed}")
    for err in report.errors:
        print(f"  - {err}")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
