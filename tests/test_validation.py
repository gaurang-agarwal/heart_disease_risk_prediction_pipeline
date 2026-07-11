"""Tests for ``src.data.validation``."""

from __future__ import annotations

import pytest

from src.data.validation import (
    SchemaError,
    ValidationReport,
    run_validation,
    validate_ranges,
    validate_schema,
)


def test_validate_schema_accepts_valid(valid_df):
    """A schema-correct DataFrame passes without raising."""
    assert validate_schema(valid_df) is None


def test_validate_schema_rejects_missing_column(invalid_df):
    """A DataFrame missing a required column raises ``SchemaError``."""
    with pytest.raises(SchemaError):
        validate_schema(invalid_df)


def test_validate_ranges_flags_out_of_range(out_of_range_df):
    """Out-of-range numeric and categorical values are flagged."""
    flags = validate_ranges(out_of_range_df)

    assert isinstance(flags, dict)
    assert len(flags) > 0
    assert "age" in flags
    assert "cp" in flags


def test_validate_ranges_valid_is_empty(valid_df):
    """A valid DataFrame produces no range violations."""
    assert validate_ranges(valid_df) == {}


def test_run_validation_report_shape(valid_df):
    """A valid DataFrame yields a passing report with the expected structure."""
    report = run_validation(valid_df)

    assert isinstance(report, ValidationReport)
    assert report.passed is True
    assert report.errors == []
    assert report.n_rows == len(valid_df)
    assert report.n_nulls == {}


def test_run_validation_report_fails_on_corrupt(out_of_range_df):
    """A corrupted DataFrame yields a failing report with error messages."""
    report = run_validation(out_of_range_df)

    assert report.passed is False
    assert len(report.errors) > 0
