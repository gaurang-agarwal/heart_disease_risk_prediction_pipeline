"""Tests for ``src.data.ingestion``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data.ingestion import download_dataset, load_raw


def test_download_dataset_writes_file(tmp_path):
    """A mocked successful download writes a non-empty CSV to disk."""
    dest = tmp_path / "nested" / "heart.csv"
    payload = b"age,target\n63,1\n37,0\n"

    mock_response = MagicMock()
    mock_response.content = payload
    mock_response.raise_for_status.return_value = None

    with patch("src.data.ingestion.requests.get", return_value=mock_response) as mock_get:
        result = download_dataset("http://example.com/heart.csv", dest)

    mock_get.assert_called_once()
    assert result == dest
    assert dest.exists()
    assert dest.stat().st_size > 0
    assert dest.read_bytes() == payload


def test_download_dataset_raises_on_unreachable(tmp_path):
    """A network failure is surfaced as a ``ConnectionError``."""
    dest = tmp_path / "heart.csv"

    with patch(
        "src.data.ingestion.requests.get",
        side_effect=requests.exceptions.ConnectionError("unreachable"),
    ):
        with pytest.raises(ConnectionError):
            download_dataset("http://unreachable.invalid/heart.csv", dest)


def test_load_raw_returns_dataframe(sample_csv):
    """``load_raw`` returns a DataFrame with the expected shape and columns."""
    df = load_raw(sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 8
    assert df.shape[1] == 14
    for col in ("age", "sex", "cp", "target"):
        assert col in df.columns


def test_load_raw_missing_file_raises(tmp_path):
    """``load_raw`` raises ``FileNotFoundError`` for a missing path."""
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "does_not_exist.csv")
