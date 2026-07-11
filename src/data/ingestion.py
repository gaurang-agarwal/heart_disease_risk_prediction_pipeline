"""Data ingestion for the Heart Disease MLOps pipeline.

Downloads the UCI Heart Disease CSV from a configurable mirror URL to a
local destination and loads it into a :class:`pandas.DataFrame`. The
download URL and destination path are sourced from :mod:`src.config`
(overridable via environment variables) so that nothing is hardcoded to
a specific machine.

Run as a module to fetch the raw dataset::

    python -m src.data.ingestion            # download to RAW_DATA_PATH
    python -m src.data.ingestion --verify   # download then verify columns
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from src import config

# Columns the raw UCI Heart Disease dataset is expected to contain.
EXPECTED_COLUMNS: list[str] = config.FEATURE_COLS + [config.TARGET_COL]


def download_dataset(url: str, dest_path: Path, timeout: int = 30) -> Path:
    """Download the UCI Heart Disease CSV from ``url`` to ``dest_path``.

    Parameters
    ----------
    url:
        HTTP(S) URL of the raw CSV dataset.
    dest_path:
        Filesystem location to write the downloaded CSV to. Parent
        directories are created if they do not already exist.
    timeout:
        Per-request timeout in seconds. Defaults to ``30``.

    Returns
    -------
    pathlib.Path
        The resolved path that was written.

    Raises
    ------
    ConnectionError
        If the dataset cannot be reached or returns an error status.
    IOError
        If the downloaded content cannot be written to ``dest_path``.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise ConnectionError(f"Failed to download dataset from {url!r}: {exc}") from exc

    try:
        dest_path.write_bytes(response.content)
    except OSError as exc:
        raise IOError(f"Failed to write dataset to {dest_path!r}: {exc}") from exc

    return dest_path


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw CSV at ``path`` into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    path:
        Path to the raw CSV file.

    Returns
    -------
    pandas.DataFrame
        The parsed dataset.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path!r}")
    return pd.read_csv(path)


def main() -> None:
    """CLI: download the raw dataset and optionally verify its schema."""
    parser = argparse.ArgumentParser(description="Download the raw Heart Disease dataset.")
    parser.add_argument(
        "--url",
        default=config.DATA_URL,
        help="Source URL for the raw CSV (defaults to config.DATA_URL).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=config.RAW_DATA_PATH,
        help="Destination path (defaults to config.RAW_DATA_PATH).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After downloading, assert the expected columns are present.",
    )
    args = parser.parse_args()

    dest = download_dataset(args.url, args.dest, timeout=args.timeout)
    df = load_raw(dest)
    print(f"Downloaded {len(df)} rows to {dest}")

    if args.verify:
        missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing:
            raise SystemExit(f"Verification failed; missing columns: {missing}")
        print(f"Verification passed; {len(df.columns)} columns present.")


if __name__ == "__main__":
    main()
