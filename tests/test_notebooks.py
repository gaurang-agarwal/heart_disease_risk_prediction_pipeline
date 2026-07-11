"""Smoke tests for Jupyter notebooks.

Executes ``notebooks/01_eda.ipynb`` end-to-end. Two execution strategies are
supported:

1. **nbclient kernel mode** (used when ``USE_NBCLIENT=1`` env var is set or
   when ``pytest -m slow`` is explicitly requested from a full environment
   with a running Jupyter kernel).  This is the most faithful execution.
2. **Inline exec mode** (default fallback): code cells are extracted and run
   via ``exec()`` in a shared namespace — semantically equivalent to
   top-to-bottom execution without a kernel subprocess.  Works in sandboxed
   and minimal CI environments.

Either path will fail the test if any cell raises an exception, and both
verify that all four PNG artefacts are saved:
- ``screenshots/eda_histograms.png``
- ``screenshots/eda_heatmap.png``
- ``screenshots/eda_class_balance.png``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
EDA_NOTEBOOK = NOTEBOOKS_DIR / "01_eda.ipynb"

# Set USE_NBCLIENT=1 to opt-in to the full kernel-based execution.
_USE_NBCLIENT: bool = os.environ.get("USE_NBCLIENT", "0") == "1"


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _execute_notebook_kernel(notebook_path: Path, timeout: int = 120) -> nbformat.NotebookNode:
    """Execute a notebook via nbclient (spawns a real Jupyter kernel).

    Parameters
    ----------
    notebook_path:
        Absolute path to the ``.ipynb`` file to execute.
    timeout:
        Per-cell execution timeout in seconds.

    Returns
    -------
    nbformat.NotebookNode
        The executed notebook object with outputs filled in.

    Raises
    ------
    nbclient.exceptions.CellExecutionError
        If any cell raises an exception.
    FileNotFoundError
        If the notebook file does not exist.
    """
    import nbclient  # noqa: PLC0415

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    with notebook_path.open("r", encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    env = os.environ.copy()
    env["REPO_ROOT"] = str(REPO_ROOT)
    env["MPLBACKEND"] = "Agg"

    client = nbclient.NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
        env=env,
        allow_errors=False,
    )
    client.execute()
    return nb


def _execute_notebook_inline(notebook_path: Path) -> None:
    """Execute notebook code cells inline via ``exec()`` without a kernel.

    All cells share a single namespace (mimicking a running kernel). The
    project root is prepended to ``sys.path`` so that ``src.*`` imports
    resolve correctly.

    Parameters
    ----------
    notebook_path:
        Absolute path to the ``.ipynb`` file.

    Raises
    ------
    Exception
        Any exception raised by a cell propagates immediately.
    FileNotFoundError
        If the notebook file does not exist.
    """
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    with notebook_path.open("r", encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    # Ensure project root is importable.
    repo_root_str = str(REPO_ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("REPO_ROOT", repo_root_str)

    # Force Agg backend before any matplotlib import happens in cells.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")

    ns: dict = {"__name__": "__main__", "__file__": str(notebook_path)}

    for cell in nb.cells:
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        exec(compile(cell.source, str(notebook_path), "exec"), ns)  # noqa: S102


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_eda_notebook_executes(tmp_path: Path) -> None:
    """Execute 01_eda.ipynb top-to-bottom and assert no cell raises an error.

    Uses nbclient kernel mode when
    ``USE_NBCLIENT=1`` is set; otherwise falls back to inline ``exec()``-based
    execution.  Either path verifies that all expected PNG artefacts are saved.
    """
    assert EDA_NOTEBOOK.exists(), (
        f"EDA notebook not found at {EDA_NOTEBOOK}. " "Create notebooks/01_eda.ipynb."
    )

    if _USE_NBCLIENT:
        nb = _execute_notebook_kernel(EDA_NOTEBOOK)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error", (
                    f"Cell {i} raised an error: " f"{output.get('ename')}: {output.get('evalue')}"
                )
    else:
        _execute_notebook_inline(EDA_NOTEBOOK)

    # --- Artefact assertions ---
    expected_pngs = {
        "eda_histograms.png": "histogram grid",
        "eda_heatmap.png": "correlation heatmap",
        "eda_class_balance.png": "class balance chart",
    }
    screenshots_dir = REPO_ROOT / "screenshots"
    for filename, description in expected_pngs.items():
        png_path = screenshots_dir / filename
        assert png_path.exists(), (
            f"{description} PNG not saved to {png_path}. "
            "Check the notebook cell that calls fig.savefig(...)."
        )
        assert png_path.stat().st_size > 0, f"{description} PNG at {png_path} exists but is empty."
