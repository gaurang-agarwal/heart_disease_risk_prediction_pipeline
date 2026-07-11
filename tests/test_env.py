"""Tests for reproducible environment definitions.

Validates that ``requirements.txt`` has every dependency pinned to an exact
version and that ``environment.yml`` specifies Python 3.11.
"""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"
ENVIRONMENT_FILE = REPO_ROOT / "environment.yml"


def _requirement_lines() -> list[str]:
    """Return non-comment, non-blank lines from requirements.txt."""
    lines = []
    for raw in REQUIREMENTS_FILE.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def test_requirements_pinned():
    """Every non-comment line in requirements.txt must contain '==' (exact pin)."""
    assert REQUIREMENTS_FILE.exists(), "requirements.txt not found"
    unpinned = [ln for ln in _requirement_lines() if "==" not in ln]
    assert unpinned == [], f"Unpinned dependencies found: {unpinned}"


def test_environment_yml_python_version():
    """environment.yml must specify Python 3.11 under the 'dependencies' key."""
    assert ENVIRONMENT_FILE.exists(), "environment.yml not found"
    data = yaml.safe_load(ENVIRONMENT_FILE.read_text())
    deps = data.get("dependencies", [])
    python_entries = [d for d in deps if isinstance(d, str) and d.startswith("python")]
    assert python_entries, "No python dependency found in environment.yml"
    assert any(
        "3.11" in entry for entry in python_entries
    ), f"Python 3.11 not found; got: {python_entries}"


def test_environment_yml_has_name():
    """environment.yml must have a 'name' field set to 'heart-mlops'."""
    data = yaml.safe_load(ENVIRONMENT_FILE.read_text())
    assert (
        data.get("name") == "heart-mlops"
    ), f"Expected name='heart-mlops', got: {data.get('name')}"
