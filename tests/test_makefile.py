"""Tests for the developer ``Makefile`` targets."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE_PATH = PROJECT_ROOT / "Makefile"

REQUIRED_TARGETS = ["setup", "data", "validate", "lint", "test"]


def _declared_targets(makefile_text: str) -> set[str]:
    """Return the set of target names declared in ``makefile_text``.

    A target is any line of the form ``name:`` (optionally followed by
    prerequisites), ignoring pattern rules and the leading-tab recipe lines.
    """
    targets: set[str] = set()
    for line in makefile_text.splitlines():
        if line.startswith("\t") or not line.strip():
            continue
        match = re.match(r"^([A-Za-z0-9_-]+)\s*:", line)
        if match:
            targets.add(match.group(1))
    return targets


@pytest.mark.parametrize("target", REQUIRED_TARGETS)
def test_makefile_targets_present(target: str) -> None:
    """Every required developer target is declared in the Makefile."""
    assert MAKEFILE_PATH.exists(), "Makefile is missing from the repo root"
    targets = _declared_targets(MAKEFILE_PATH.read_text(encoding="utf-8"))
    assert target in targets, f"Makefile is missing required target: {target!r}"
