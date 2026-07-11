"""Tests for documentation completeness."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
REPORT = REPO_ROOT / "reports" / "final_report.md"

REQUIRED_README_SECTIONS = [
    "## Overview",
    "## Architecture",
    "## Setup",
    "## Data",
    "## Training",
    "## MLflow",
    "## API",
    "## Docker",
    "## Kubernetes",
    "## Monitoring",
    "## Testing",
    "## CI/CD",
]

REQUIRED_REPORT_HEADERS = [
    "## Marks",
]


def _readme_text() -> str:
    assert README.exists(), "README.md not found"
    return README.read_text()


def _report_text() -> str:
    assert REPORT.exists(), "reports/final_report.md not found"
    return REPORT.read_text()


def test_readme_sections_present():
    """README must contain all required section headers."""
    text = _readme_text()
    missing = []
    for section in REQUIRED_README_SECTIONS:
        # Match section header content case-insensitively (strip ##)
        keyword = section.lstrip("#").strip().lower()
        if keyword not in text.lower():
            missing.append(section)
    assert not missing, f"README missing sections: {missing}"


def test_readme_has_predict_example():
    """README must include an example /predict request."""
    text = _readme_text()
    assert "/predict" in text, "README missing /predict endpoint example"


def test_readme_has_docker_instructions():
    """README must include Docker build/run instructions."""
    text = _readme_text()
    assert "docker" in text.lower(), "README missing Docker instructions"


def test_readme_has_makefile_commands():
    """README must mention Makefile commands or `make`."""
    text = _readme_text()
    assert "make" in text.lower(), "README does not mention Make commands"


def test_report_has_marks_matrix():
    """final_report.md must contain a marks traceability matrix (table)."""
    text = _report_text()
    assert (
        "marks" in text.lower() or "|" in text
    ), "report must include marks traceability matrix (markdown table)"


def test_report_has_model_comparison():
    """final_report.md must contain model comparison results."""
    text = _report_text()
    keywords = ["logistic", "random forest", "roc", "accuracy", "f1"]
    found = [kw for kw in keywords if kw.lower() in text.lower()]
    assert len(found) >= 3, f"Report should mention model comparison; found only: {found}"


def test_report_has_eda_findings():
    """final_report.md must contain EDA findings."""
    text = _report_text()
    assert (
        "eda" in text.lower() or "exploratory" in text.lower()
    ), "Report must include EDA findings"
