"""Tests for the GitHub Actions CI/CD workflow definition."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_YAML = REPO_ROOT / ".github" / "workflows" / "ci.yml"

REQUIRED_JOBS = {"lint", "test", "train", "docker-build", "deploy"}


def _load_ci() -> dict:
    """Parse and return the CI YAML as a dictionary.

    YAML parses the ``on:`` key as the Python boolean ``True``.  We normalise
    the key back to the string ``"on"`` so the rest of the test code can use
    string-key lookups uniformly.
    """
    assert CI_YAML.exists(), f"CI workflow not found at {CI_YAML}"
    data = yaml.safe_load(CI_YAML.read_text())
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def test_ci_yaml_valid():
    """ci.yml must parse as valid YAML without errors."""
    data = _load_ci()
    assert isinstance(data, dict), "ci.yml must parse to a dict"


def test_ci_has_required_jobs():
    """ci.yml must define all five required jobs: lint, test, train, docker-build, deploy."""
    data = _load_ci()
    jobs = set(data.get("jobs", {}).keys())
    missing = REQUIRED_JOBS - jobs
    assert not missing, f"CI workflow is missing jobs: {missing}"


def test_ci_lint_job_has_flake8_and_black():
    """The lint job must run both flake8 and black --check."""
    data = _load_ci()
    lint_steps = data["jobs"]["lint"]["steps"]
    step_text = " ".join(str(step.get("run", "")) for step in lint_steps)
    assert "flake8" in step_text, "lint job must run flake8"
    assert "black" in step_text, "lint job must run black"


def test_ci_test_job_needs_lint():
    """The test job must depend on the lint job."""
    data = _load_ci()
    test_needs = data["jobs"]["test"].get("needs", [])
    if isinstance(test_needs, str):
        test_needs = [test_needs]
    assert "lint" in test_needs, "test job must need lint"


def test_ci_train_job_needs_test():
    """The train job must depend on the test job."""
    data = _load_ci()
    train_needs = data["jobs"]["train"].get("needs", [])
    if isinstance(train_needs, str):
        train_needs = [train_needs]
    assert "test" in train_needs, "train job must need test"


def test_ci_docker_build_needs_train():
    """The docker-build job must depend on the train job."""
    data = _load_ci()
    needs = data["jobs"]["docker-build"].get("needs", [])
    if isinstance(needs, str):
        needs = [needs]
    assert "train" in needs, "docker-build job must need train"


def test_ci_deploy_needs_docker_build():
    """The deploy job must depend on the docker-build job."""
    data = _load_ci()
    needs = data["jobs"]["deploy"].get("needs", [])
    if isinstance(needs, str):
        needs = [needs]
    assert "docker-build" in needs, "deploy job must need docker-build"


def test_ci_triggers_on_push_and_pr():
    """Workflow must trigger on both push and pull_request events."""
    data = _load_ci()
    triggers = data.get("on", {})
    if isinstance(triggers, list):
        trigger_keys = set(triggers)
    else:
        trigger_keys = set(triggers.keys())
    assert "push" in trigger_keys, "Workflow must trigger on push"
    assert "pull_request" in trigger_keys, "Workflow must trigger on pull_request"
