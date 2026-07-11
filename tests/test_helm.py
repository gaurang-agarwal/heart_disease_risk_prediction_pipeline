"""Tests for the Kubernetes Helm chart."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CHART_DIR = REPO_ROOT / "k8s" / "helm" / "heart-disease"
CHART_YAML = CHART_DIR / "Chart.yaml"
VALUES_YAML = CHART_DIR / "values.yaml"


# ---------------------------------------------------------------------------
# Chart scaffold tests
# ---------------------------------------------------------------------------


def test_chart_yaml_valid():
    """Chart.yaml must exist and parse as valid YAML with required keys."""
    assert CHART_YAML.exists(), "Chart.yaml not found"
    data = yaml.safe_load(CHART_YAML.read_text())
    assert "name" in data, "Chart.yaml missing 'name'"
    assert "version" in data, "Chart.yaml missing 'version'"
    assert "apiVersion" in data, "Chart.yaml missing 'apiVersion'"


def test_values_has_image_repo():
    """values.yaml must define image.repository."""
    assert VALUES_YAML.exists(), "values.yaml not found"
    data = yaml.safe_load(VALUES_YAML.read_text())
    assert "image" in data, "values.yaml missing 'image' key"
    assert "repository" in data["image"], "values.yaml missing 'image.repository'"
    assert data["image"]["repository"], "image.repository must not be empty"


def test_values_has_replica_count():
    """values.yaml must define replicaCount."""
    data = yaml.safe_load(VALUES_YAML.read_text())
    assert "replicaCount" in data, "values.yaml missing 'replicaCount'"
    assert isinstance(data["replicaCount"], int), "replicaCount must be an integer"


def test_values_has_service_config():
    """values.yaml must define service.type and service.port."""
    data = yaml.safe_load(VALUES_YAML.read_text())
    assert "service" in data, "values.yaml missing 'service'"
    assert "type" in data["service"], "values.yaml missing service.type"
    assert "port" in data["service"], "values.yaml missing service.port"


def test_values_has_resources():
    """values.yaml must define resources with requests and limits."""
    data = yaml.safe_load(VALUES_YAML.read_text())
    assert "resources" in data, "values.yaml missing 'resources'"
    assert "requests" in data["resources"], "values.yaml missing resources.requests"
    assert "limits" in data["resources"], "values.yaml missing resources.limits"


def test_values_has_ingress_config():
    """values.yaml must define ingress.enabled and ingress.host."""
    data = yaml.safe_load(VALUES_YAML.read_text())
    assert "ingress" in data, "values.yaml missing 'ingress'"
    assert "enabled" in data["ingress"], "values.yaml missing ingress.enabled"
    assert "host" in data["ingress"], "values.yaml missing ingress.host"


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------


def test_deployment_yaml_exists():
    """templates/deployment.yaml must exist."""
    assert (CHART_DIR / "templates" / "deployment.yaml").exists()


def test_service_yaml_exists():
    """templates/service.yaml must exist."""
    assert (CHART_DIR / "templates" / "service.yaml").exists()


def test_ingress_yaml_exists():
    """templates/ingress.yaml must exist."""
    assert (CHART_DIR / "templates" / "ingress.yaml").exists()


def test_configmap_yaml_exists():
    """templates/configmap.yaml must exist."""
    assert (CHART_DIR / "templates" / "configmap.yaml").exists()


def test_deployment_has_probes():
    """deployment.yaml template must reference liveness and readiness probe paths."""
    deployment_text = (CHART_DIR / "templates" / "deployment.yaml").read_text()
    assert "livenessProbe" in deployment_text, "deployment.yaml missing livenessProbe"
    assert "readinessProbe" in deployment_text, "deployment.yaml missing readinessProbe"


def test_deployment_has_resource_limits():
    """deployment.yaml template must reference resources."""
    deployment_text = (CHART_DIR / "templates" / "deployment.yaml").read_text()
    assert "resources" in deployment_text, "deployment.yaml missing resources section"


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm CLI not installed")
def test_template_renders():
    """helm template must produce parseable YAML without errors."""
    result = subprocess.run(
        ["helm", "template", "test-release", str(CHART_DIR)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"helm template failed:\n{result.stderr}"
    rendered = result.stdout
    docs = list(yaml.safe_load_all(rendered))
    assert len(docs) > 0, "helm template produced no documents"


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm CLI not installed")
def test_render_only():
    """CI-safe: helm template must succeed (no Minikube required)."""
    result = subprocess.run(
        ["helm", "template", "ci-test", str(CHART_DIR)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"helm template failed:\n{result.stderr}"
