"""Tests for Prometheus config, Grafana dashboard, and docker-compose stack."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMETHEUS_YML = REPO_ROOT / "monitoring" / "prometheus" / "prometheus.yml"
DASHBOARD_JSON = REPO_ROOT / "monitoring" / "grafana" / "dashboards" / "heart_api_dashboard.json"
DATASOURCE_YML = (
    REPO_ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "datasource.yml"
)
DASHBOARDS_PROV = (
    REPO_ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
)
COMPOSE_YML = REPO_ROOT / "docker-compose.yml"

REQUIRED_SERVICES = {"api", "mlflow", "prometheus", "grafana"}


# ---------------------------------------------------------------------------
# Prometheus config
# ---------------------------------------------------------------------------


def test_prometheus_yml_valid():
    """prometheus.yml must exist and parse as valid YAML."""
    assert PROMETHEUS_YML.exists(), "prometheus.yml not found"
    data = yaml.safe_load(PROMETHEUS_YML.read_text())
    assert isinstance(data, dict), "prometheus.yml must parse to a dict"


def test_scrape_target_present():
    """prometheus.yml must contain a scrape job named 'heart-api'."""
    data = yaml.safe_load(PROMETHEUS_YML.read_text())
    jobs = [cfg["job_name"] for cfg in data.get("scrape_configs", [])]
    assert "heart-api" in jobs, f"Expected job 'heart-api'; found: {jobs}"


def test_prometheus_scrape_interval():
    """prometheus.yml global.scrape_interval must be defined."""
    data = yaml.safe_load(PROMETHEUS_YML.read_text())
    assert "global" in data, "prometheus.yml missing 'global' section"
    assert "scrape_interval" in data["global"], "Missing global.scrape_interval"


# ---------------------------------------------------------------------------
# Grafana dashboard
# ---------------------------------------------------------------------------


def test_dashboard_json_valid():
    """heart_api_dashboard.json must be valid JSON with at least one panel."""
    assert DASHBOARD_JSON.exists(), "Grafana dashboard JSON not found"
    data = json.loads(DASHBOARD_JSON.read_text())
    assert "panels" in data, "Dashboard JSON missing 'panels' key"
    assert len(data["panels"]) > 0, "Dashboard must have at least one panel"


def test_datasource_points_to_prometheus():
    """Grafana datasource provisioning must define a Prometheus datasource."""
    assert DATASOURCE_YML.exists(), "datasource.yml not found"
    data = yaml.safe_load(DATASOURCE_YML.read_text())
    sources = data.get("datasources", [])
    types = [ds.get("type") for ds in sources]
    assert "prometheus" in types, f"No prometheus datasource found; types: {types}"


def test_dashboard_provisioning_valid():
    """dashboards.yml must exist and parse as valid YAML."""
    assert DASHBOARDS_PROV.exists(), "dashboards provisioning YAML not found"
    data = yaml.safe_load(DASHBOARDS_PROV.read_text())
    assert "providers" in data, "dashboards.yml must have 'providers'"


def test_dashboard_has_request_rate_panel():
    """Dashboard must include a panel related to request rate."""
    data = json.loads(DASHBOARD_JSON.read_text())
    titles = [p.get("title", "").lower() for p in data["panels"]]
    rate_panels = [t for t in titles if "request" in t or "rate" in t]
    assert rate_panels, f"No request-rate panel found; panel titles: {titles}"


def test_dashboard_has_predictions_panel():
    """Dashboard must include a panel related to predictions."""
    data = json.loads(DASHBOARD_JSON.read_text())
    titles = [p.get("title", "").lower() for p in data["panels"]]
    pred_panels = [t for t in titles if "predict" in t]
    assert pred_panels, f"No predictions panel found; panel titles: {titles}"


# ---------------------------------------------------------------------------
# docker-compose stack
# ---------------------------------------------------------------------------


def test_compose_yaml_valid():
    """docker-compose.yml must exist and parse as valid YAML."""
    assert COMPOSE_YML.exists(), "docker-compose.yml not found"
    data = yaml.safe_load(COMPOSE_YML.read_text())
    assert isinstance(data, dict), "docker-compose.yml must parse to a dict"


def test_compose_has_all_services():
    """docker-compose.yml must define api, mlflow, prometheus, and grafana services."""
    data = yaml.safe_load(COMPOSE_YML.read_text())
    services = set(data.get("services", {}).keys())
    missing = REQUIRED_SERVICES - services
    assert not missing, f"docker-compose.yml is missing services: {missing}"


def test_compose_api_has_health_check():
    """The api service must define a healthcheck."""
    data = yaml.safe_load(COMPOSE_YML.read_text())
    api_service = data["services"]["api"]
    assert "healthcheck" in api_service, "api service missing healthcheck"


def test_compose_volumes_defined():
    """docker-compose.yml must define named volumes."""
    data = yaml.safe_load(COMPOSE_YML.read_text())
    assert "volumes" in data, "docker-compose.yml missing 'volumes' key"
    assert len(data["volumes"]) > 0, "At least one named volume must be defined"
