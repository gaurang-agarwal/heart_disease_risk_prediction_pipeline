.PHONY: setup data validate lint format test clean train \
        build-image minikube-up image-load deploy undeploy \
        compose-up compose-down

PYTHON      ?= python
IMAGE_NAME  ?= heart-api
IMAGE_TAG   ?= latest
CHART_DIR   ?= k8s/helm/heart-disease
RELEASE     ?= heart-disease
NAMESPACE   ?= default

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------
setup:  ## Install project dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:  ## Download the raw UCI Heart Disease dataset to RAW_DATA_PATH
	$(PYTHON) -m src.data.ingestion --verify

validate:  ## Run schema + data-quality validation on the raw dataset
	$(PYTHON) -c "from src.data.ingestion import load_raw; from src.data.validation import run_validation; from src import config; report = run_validation(load_raw(config.RAW_DATA_PATH)); print(report); raise SystemExit(0 if report.passed else 1)"

lint:  ## Run flake8 + black --check
	$(PYTHON) -m flake8 .
	$(PYTHON) -m black --check .

format:  ## Auto-format with black
	$(PYTHON) -m black .

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -v

train:  ## Run the full training pipeline
	$(PYTHON) -m src.data.ingestion
	$(PYTHON) -m src.models.train

clean:  ## Remove caches and build artifacts
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov coverage.xml

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
build-image:  ## Build the API Docker image
	docker build -f docker/Dockerfile.api -t $(IMAGE_NAME):$(IMAGE_TAG) .

# ---------------------------------------------------------------------------
# Minikube / Kubernetes deployment
# ---------------------------------------------------------------------------
minikube-up:  ## Start a local Minikube cluster
	minikube start --driver=docker --cpus=2 --memory=4096

image-load:  ## Load the local Docker image into Minikube
	minikube image load $(IMAGE_NAME):$(IMAGE_TAG)

deploy:  ## Install/upgrade the Helm chart on Minikube
	helm upgrade --install $(RELEASE) $(CHART_DIR) \
		--set image.repository=$(IMAGE_NAME) \
		--set image.tag=$(IMAGE_TAG) \
		--namespace $(NAMESPACE) \
		--create-namespace \
		--wait

undeploy:  ## Uninstall the Helm release from Minikube
	helm uninstall $(RELEASE) --namespace $(NAMESPACE)

# ---------------------------------------------------------------------------
# Docker Compose (local monitoring stack)
# ---------------------------------------------------------------------------
compose-up:  ## Start the full local stack (API + MLflow + Prometheus + Grafana)
	docker compose up -d --build

compose-down:  ## Stop and remove the local stack
	docker compose down -v
