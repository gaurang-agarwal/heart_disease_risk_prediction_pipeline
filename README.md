# Heart Disease Risk Prediction — MLOps Pipeline

**BITS Pilani AIMLCZG523 Assignment | 50 marks**

An end-to-end, cloud-ready, reproducible, containerized, monitored, and CI/CD-automated machine learning pipeline for predicting heart disease risk using the UCI Heart Disease dataset (binary classification).

## Overview

- **Dataset:** UCI Heart Disease (303 records, 13 features, binary target)
- **Models:** Logistic Regression + Random Forest (best selected by ROC-AUC)
- **Serving:** FastAPI `/predict` with Prometheus instrumentation
- **Tracking:** MLflow experiment tracking + model registry
- **Deployment:** Docker + Helm chart on Minikube
- **Monitoring:** Prometheus scrape + Grafana dashboard
- **CI/CD:** GitHub Actions (lint → test → train → build → deploy)

---

## Architecture

```
Raw CSV → Validation → EDA → Feature Engineering → Training (MLflow) →
Model Registry → FastAPI /predict → Docker → Kubernetes (Helm) →
Prometheus + Grafana
```

---

## Setup

### Prerequisites

- Python 3.11, pip
- (Optional) Conda, Docker, Minikube, Helm, kubectl

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd heart-disease-mlops

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Conda (alternative)

```bash
conda env create -f environment.yml
conda activate heart-mlops
```

---

## Data

Download the UCI Heart Disease dataset:

```bash
make data
# or
python -m src.data.ingestion
```

Run data quality validation:

```bash
make validate
```

---

## Feature Engineering & Training

Run the complete training pipeline (ingestion → preprocessing → feature engineering → training → MLflow logging → model selection):

```bash
make train
# or
python -m src.models.train
```

View MLflow experiments:

```bash
mlflow ui --host 0.0.0.0 --port 5000 --default-artifact-root "${pwd}/mlruns" --backend-store-uri "sqlite:///${pwd}/mlruns/mlflow.db"
# Open http://localhost:5000
```

---

## MLflow Experiment Tracking

All runs are tracked in MLflow with:
- **Parameters:** model type, hyperparameters, random state, test size
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC (train + CV + test)
- **Artifacts:** confusion matrix, ROC curve, PR curve, preprocessor `.joblib`, model

The best model is registered in the **MLflow Model Registry** at `Staging`.

---

## API Serving

### Local run

```bash
# Set model bundle path
export MODEL_BUNDLE_PATH=models/bundle.joblib

uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service liveness + model load status |
| `/predict` | POST | Binary prediction + probability |
| `/metrics` | GET | Prometheus metrics exposition |
| `/docs` | GET | Swagger UI |

### Example `/predict` request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3,
    "slope": 0, "ca": 0, "thal": 1
  }'
```

Response:

```json
{"prediction": 1, "probability": 0.87, "model_version": "2"}
```

### Standalone inference CLI

```bash
# Single JSON record
python -m src.models.predict --input sample.json --bundle models/bundle.joblib

# Batch CSV
python -m src.models.predict --input records.csv
```

---

## Docker

### Build image

```bash
make build-image
# or
docker build -f docker/Dockerfile.api -t heart-api:latest .
```

### Run container

```bash
docker run -d \
  -p 8000:8000 \
  -e MODEL_BUNDLE_PATH=/app/models/bundle.joblib \
  -v $(pwd)/models:/app/models:ro \
  --name heart-api \
  heart-api:latest
```

---

## Kubernetes (Minikube)

### Prerequisites

- Minikube, kubectl, Helm installed

### Deploy

```bash
# 1. Start Minikube
make minikube-up

# 2. Build and load image
make build-image
make image-load

# 3. Deploy with Helm
make deploy

# 4. Verify
kubectl get pods
kubectl rollout status deployment/heart-disease

# 5. Access service
minikube service heart-disease --url
```

### Undeploy

```bash
make undeploy
```

---

## Monitoring

### Local monitoring stack (Docker Compose)

```bash
make compose-up
```

Services:
- **API:** http://localhost:8000
- **MLflow:** http://localhost:5000
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/admin)

The Grafana dashboard **Heart Disease API Dashboard** auto-provisions with panels for:
- Request rate (req/s)
- p50 / p95 latency
- Error rate
- Total predictions by class (0/1)

### Stop stack

```bash
make compose-down
```

---

## Testing

```bash
# Full test suite
make test
# or
pytest tests/ -v

# Specific test files
pytest tests/test_predict.py -v
pytest tests/test_api.py -v
pytest tests/test_monitoring.py -v
pytest tests/test_helm.py -v
```

---

## CI/CD (GitHub Actions)

The `.github/workflows/ci.yml` pipeline runs on every push/PR:

| Job | Trigger | Steps |
|---|---|---|
| **lint** | push/PR | flake8 + black --check |
| **test** | after lint | pytest --cov |
| **train** | after test | ingest + train + upload model artifact |
| **docker-build** | after train | docker build + smoke test |
| **deploy** | manual / `main` branch | helm upgrade --install |

---

## Linting & Formatting

```bash
# Check
make lint

# Format
make format
```

---

## Project Structure

```
heart-disease-mlops/
├── src/                   # Source code
│   ├── config.py          # Central config (paths, columns, hyperparams)
│   ├── data/              # Ingestion, validation, preprocessing
│   ├── features/          # sklearn ColumnTransformer pipeline
│   ├── models/            # Training, evaluation, registry, inference
│   └── api/               # FastAPI app, schemas, middleware, loader
├── tests/                 # Pytest test suite
├── notebooks/             # EDA and model experiments
├── docker/                # Dockerfile for API
├── k8s/helm/              # Helm chart for Kubernetes
├── monitoring/            # Prometheus + Grafana configs
├── reports/               # Final project report
├── screenshots/           # Deployment and monitoring screenshots
├── .github/workflows/     # GitHub Actions CI/CD
├── docker-compose.yml     # Full local stack
├── Makefile               # Developer shortcuts
├── requirements.txt       # Pinned pip dependencies
└── environment.yml        # Conda environment
```
