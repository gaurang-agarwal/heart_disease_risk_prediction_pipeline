# Final Report — Heart Disease Risk Prediction MLOps Pipeline

**Course:** BITS Pilani AIMLCZG523 — MLOps  
**Assignment:** End-to-End MLOps Pipeline  
**Dataset:** UCI Heart Disease (binary classification)  
**Stack:** Python 3.11 · scikit-learn · MLflow · FastAPI · Docker · Minikube · Helm · Prometheus · Grafana · GitHub Actions

---

## 1. Approach & Methodology

This project implements a production-grade MLOps pipeline for predicting heart disease risk from the UCI Heart Disease dataset. The pipeline covers every phase of the ML lifecycle:

1. **Data Acquisition & Validation** — Automated download from UCI mirror; schema and range validation with structured `ValidationReport`.
2. **Exploratory Data Analysis** — Univariate distributions, correlation heatmap, class balance, bivariate feature-vs-target analysis (Student's t-test for significance).
3. **Feature Engineering** — `sklearn.compose.ColumnTransformer`: `StandardScaler` for continuous features, `OneHotEncoder` (handle_unknown='ignore') for categoricals. Preprocessor serialised to `.joblib` for reproducible inference.
4. **Model Training** — `LogisticRegression` and `RandomForestClassifier`; 5-fold stratified cross-validation; test-set evaluation.
5. **MLflow Tracking** — Each run logs hyperparameters, CV metrics, test metrics, confusion matrix, ROC, and PR curve artifacts. Best model registered in the Model Registry.
6. **Model Packaging** — Full pipeline (preprocessor + model) serialised as a `.joblib` bundle for portable inference.
7. **API Serving** — FastAPI `/predict`, `/health`, `/metrics`; Pydantic validation; Prometheus instrumentation; non-root Docker image.
8. **Kubernetes Deployment** — Helm chart with liveness/readiness probes, resource limits, ConfigMap env injection, optional Ingress and ServiceMonitor.
9. **Monitoring** — Prometheus scraping `/metrics`; Grafana dashboard with request rate, p50/p95 latency, error rate, and predictions-by-class panels.
10. **CI/CD** — GitHub Actions: lint → test → train → docker-build (smoke) → deploy (gated).

---

## 2. EDA Findings

### Dataset Summary
- **Rows:** 303 | **Features:** 13 | **Target:** binary (0 = no disease, 1 = disease)
- **Class balance:** ~54% positive (heart disease) vs ~46% negative — mild imbalance; no oversampling applied.

### Key Insights

1. **Age vs. Target:** Patients with heart disease (class 1) tend to be slightly younger than non-diseased patients — counter-intuitive but statistically significant (p < 0.05, t-test). This reflects the dataset's clinic referral bias.
2. **Thalach (Max Heart Rate) is the strongest discriminator:** Heart-disease patients achieve significantly higher `thalach` (p < 0.001). High max heart rate during exercise indicates cardiac reserve — disease burden lowers resting thresholds but not peak effort in this cohort.
3. **Chest Pain Type (cp) highly predictive:** Asymptomatic chest pain (cp=0) is paradoxically most associated with disease (likely silent ischaemia). Typical angina (cp=3) patients show higher disease prevalence after controlling for other factors.
4. **Oldpeak (ST depression) separates classes well:** Non-zero ST depression is a reliable biomarker; median oldpeak for class 0 is ~1.6, for class 1 ~0.5.

### Correlation
- `thalach` and `slope` are positively correlated with class 1.
- `exang`, `oldpeak`, and `ca` are negatively correlated (higher → class 0).
- Multicollinearity is low among continuous features.

---

## 3. Model Comparison

| Model | CV Accuracy | CV ROC-AUC | Test Accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.84 | ~0.91 | ~0.84 | ~0.85 | ~0.92 |
| Random Forest | ~0.83 | ~0.90 | ~0.85 | ~0.86 | ~0.91 |

**Winner:** Logistic Regression selected (highest ROC-AUC), which is consistent with the near-linear separability of the feature space after scaling. Both models perform within 1–2% of each other, confirming the dataset is well-suited to linear methods.

> Note: Exact values vary by run; MLflow UI captures the reproducible logged numbers.

---

## 4. MLflow Experiment Tracking

- **Experiment name:** `heart-disease`
- **Backend store:** SQLite (`mlruns/mlflow.db`) locally; file-based in CI.
- **Two runs** logged per execution (one per model) with:
  - All sklearn hyperparameters
  - 5-fold CV mean ± std for accuracy, precision, recall, F1, ROC-AUC
  - Test-set evaluation metrics
  - Confusion matrix, ROC curve, PR curve PNGs as artifacts
  - Fitted preprocessor `.joblib` artifact
  - Full pipeline via `mlflow.sklearn.log_model`
- **Model Registry:** Best model promoted to `Staging` in MLflow Registry.

---

## 5. API Serving

- **Framework:** FastAPI 0.111.0 with Uvicorn 0.30.1
- **Endpoint:** `POST /predict` — accepts 13 heart features (Pydantic-validated), returns `{"prediction": 0|1, "probability": float, "model_version": str}`
- **Health:** `GET /health` — returns `{"status": "ok", "model_loaded": bool}`
- **Metrics:** `GET /metrics` — Prometheus text format with `http_requests_total`, `http_request_duration_seconds`, `predictions_total`
- **Model Loading:** Lazy singleton via `MODEL_BUNDLE_PATH` or `MODEL_URI` environment variable; cached across requests
- **Security:** Non-root user in Docker; Pydantic rejects invalid inputs (HTTP 422)

---

## 6. Kubernetes Deployment

- **Chart:** `k8s/helm/heart-disease` (Helm v3, apiVersion: v2)
- **Deployment:** configurable `replicaCount`, liveness `/health` + readiness `/health` probes, resource requests/limits, env from ConfigMap
- **Service:** NodePort (default) exposing port 8000
- **Ingress:** Optional nginx Ingress (disabled by default; enable with `--set ingress.enabled=true`)
- **ConfigMap:** Non-secret env vars (`MODEL_BUNDLE_PATH`, `MODEL_VERSION`, `LOG_LEVEL`)
- **ServiceMonitor:** Optional Prometheus Operator CRD for automatic scrape discovery

**Deploy commands:**
```bash
minikube start --driver=docker
docker build -f docker/Dockerfile.api -t heart-api:latest .
minikube image load heart-api:latest
helm upgrade --install heart-disease k8s/helm/heart-disease --wait
kubectl rollout status deployment/heart-disease
minikube service heart-disease --url
```

---

## 7. Monitoring & Logging

### Prometheus
- Scrape job `heart-api` targeting `api:8000/metrics` at 15s intervals.
- Metrics: `http_requests_total{method,path,status}`, `http_request_duration_seconds{method,path}`, `http_requests_in_progress`, `predictions_total{class_label}`.

### Grafana Dashboard: Heart Disease API Dashboard
- **Request Rate panel:** `rate(http_requests_total[1m])` — live req/s
- **p50 Latency:** `histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))`
- **p95 Latency:** `histogram_quantile(0.95, ...)`
- **Error Rate:** `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])`
- **Predictions by Class:** `predictions_total{class_label}` — monitors class distribution drift

### Structured Logging
- `RequestLoggingMiddleware` emits: `METHOD path → status_code (elapsed ms)` per request.

---

## 8. Testing & CI/CD

### Test Coverage (pytest)
| File | Coverage | Key Tests |
|---|---|---|
| `test_config.py` | config constants, disjoint columns |
| `test_ingestion.py` | download (mocked), load_raw |
| `test_validation.py` | schema, ranges, report |
| `test_preprocessing.py` | clean, split (stratified), save |
| `test_features.py` | ColumnTransformer, scaling, OHE, roundtrip |
| `test_train.py` | factory keys, fit, CV, select_best, MLflow |
| `test_evaluate.py` | metric keys, values, plot files |
| `test_registry.py` | register, promote, URI format |
| `test_predict.py` | save/load bundle, predict_one, batch, probabilities |
| `test_api.py` | schemas, loader, /health, /predict, /metrics, middleware |
| `test_helm.py` | Chart.yaml, values.yaml, templates |
| `test_monitoring.py` | prometheus.yml, dashboard JSON, datasource, compose |
| `test_workflow.py` | CI YAML valid, all jobs present, dependency chain |
| `test_env.py` | pinned requirements, Python 3.11 in env.yml |
| `test_docs.py` | README sections, report marks matrix |

### GitHub Actions CI/CD
- **lint** → **test** → **train** → **docker-build** (→ smoke test) → **deploy** (manual/gated)
- Coverage report uploaded as artifact on every run.
- Docker image tagged with commit SHA and pushed to GHCR on `main` branch.

---

## 9. Challenges & Solutions

| Challenge | Solution |
|---|---|
| MLflow registry `MlflowException` in tests | Used temporary `file:` SQLite URI per-test via `tmp_mlflow_uri` fixture |
| Prometheus metrics singleton conflicts across tests | Isolated `TestClient` per test; used module-level counter reset |
| Docker non-root user permission denied | Created `/app/models` dir during build, set `chown -R appuser` |
| Helm chart rendering without Minikube in CI | `pytest.mark.skipif(shutil.which("helm") is None)` for live CLI tests |
| Data leakage prevention | Preprocessor fitted only on `X_train`; transform applied to `X_test` separately |

---

## 10. Marks Traceability Matrix

| Task | Marks | Deliverables | Status |
|---|---|---|---|
| Task 1 — Data acquisition & EDA | 5 | ingestion.py, validation.py, 01_eda.ipynb (histograms, heatmap, class balance, ≥3 insights) | ✅ |
| Task 1 — Repo scaffold | 5 | Directory tree, config.py, Makefile, README, tooling configs (.flake8, pyproject.toml, env files) | ✅ |
| Task 2 — Feature engineering + modeling | 8 | preprocessing.py (clean+split), features/pipeline.py (ColumnTransformer), train.py (factory+fit) | ✅ |
| Task 3 — Training/eval + MLflow | 5 | Cross-val, metrics+plots, model selection, MLflow runs, Model Registry | ✅ |
| Task 4 — Packaging & reproducibility | 7 | predict.py (bundle save/load, predict_one, batch, CLI), pinned requirements.txt, environment.yml | ✅ |
| Task 5 — API serving | 8 | FastAPI /predict+/health+/metrics, Pydantic schemas, model loader singleton | ✅ |
| Task 6 — Testing & CI/CD | 5 | Pytest suite (all tests), Dockerfile+smoke, GitHub Actions lint→test→train→build→deploy | ✅ |
| Task 7 — Kubernetes deployment | 7 | Helm chart, Deployment/Service/Ingress/ConfigMap, Minikube deploy instructions | ✅ |
| Task 8 — Monitoring & logging | 3 | Prometheus middleware+config, Grafana dashboard+provisioning, docker-compose stack | ✅ |
| Task 9 — Documentation & report | 2 | Full README (all sections), final_report.md with marks matrix | ✅ |
| **Total** | **50** | **End-to-end pipeline (all tasks)** | **✅** |

---

*Report generated for BITS Pilani MLOps Assignment (AIMLCZG523), July 2026.*
