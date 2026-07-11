"""Pydantic request / response schemas for the Heart Disease prediction API.

Defines three models:

* :class:`HeartFeatures` — validates and types all 13 input features.
* :class:`PredictionResponse` — output contract for ``POST /predict``.
* :class:`HealthResponse` — output contract for ``GET /health``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HeartFeatures(BaseModel):
    """Validated input schema for a single patient's heart-disease risk features.

    All 13 UCI Heart Disease features are required.  Field validators enforce
    physiological plausibility; Pydantic raises ``ValidationError`` (→ HTTP 422)
    for out-of-range inputs.
    """

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0–3)")
    trestbps: int = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0/1)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0–2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (0/1)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels coloured by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia type (0–3)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for ``POST /predict``.

    Attributes
    ----------
    prediction:
        Binary class label — ``0`` (no heart disease) or ``1`` (heart disease).
    probability:
        Model's estimated probability for the positive class (``[0.0, 1.0]``).
    model_version:
        Version string of the loaded model, e.g. ``"1"`` or ``"latest"``.
    """

    prediction: int
    probability: float
    model_version: str


class HealthResponse(BaseModel):
    """Response schema for ``GET /health``.

    Attributes
    ----------
    status:
        Human-readable service status, typically ``"ok"``.
    model_loaded:
        ``True`` if the prediction model has been successfully loaded.
    """

    status: str
    model_loaded: bool
