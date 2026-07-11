"""MLflow Model Registry: register, promote, and retrieve production models.

Provides three functions that wrap the MLflow registry API:

* :func:`register_model` — register a logged model from a run into the registry.
* :func:`promote_to_stage` — transition a model version to Staging/Production.
* :func:`get_production_model_uri` — return the ``models:/`` URI for Production.

Typical usage::

    from src.models.registry import register_model, promote_to_stage

    mv = register_model(run_id, model_name="heart-disease-clf")
    promote_to_stage("heart-disease-clf", version=int(mv.version), stage="Staging")
"""

from __future__ import annotations

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient


def register_model(
    run_id: str,
    model_name: str = "heart-disease-clf",
) -> ModelVersion:
    """Register the ``model`` artifact from *run_id* in the MLflow Model Registry.

    Creates the registered model entry if it does not already exist, then
    creates a new version pointing to the run's ``model`` artifact.

    Parameters
    ----------
    run_id:
        The MLflow run ID whose ``model`` artifact should be registered.
    model_name:
        Target name in the MLflow Model Registry (created if absent).

    Returns
    -------
    ModelVersion
        The newly created :class:`~mlflow.entities.model_registry.ModelVersion`
        object, including its integer ``version`` and ``status``.

    Raises
    ------
    mlflow.exceptions.MlflowException
        If the run cannot be found or registration encounters a tracking error.
    """
    model_uri = f"runs:/{run_id}/model"
    mv: ModelVersion = mlflow.register_model(model_uri=model_uri, name=model_name)
    return mv


def promote_to_stage(
    model_name: str,
    version: int,
    stage: str = "Staging",
) -> None:
    """Transition a registered model version to the target stage.

    Parameters
    ----------
    model_name:
        Name of the registered model in the MLflow Model Registry.
    version:
        Integer version number to transition.
    stage:
        Destination lifecycle stage.  Accepted values: ``"Staging"``,
        ``"Production"``, ``"Archived"``, ``"None"``.

    Raises
    ------
    mlflow.exceptions.MlflowException
        If the model or version does not exist, or the transition is rejected.
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=False,
    )


def get_production_model_uri(model_name: str) -> str:
    """Return the ``models:/`` URI pointing to the Production stage of *model_name*.

    The URI can be passed directly to :func:`mlflow.sklearn.load_model` or
    :func:`mlflow.pyfunc.load_model` to load the production model.

    Parameters
    ----------
    model_name:
        Name of the registered model in the MLflow Model Registry.

    Returns
    -------
    str
        URI of the form ``"models:/<model_name>/Production"``.

    Raises
    ------
    mlflow.exceptions.MlflowException
        If the tracking server cannot be reached (URI is constructed locally
        so this is only raised when callers subsequently *use* the URI to load).
    """
    return f"models:/{model_name}/Production"
