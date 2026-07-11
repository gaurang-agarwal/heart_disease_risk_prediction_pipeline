"""Feature engineering pipeline: scaling, encoding, and persistence.

Provides a factory for building a scikit-learn ``ColumnTransformer`` that applies
``StandardScaler`` to numeric features and ``OneHotEncoder`` to categorical features,
along with utilities to inspect output feature names and persist/reload a fitted
preprocessor via *joblib*.

Typical usage::

    from src import config
    from src.features.pipeline import build_preprocessor, save_preprocessor

    pre = build_preprocessor(config.NUMERIC_COLS, config.CATEGORICAL_COLS)
    pre.fit(X_train)
    X_tr = pre.transform(X_train)
    X_te = pre.transform(X_test)          # uses train-fit params → no leakage
    path = save_preprocessor(pre, Path("models/preprocessor.joblib"))
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """Return an *unfitted* ``ColumnTransformer`` for scaling and encoding.

    The transformer applies:

    * **StandardScaler** — zero-mean / unit-variance scaling on *numeric_cols*.
    * **OneHotEncoder** (``handle_unknown='ignore'``, dense output) — binary
      indicator expansion on *categorical_cols*.

    All columns not listed in either argument are dropped (``remainder='drop'``).

    Parameters
    ----------
    numeric_cols:
        Names of continuous/numeric feature columns to scale.
    categorical_cols:
        Names of discrete/categorical feature columns to one-hot encode.

    Returns
    -------
    ColumnTransformer
        An unfitted sklearn ``ColumnTransformer`` ready for ``.fit_transform()``.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return output feature names after the preprocessor has been fitted.

    Delegates to :meth:`sklearn.compose.ColumnTransformer.get_feature_names_out`
    which was stabilised in scikit-learn 1.0.

    Parameters
    ----------
    preprocessor:
        A **fitted** ``ColumnTransformer`` (i.e. ``.fit()`` or
        ``.fit_transform()`` has been called).

    Returns
    -------
    list[str]
        Feature names in the same column order as the array returned by
        ``.transform()``.

    Raises
    ------
    sklearn.exceptions.NotFittedError
        If the preprocessor has not been fitted yet.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception as exc:
        raise NotFittedError(
            "Call fit() or fit_transform() on the preprocessor before "
            "calling get_feature_names()."
        ) from exc


def save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> Path:
    """Persist a fitted preprocessor to disk using joblib serialisation.

    Parent directories of *path* are created automatically.

    Parameters
    ----------
    preprocessor:
        Fitted ``ColumnTransformer`` to serialise.
    path:
        Destination file path.  The ``.joblib`` extension is conventional but
        not enforced.

    Returns
    -------
    Path
        The *resolved* absolute path where the artefact was written.

    Raises
    ------
    IOError
        If the file cannot be written (e.g. permission denied).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    return path.resolve()


def load_preprocessor(path: Path) -> ColumnTransformer:
    """Load a fitted preprocessor from a joblib file.

    Parameters
    ----------
    path:
        Path to the ``.joblib`` file previously written by
        :func:`save_preprocessor`.

    Returns
    -------
    ColumnTransformer
        The deserialised, fitted preprocessor ready for ``.transform()``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist on the filesystem.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor artefact not found: {path}")
    return joblib.load(path)
