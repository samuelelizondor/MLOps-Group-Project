"""Train step: fit the preprocessing recipe and the model together, on the training split only.

Inputs: X_train and y_train (split in main.py), the unfitted recipe (features.py), problem_type,
model_params (SETTINGS["model"], including model_type) and random_state.
Outputs: a fitted Pipeline (preprocess + model), ready to evaluate, save and serve.
Used by: main.py; evaluate.py, utils.save_model and infer.py use what it returns.
Why: the only .fit() in src/, so nothing else learns from validation or test rows.
"""

import logging

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Supported estimators per (problem_type, model_type); model_params["model_type"] picks one.
_ESTIMATORS = {
    ("regression", "ridge"): Ridge,
    ("regression", "hist_gradient_boosting"): HistGradientBoostingRegressor,
    ("classification", "logistic_regression"): LogisticRegression,
    ("classification", "hist_gradient_boosting"): HistGradientBoostingClassifier,
}


def _build_estimator(problem_type: str, model_params: dict, random_state: int):
    """Create the unfitted estimator for model_params["model_type"], seeded when it takes a seed."""
    if problem_type not in ("classification", "regression"):
        raise ValueError(
            f"Unknown problem_type '{problem_type}'; use classification or regression"
        )
    params = dict(model_params)  # copy: SETTINGS["model"] is never changed
    # model_type falls back to the default because the contract allows no params
    model_type = params.pop("model_type", "hist_gradient_boosting")
    estimator_class = _ESTIMATORS.get((problem_type, model_type))
    if estimator_class is None:
        raise ValueError(f"Unknown model_type '{model_type}' for {problem_type}")
    if "random_state" in estimator_class().get_params():
        params["random_state"] = random_state
    return estimator_class(**params)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    problem_type: str,
    model_params: dict | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Fit the preprocessing recipe and the model together on the training split.

    Inputs: X_train, y_train; preprocessor, the unfitted recipe (cloned, so the caller's stays
    unfitted); problem_type; model_params from SETTINGS["model"] (copied, never changed); seed.
    Outputs: a fitted Pipeline with steps "preprocess" and "model".
    Why: the only .fit() in src/; fitting recipe and model together keeps inference identical.
    """
    logger.info("Training %s model on %s rows", problem_type, len(X_train))
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train have different lengths: {len(X_train)} vs {len(y_train)}"
        )
    model = _build_estimator(problem_type, model_params or {}, random_state)
    pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", model)])
    pipeline.fit(X_train, y_train)
    return pipeline
