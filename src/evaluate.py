"""Evaluate step: score a fitted Pipeline and return plain floats under the agreed metric keys.

Inputs: fitted model (train.py), X_eval/y_eval (validation for gates, test for the audit),
problem_type, and the on-time and big-miss minutes from SETTINGS["business"].
Outputs: mae, rmse, r2, bias, on_time_share, big_miss_share (regression); roc_auc, pr_auc.
Used by: main.py, which checks the quality gates on the validation metrics.
Why: gates are judged on these numbers, so keys, signs and units must be exactly as agreed.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _regression_scores(
    y_true: pd.Series,
    predictions: np.ndarray,
    on_time_minutes: float,
    big_miss_minutes: float,
) -> dict[str, float]:
    """Regression metrics plus the two business shares; error = prediction - actual."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    error = predicted - actual  # > 0: quote too long, < 0: quote too short
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(error)),
        "on_time_share": float(np.mean(np.abs(error) <= on_time_minutes)),
        "big_miss_share": float(np.mean(np.abs(error) > big_miss_minutes)),
    }


def _classification_scores(
    model: Pipeline, x_eval: pd.DataFrame, y_eval: pd.Series
) -> dict[str, float]:
    """Threshold-free classification metrics from the positive-class probability."""
    positive_probability = model.predict_proba(x_eval)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_eval, positive_probability)),
        "pr_auc": float(average_precision_score(y_eval, positive_probability)),
    }


def evaluate_model(
    model: Pipeline,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    problem_type: str,
    on_time_minutes: float = 3,
    big_miss_minutes: float = 10,
) -> dict[str, float]:
    """Score a fitted pipeline; return plain floats keyed by metric name.

    Inputs: model, a fitted Pipeline; X_eval, y_eval, the split to score; problem_type;
    on_time_minutes and big_miss_minutes from SETTINGS["business"] (defaults are fallbacks only).
    Outputs: regression keys mae, rmse, r2, bias, on_time_share, big_miss_share (shares are
    fractions); classification keys roc_auc, pr_auc.
    Why: the quality gates read these keys, so names, signs and units never change.
    """
    logger.info("Evaluating %s model on %s rows", problem_type, len(X_eval))
    if len(X_eval) == 0:
        raise ValueError("X_eval is empty; check the split sizes in SETTINGS['split']")
    if len(X_eval) != len(y_eval):
        raise ValueError(
            f"X_eval and y_eval have different lengths: {len(X_eval)} vs {len(y_eval)}"
        )
    if problem_type == "regression":
        predictions = model.predict(X_eval)
        return _regression_scores(
            y_eval, predictions, on_time_minutes, big_miss_minutes
        )
    if problem_type == "classification":
        return _classification_scores(model, X_eval, y_eval)
    raise ValueError(
        f"Unknown problem_type '{problem_type}'; use classification or regression"
    )
