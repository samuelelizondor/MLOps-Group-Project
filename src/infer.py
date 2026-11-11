"""Infer step: quote every booking with the fitted Pipeline, keeping the input index.

Inputs: the fitted Pipeline (reloaded in main.py), cleaned and validated feature columns,
include_proba (classification) and min_prediction (SETTINGS["cleaning"]["min_minutes"]).
Outputs: a DataFrame with the input's index: prediction (+ probability when include_proba).
Used by: main.py (batch) and, later, the API; main.py puts trip_id back next to each quote.
Why: no booking is dropped or reordered, so every quote stays traceable to its trip.
"""

import logging

import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def run_inference(
    model: Pipeline,
    X_infer: pd.DataFrame,
    include_proba: bool = False,
    min_prediction: float | None = None,
) -> pd.DataFrame:
    """Predict for every row; same index as X_infer, with "prediction" (+ "probability").

    Inputs: model, a fitted Pipeline; X_infer, the feature columns of the rows to quote;
    include_proba, add the positive-class probability (classifiers only); min_prediction,
    a floor for regression quotes (None keeps raw predictions).
    Outputs: a new DataFrame indexed like X_infer; X_infer itself is never changed.
    Why: one serving function for batch and API, so every quote gets the same floor.
    """
    logger.info("Scoring %s rows", len(X_infer))
    if X_infer.empty:
        raise ValueError("No rows to score; check the inference data and cleaning")
    if include_proba and not hasattr(model, "predict_proba"):
        raise ValueError("include_proba=True needs a classifier with predict_proba")
    if min_prediction is not None and hasattr(model, "predict_proba"):
        raise ValueError(
            "min_prediction applies to regression quotes only, not class labels"
        )
    result = pd.DataFrame(index=X_infer.index)
    result["prediction"] = model.predict(X_infer)
    if min_prediction is not None:
        result["prediction"] = result["prediction"].clip(lower=min_prediction)
    if include_proba:
        result["probability"] = model.predict_proba(X_infer)[:, 1]
    return result
