"""Tests for src/evaluate.py: evaluate_model returns plain floats under the agreed metric keys.

Unit tests check known regression values (business shares included) and classification keys,
and that bad inputs raise; the integration test scores a model fitted by train_model.
"""

import math

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.evaluate import evaluate_model


def _make_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Four trips; a model quoting 20 min errs by +3, -10, +11 and -2 (boundaries on purpose)."""
    X = pd.DataFrame({"straight_km": [3.0, 9.0, 1.0, 4.0]})
    y = pd.Series([17.0, 30.0, 9.0, 22.0], name="duration_min")
    return X, y


def test_unit_regression_metrics_match_hand_computed_values():
    """Errors +3, -10, +11, -2 give exact metrics; every value is a plain Python float."""
    X, y = _make_xy()
    model = DummyRegressor(strategy="constant", constant=20.0).fit(X, y)

    metrics = evaluate_model(
        model, X, y, "regression", on_time_minutes=3, big_miss_minutes=10
    )

    assert set(metrics) == {
        "mae",
        "rmse",
        "r2",
        "bias",
        "on_time_share",
        "big_miss_share",
    }
    assert all(type(value) is float for value in metrics.values())
    assert metrics["mae"] == pytest.approx(6.5)
    assert metrics["bias"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(math.sqrt(58.5))
    assert metrics["r2"] == pytest.approx(1 - 234 / 233)
    assert metrics["on_time_share"] == pytest.approx(0.5)
    assert metrics["big_miss_share"] == pytest.approx(0.25)


def test_unit_classification_keys_and_bad_inputs_raise():
    """Classification returns threshold-free roc_auc and pr_auc only; bad inputs stop the run."""
    X, y = _make_xy()
    labels = pd.Series([0, 1, 0, 1])
    classifier = DummyClassifier(strategy="prior").fit(X, labels)

    assert evaluate_model(classifier, X, labels, "classification") == {
        "roc_auc": 0.5,
        "pr_auc": 0.5,
    }

    model = DummyRegressor(strategy="constant", constant=20.0).fit(X, y)
    with pytest.raises(ValueError, match="empty"):
        evaluate_model(model, X.iloc[0:0], y.iloc[0:0], "regression")
    with pytest.raises(ValueError, match="different lengths"):
        evaluate_model(model, X, y.iloc[:2], "regression")
    with pytest.raises(ValueError, match="Unknown problem_type"):
        evaluate_model(model, X, y, "ranking")


def test_integration_scores_a_model_fitted_by_train_model():
    """A Pipeline from train_model is scored with the agreed keys, sensible values, plain floats."""
    from src.train import train_model
    from tests.test_train import _make_simple_recipe
    from tests.test_train import _make_xy as _make_train_xy

    X, y = _make_train_xy()
    params = {
        "model_type": "hist_gradient_boosting",
        "max_iter": 50,
        "min_samples_leaf": 2,
    }
    model = train_model(
        X, y, _make_simple_recipe(), "regression", params, random_state=42
    )

    metrics = evaluate_model(
        model, X, y, "regression", on_time_minutes=3, big_miss_minutes=10
    )

    assert set(metrics) == {
        "mae",
        "rmse",
        "r2",
        "bias",
        "on_time_share",
        "big_miss_share",
    }
    assert all(type(value) is float for value in metrics.values())
    assert 0.0 <= metrics["mae"] <= metrics["rmse"]
    assert 0.0 <= metrics["big_miss_share"] <= 1.0
    assert 0.0 <= metrics["on_time_share"] <= 1.0
    assert metrics["r2"] > 0.9
