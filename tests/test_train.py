"""Tests for src/train.py: train_model fits recipe and model together, the only .fit() in src/.

Unit tests check a fitted Pipeline with one prediction per row, and that bad inputs raise;
the integration test chains get_feature_preprocessor, train_model and predict on tiny data.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.train import train_model


def _make_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Forty trips: trip time grows 3 min per km, and Queens pickups take 5 minutes longer."""
    straight_km = np.linspace(0.5, 20.0, 40)
    borough = np.array(["Manhattan", "Queens"] * 20)
    X = pd.DataFrame({"straight_km": straight_km, "pu_borough": borough})
    queens_extra = np.where(borough == "Queens", 5.0, 0.0)
    y = pd.Series(4.0 + 3.0 * straight_km + queens_extra, name="duration_min")
    return X, y


def _make_simple_recipe() -> ColumnTransformer:
    """A minimal unfitted recipe built here, so changes in features.py cannot break these tests."""
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), ["straight_km"]),
            ("categorical", one_hot, ["pu_borough"]),
        ]
    )


def test_unit_regression_returns_fitted_pipeline_and_keeps_inputs():
    """Fitted Pipeline, one prediction per row; params reach the model; inputs stay unchanged."""
    X, y = _make_xy()
    recipe = _make_simple_recipe()
    params = {
        "model_type": "hist_gradient_boosting",
        "loss": "quantile",
        "quantile": 0.7,
        "max_iter": 50,
        "min_samples_leaf": 2,
    }
    params_before = dict(params)

    pipeline = train_model(X, y, recipe, "regression", params, random_state=7)

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocess", "model"]
    check_is_fitted(pipeline)
    predictions = pipeline.predict(X)
    assert predictions.shape == (40,)
    assert np.corrcoef(predictions, y)[0, 1] > 0.9
    model = pipeline.named_steps["model"]
    assert (model.quantile, model.random_state) == (0.7, 7)
    assert params == params_before
    with pytest.raises(NotFittedError):
        check_is_fitted(recipe)


def test_unit_classification_works_and_bad_inputs_raise():
    """Classification fits with probabilities; unknown types and length mismatches raise."""
    X, y = _make_xy()
    long_trip = (y > y.median()).astype(int)
    recipe = _make_simple_recipe()

    classifier = train_model(
        X, long_trip, recipe, "classification", {"model_type": "logistic_regression"}
    )
    assert classifier.predict_proba(X).shape == (40, 2)

    default = train_model(X, y, recipe, "regression")
    assert (
        type(default.named_steps["model"]).__name__ == "HistGradientBoostingRegressor"
    )

    with pytest.raises(ValueError, match="Unknown problem_type"):
        train_model(X, y, recipe, "ranking")
    with pytest.raises(ValueError, match="Unknown model_type"):
        train_model(X, y, recipe, "regression", {"model_type": "random_forest"})
    with pytest.raises(ValueError, match="different lengths"):
        train_model(X, y.iloc[:10], recipe, "regression")


def test_unit_model_without_a_seed_trains_without_one(monkeypatch):
    """A registered model that takes no random_state still trains; the seed is passed only if accepted."""
    from sklearn.linear_model import LinearRegression

    import src.train as train_module

    monkeypatch.setitem(
        train_module._ESTIMATORS, ("regression", "linear"), LinearRegression
    )
    X, y = _make_xy()

    pipeline = train_model(
        X,
        y,
        _make_simple_recipe(),
        "regression",
        {"model_type": "linear"},
        random_state=7,
    )

    assert "random_state" not in pipeline.named_steps["model"].get_params()
    assert pipeline.predict(X).shape == (40,)


def test_integration_feature_recipe_then_train_then_predict_survives_reload(tmp_path):
    """The real recipe trains with the chosen model; a saved and reloaded Pipeline predicts the same."""
    from src.features import get_feature_preprocessor
    from src.utils import load_model, save_model
    from tests.test_features import _make_feature_df

    df = _make_feature_df()
    X, y = df.drop(columns=["duration_min"]), df["duration_min"]
    recipe = get_feature_preprocessor(
        ["straight_km", "passenger_count"], ["pu_borough"], ["zone_pair"]
    )
    params = {
        "model_type": "hist_gradient_boosting",
        "loss": "quantile",
        "quantile": 0.58,
        "max_iter": 20,
    }

    pipeline = train_model(X, y, recipe, "regression", params, random_state=42)
    save_model(pipeline, tmp_path / "model.joblib")
    reloaded = load_model(tmp_path / "model.joblib")

    predictions = pipeline.predict(X)
    assert predictions.shape == (20,)
    assert not np.isnan(predictions).any()
    np.testing.assert_allclose(reloaded.predict(X), predictions)
