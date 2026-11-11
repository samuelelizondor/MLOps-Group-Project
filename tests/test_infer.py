"""Tests for src/infer.py: run_inference predicts every row and keeps the input index.

Unit tests check the prediction column, index, unchanged input and the optional floor, plus
probabilities and bad inputs; the integration test scores a model saved and reloaded with utils.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.infer import run_inference


def _make_bookings() -> pd.DataFrame:
    """Three bookings with index 10, 20, 30, so a reset index cannot pass the checks by accident."""
    return pd.DataFrame({"straight_km": [1.0, 5.0, 12.0]}, index=[10, 20, 30])


def test_unit_regression_keeps_index_and_input_and_applies_floor():
    """One prediction per booking, same index, input unchanged; min_prediction lifts low quotes."""
    bookings = _make_bookings()
    before = bookings.copy()
    tiny = DummyRegressor(strategy="constant", constant=0.4).fit(bookings, [0.4] * 3)
    normal = DummyRegressor(strategy="constant", constant=5.0).fit(bookings, [5.0] * 3)

    raw = run_inference(tiny, bookings)
    floored = run_inference(tiny, bookings, min_prediction=1.0)
    untouched = run_inference(normal, bookings, min_prediction=1.0)

    assert list(raw.columns) == ["prediction"]
    assert list(raw.index) == [10, 20, 30]
    assert raw["prediction"].tolist() == pytest.approx([0.4, 0.4, 0.4])
    assert floored["prediction"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert untouched["prediction"].tolist() == pytest.approx([5.0, 5.0, 5.0])
    pd.testing.assert_frame_equal(bookings, before)


def test_unit_probabilities_and_bad_inputs():
    """include_proba adds a probability column for classifiers; empty input or no proba raise."""
    bookings = _make_bookings()
    classifier = DummyClassifier(strategy="prior").fit(bookings, [0, 1, 1])

    scored = run_inference(classifier, bookings, include_proba=True)

    assert list(scored.columns) == ["prediction", "probability"]
    assert list(scored.index) == [10, 20, 30]
    assert scored["probability"].tolist() == pytest.approx([2 / 3] * 3)

    regressor = DummyRegressor(strategy="constant", constant=5.0).fit(
        bookings, [5.0] * 3
    )
    with pytest.raises(ValueError, match="No rows"):
        run_inference(regressor, bookings.iloc[0:0])
    with pytest.raises(ValueError, match="predict_proba"):
        run_inference(regressor, bookings, include_proba=True)
    with pytest.raises(ValueError, match="regression quotes only"):
        run_inference(classifier, bookings, min_prediction=1.0)


def test_integration_saved_and_reloaded_model_scores_every_booking(tmp_path):
    """A model saved with utils and reloaded quotes every booking, in order, with the floor."""
    from src.train import train_model
    from src.utils import load_model, save_model
    from tests.test_train import _make_simple_recipe, _make_xy

    X, y = _make_xy()
    params = {"min_samples_leaf": 2}
    model = train_model(
        X, y, _make_simple_recipe(), "regression", params, random_state=42
    )
    save_model(model, tmp_path / "model.joblib")
    reloaded = load_model(tmp_path / "model.joblib")
    bookings = X.iloc[[5, 1, 30]]  # reordered rows, non-default index

    scored = run_inference(reloaded, bookings, min_prediction=1.0)

    assert list(scored.index) == [5, 1, 30]
    expected = np.maximum(model.predict(bookings), 1.0)
    np.testing.assert_allclose(scored["prediction"], expected)
