"""Tests for src/features.py: get_feature_preprocessor returns an unfitted ColumnTransformer.

Unit tests check the recipe is unfitted with the agreed groups, and that bad column lists raise;
the integration test fits it on tiny data: no missing values, expected number of output columns.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.features import get_feature_preprocessor


def _make_feature_df() -> pd.DataFrame:
    """Twenty trips: two numeric columns, one categorical, one route key, a few gaps, the target."""
    rows = 20
    straight_km = np.linspace(0.5, 10.0, rows)
    straight_km[[3, 11]] = np.nan
    passenger_count = np.ones(rows)
    passenger_count[7] = np.nan
    borough = np.array(["Manhattan", "Queens"] * 10, dtype=object)
    borough[5] = np.nan
    data = {
        "straight_km": straight_km,
        "passenger_count": passenger_count,
        "pu_borough": borough,
        "zone_pair": ["79_119", "100_234", "132_79", "164_87"] * 5,
        "duration_min": 5.0 + 3.0 * np.linspace(0.5, 10.0, rows),
    }
    return pd.DataFrame(data)


def test_unit_returns_unfitted_recipe_with_three_groups():
    """An unfitted ColumnTransformer with the agreed groups; seed and folds reach the encoder."""
    preprocessor = get_feature_preprocessor(
        ["straight_km", "passenger_count"],
        ["pu_borough"],
        ["zone_pair"],
        random_state=7,
        target_encoder_folds=3,
    )

    assert isinstance(preprocessor, ColumnTransformer)
    with pytest.raises(NotFittedError):
        check_is_fitted(preprocessor)
    groups = {name: columns for name, _, columns in preprocessor.transformers}
    assert groups == {
        "numeric": ["straight_km", "passenger_count"],
        "categorical": ["pu_borough"],
        "zones": ["zone_pair"],
    }
    folds = preprocessor.transformers[2][1].named_steps["encode"].cv
    assert (folds.n_splits, folds.random_state) == (3, 7)


def test_unit_bad_column_lists_raise():
    """An empty group, a column in two groups, or a string instead of a list stops the run."""
    with pytest.raises(ValueError, match="empty"):
        get_feature_preprocessor([], ["pu_borough"], ["zone_pair"])
    with pytest.raises(ValueError, match="more than one group"):
        get_feature_preprocessor(["straight_km"], ["pu_borough"], ["pu_borough"])
    with pytest.raises(ValueError, match="list of column names"):
        get_feature_preprocessor("straight_km", ["pu_borough"], ["zone_pair"])


def test_integration_fit_transform_fills_gaps_and_has_expected_width():
    """Fitted on tiny data, the recipe fills every gap and returns 2 + 2 + 1 = 5 columns."""
    df = _make_feature_df()
    preprocessor = get_feature_preprocessor(
        ["straight_km", "passenger_count"], ["pu_borough"], ["zone_pair"]
    )

    transformed = preprocessor.fit_transform(
        df.drop(columns=["duration_min"]), df["duration_min"]
    )

    assert transformed.shape == (20, 5)
    assert not np.isnan(transformed).any()
