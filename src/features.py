"""Features step: the unfitted preprocessing recipe for the three feature groups.

Inputs: SETTINGS["features"] column lists (numeric, categorical, target_encoded), seed, folds.
Outputs: an unfitted ColumnTransformer; train.py fits it on training rows only.
Used by: main.py builds it, train.py fits it, the saved Pipeline reuses it at inference.
Why: medians, categories and typical trip times are learned after the split, never before.
"""

import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

logger = logging.getLogger(__name__)


def _check_column_groups(groups: dict[str, list[str]]) -> None:
    """Stop the run if a group is not a list, is empty, or shares a column with another group."""
    seen: set[str] = set()
    for name, columns in groups.items():
        if not isinstance(columns, list):
            raise ValueError(
                f"{name} must be a list of column names, got {type(columns).__name__}"
            )
        if not columns:
            raise ValueError(f"{name} is empty; check SETTINGS['features']")
        repeated = seen.intersection(columns)
        if repeated:
            raise ValueError(f"Columns in more than one group: {sorted(repeated)}")
        seen.update(columns)


def _target_encoded_pipeline(random_state: int, target_encoder_folds: int) -> Pipeline:
    """TargetEncoder with seeded cross-fitting (a row never sees its own label), then scaling."""
    folds = KFold(
        n_splits=target_encoder_folds, shuffle=True, random_state=random_state
    )
    return Pipeline(
        [
            ("encode", TargetEncoder(target_type="continuous", cv=folds)),
            ("scale", StandardScaler()),
        ]
    )


def get_feature_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    target_encoded_columns: list[str],
    random_state: int = 42,
    target_encoder_folds: int = 5,
) -> ColumnTransformer:
    """Return the unfitted preprocessing recipe for the three feature groups.

    Inputs: column lists per group (SETTINGS["features"]); random_state and target_encoder_folds
    for the encoder's cross-fitting (from SETTINGS; defaults are fallbacks only).
    Outputs: an unfitted ColumnTransformer with numeric, categorical and zones steps.
    Why: one recipe for training and inference; train.py fits it on the training split only.
    """
    logger.info("Building the unfitted feature recipe (numeric, categorical, zones)")
    _check_column_groups(
        {
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "target_encoded_columns": target_encoded_columns,
        }
    )
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    zones = _target_encoded_pipeline(random_state, target_encoder_folds)
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
            ("zones", zones, target_encoded_columns),
        ]
    )


# Recommendations for later:
# - Add public holidays and weather *forecasts* (actual weather is not known at booking time).
# - Replace straight_km with a routing-engine distance or time estimate, if the app has one.
# - Tune TargetEncoder smoothing for rare route x time keys (most pair_band values are rare).
