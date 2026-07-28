from __future__ import annotations

from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..constants import CATEGORICAL_COLS, NUMERIC_COLS


def build_preprocessor(
    numeric_features: Sequence[str] = NUMERIC_COLS,
    categorical_features: Sequence[str] = CATEGORICAL_COLS,
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            # Median is robust for skewed income/risk-like distributions.
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
    )

