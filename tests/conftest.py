from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.constants import LABEL_COL
from src.data_processing import load_training_data, prepare_features, split_train_test


@pytest.fixture(scope="session")
def raw_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_training_data(Path("data/raw/train.csv"), label_col=LABEL_COL)
    return X, y


@pytest.fixture(scope="session")
def prepared_data(raw_data: tuple[pd.DataFrame, pd.Series]) -> tuple[pd.DataFrame, pd.Series]:
    X, y = raw_data
    X_prepared = prepare_features(X)
    return X_prepared, y


def small_train_valid_split(
    raw_X: pd.DataFrame,
    raw_y: pd.Series,
    *,
    n_samples: int = 320,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Take a deterministic subset for fast unit tests.
    X_small = raw_X.sample(n=min(n_samples, len(raw_X)), random_state=random_seed)
    y_small = raw_y.loc[X_small.index]
    X_train, X_valid, y_train, y_valid = split_train_test(
        X_small,
        y_small,
        test_size=0.2,
        random_seed=random_seed,
    )
    return X_train, X_valid, y_train, y_valid

