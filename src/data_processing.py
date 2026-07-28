from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import LABEL_COL, RANDOM_SEED
from .features.feature_engineering import clean_raw_features


def load_training_data(train_csv_path: Path, label_col: str = LABEL_COL) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(train_csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} not found in dataset columns: {list(df.columns)}")

    y = df[label_col].astype("int64")
    X = df.drop(columns=[label_col])
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_seed: int = RANDOM_SEED,
):
    return train_test_split(X, y, test_size=test_size, random_state=random_seed, stratify=y)


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw rows into tabular features aligned with preprocessing.
    """

    return clean_raw_features(X)

