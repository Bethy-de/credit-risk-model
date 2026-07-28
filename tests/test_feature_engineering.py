from __future__ import annotations

import pandas as pd

from src.constants import DECK_COL
from src.data_processing import prepare_features


def test_prepare_features_adds_cabin_deck(prepared_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X_prepared, _ = prepared_data
    assert DECK_COL in X_prepared.columns
    # Ensure missing cabin values are converted into "Unknown" deck.
    assert X_prepared[DECK_COL].isna().sum() == 0

