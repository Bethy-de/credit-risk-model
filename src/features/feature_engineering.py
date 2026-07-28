from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..constants import CABIN_COL, DECK_COL, DROP_COLS, ID_COLS


def ensure_required_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> pd.DataFrame:
    """
    Add missing columns as NA so preprocessing never crashes on incomplete inputs.
    """

    out = df.copy()
    for col in required_cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def add_cabin_deck(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if CABIN_COL not in out.columns:
        out[CABIN_COL] = pd.NA
    # Deck is the first character in cabin, e.g. "C85" -> "C"
    out[DECK_COL] = out[CABIN_COL].astype("string").str.extract(r"^([A-Za-z])", expand=False)
    out[DECK_COL] = out[DECK_COL].fillna("Unknown")
    return out


def clean_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw Titanic-like rows into a stable tabular schema for ML.
    """

    out = df.copy()
    out = add_cabin_deck(out)
    # Drop high-cardinality fields not used for this model.
    for col in DROP_COLS:
        if col in out.columns:
            out = out.drop(columns=[col])
    # IDs are not predictive for this demo model.
    for col in ID_COLS:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out

