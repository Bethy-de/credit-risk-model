from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .constants import ARTIFACTS_DIR, DEFAULT_RISK_PROB_COL
from .data_processing import prepare_features
from .models.persistence import load_model


def predict_default_risk(pipeline, raw_X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert model output into a finance-style "default risk" probability.

    This project uses the Titanic-style label where `Survived=1` is a "good" outcome.
    We define default-risk as `1 - P(Survived=1)`.
    """

    X_clean = prepare_features(raw_X)
    proba_survived = pipeline.predict_proba(X_clean)[:, 1]
    default_risk_prob = 1.0 - proba_survived

    out = raw_X.copy()
    out[DEFAULT_RISK_PROB_COL] = default_risk_prob
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run inference using the trained model.")
    parser.add_argument("--model-path", type=Path, default=ARTIFACTS_DIR / "model.joblib")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=ARTIFACTS_DIR / "predictions.csv")
    args = parser.parse_args(argv)

    pipeline = load_model(args.model_path)
    raw = pd.read_csv(args.input_csv)
    preds = predict_default_risk(pipeline, raw)
    preds.to_csv(args.output_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

