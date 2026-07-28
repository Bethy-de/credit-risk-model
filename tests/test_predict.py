from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ModelConfig
from src.constants import DEFAULT_RISK_PROB_COL, LABEL_COL
from src.data_processing import load_training_data, prepare_features, split_train_test
from src.models.training import build_pipeline, train_model
from src.predict import predict_default_risk


def test_predict_default_risk_returns_probabilities() -> None:
    X_raw, y = load_training_data(Path("data/raw/train.csv"), label_col=LABEL_COL)
    X = prepare_features(X_raw)
    X_small = X.sample(n=240, random_state=42)
    y_small = y.loc[X_small.index]
    X_train, X_valid, y_train, y_valid = split_train_test(
        X_small, y_small, test_size=0.2, random_seed=42
    )

    model_config = ModelConfig(logistic_params={"solver": "liblinear", "max_iter": 200, "C": 1.0})
    pipeline = build_pipeline(model_config)
    pipeline, _metrics = train_model(pipeline, X_train, y_train, X_valid, y_valid)

    raw_subset = pd.DataFrame(
        [
            {
                "Pclass": 3,
                "Sex": "male",
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked": "S",
                "Cabin": None,
            }
        ]
    )
    preds = predict_default_risk(pipeline, raw_subset)
    assert DEFAULT_RISK_PROB_COL in preds.columns
    prob = float(preds.loc[0, DEFAULT_RISK_PROB_COL])
    assert 0.0 <= prob <= 1.0

