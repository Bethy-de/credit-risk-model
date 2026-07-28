from __future__ import annotations

from pathlib import Path

from src.config import ModelConfig
from src.constants import LABEL_COL
from src.data_processing import load_training_data, prepare_features, split_train_test
from src.models.training import build_pipeline, train_model


def test_train_model_metrics_are_reasonable() -> None:
    X_raw, y = load_training_data(Path("data/raw/train.csv"), label_col=LABEL_COL)
    X = prepare_features(X_raw)

    # Deterministic subset.
    X_small = X.sample(n=280, random_state=42)
    y_small = y.loc[X_small.index]
    X_train, X_valid, y_train, y_valid = split_train_test(
        X_small, y_small, test_size=0.2, random_seed=42
    )

    model_config = ModelConfig(
        logistic_params={"solver": "liblinear", "max_iter": 200, "C": 1.0}
    )
    pipeline = build_pipeline(model_config)
    pipeline, metrics = train_model(pipeline, X_train, y_train, X_valid, y_valid)

    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

