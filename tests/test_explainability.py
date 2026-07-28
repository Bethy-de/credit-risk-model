from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import ModelConfig
from src.constants import LABEL_COL
from src.data_processing import load_training_data, prepare_features, split_train_test
from src.explainability.shap_explain import compute_shap
from src.models.training import build_pipeline, train_model


def test_compute_shap_returns_feature_attributions() -> None:
    X_raw, y = load_training_data(Path("data/raw/train.csv"), label_col=LABEL_COL)
    X = prepare_features(X_raw)

    X_small = X.sample(n=220, random_state=42)
    y_small = y.loc[X_small.index]
    X_train, X_valid, y_train, y_valid = split_train_test(
        X_small, y_small, test_size=0.2, random_seed=42
    )

    model_config = ModelConfig(logistic_params={"solver": "liblinear", "max_iter": 200, "C": 1.0})
    pipeline = build_pipeline(model_config)
    pipeline, _metrics = train_model(pipeline, X_train, y_train, X_valid, y_valid)

    raw_valid = X_raw.loc[X_valid.index]
    shap_values, X_trans, feature_names = compute_shap(
        pipeline,
        raw_valid,
        max_background=5,
        max_sample=8,
        random_seed=42,
    )

    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert isinstance(shap_values, np.ndarray)
    assert shap_values.ndim == 2
    assert shap_values.shape[1] == len(feature_names)
    assert shap_values.shape[0] == X_trans.shape[0]

