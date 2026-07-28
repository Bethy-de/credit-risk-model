from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..config import ModelConfig
from ..features.preprocessing import build_preprocessor


def build_pipeline(model_config: ModelConfig) -> Pipeline:
    preprocessor = build_preprocessor()
    if model_config.model_type != "logistic":
        raise ValueError(f"Unsupported model_type: {model_config.model_type!r}")

    # Deterministic + stable baseline for risk scoring / interpretability.
    model = LogisticRegression(random_state=model_config.random_seed, **model_config.logistic_params)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_model(
    pipeline: Pipeline,
    X_train,
    y_train,
    X_valid,
    y_valid,
) -> Tuple[Pipeline, Dict[str, Any]]:
    pipeline = pipeline  # explicit for readability
    pipeline.fit(X_train, y_train)

    # Reliability-focused metrics
    proba_valid = pipeline.predict_proba(X_valid)[:, 1]
    pred_valid = (proba_valid >= 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_valid, proba_valid)),
        "accuracy": float(accuracy_score(y_valid, pred_valid)),
        "y_valid_pos_rate": float(np.mean(y_valid)),
        "threshold": 0.5,
    }
    return pipeline, metrics


def get_pipeline_params(model_config: ModelConfig) -> Dict[str, Any]:
    return {"model_config": asdict(model_config)}

