from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException

from ..constants import MODEL_PATH
from ..data_processing import prepare_features
from ..models.persistence import load_model
from .pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk Model API")

_pipeline: Optional[object] = None


@app.on_event("startup")
def _load_pipeline() -> None:
    global _pipeline
    if not MODEL_PATH.exists():
        # Allow startup in environments where training hasn't been run.
        _pipeline = None
        return
    _pipeline = load_model(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model artifact not found. Run training first.")

    raw_df = pd.DataFrame([req.model_dump()])
    X_clean = prepare_features(raw_df)

    proba_survived = float(_pipeline.predict_proba(X_clean)[:, 1][0])
    default_risk_prob = float(1.0 - proba_survived)
    return PredictionResponse(
        proba_survived=proba_survived,
        default_risk_prob=default_risk_prob,
    )

