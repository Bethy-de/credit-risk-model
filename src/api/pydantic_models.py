from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str
    Age: Optional[float] = None
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: str
    Cabin: Optional[str] = None


class PredictionResponse(BaseModel):
    proba_survived: float
    default_risk_prob: float

