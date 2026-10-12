
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class Transaction(BaseModel):
    """v1-v28 are the dataset's anonymized (PCA) features."""

    model_config = ConfigDict(extra="forbid")

    # FiniteFloat
    v1: FiniteFloat
    v2: FiniteFloat
    v3: FiniteFloat
    v4: FiniteFloat
    v5: FiniteFloat
    v6: FiniteFloat
    v7: FiniteFloat
    v8: FiniteFloat
    v9: FiniteFloat
    v10: FiniteFloat
    v11: FiniteFloat
    v12: FiniteFloat
    v13: FiniteFloat
    v14: FiniteFloat
    v15: FiniteFloat
    v16: FiniteFloat
    v17: FiniteFloat
    v18: FiniteFloat
    v19: FiniteFloat
    v20: FiniteFloat
    v21: FiniteFloat
    v22: FiniteFloat
    v23: FiniteFloat
    v24: FiniteFloat
    v25: FiniteFloat
    v26: FiniteFloat
    v27: FiniteFloat
    v28: FiniteFloat
    amount: FiniteFloat = Field(ge=0, description="Transaction amount; cannot be negative")


class Prediction(BaseModel):
    prediction_id: int  # row id in the predictions table
    # A ranking score
    fraud_score: float = Field(ge=0, le=1)
    decision: Literal["APPROVE", "FLAG"]
    model_version: str


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
    threshold: float
