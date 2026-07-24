"""Modelos Pydantic dos contratos da API.

Os nomes dos campos seguem o dataset da TLC (PULocationID, trip_type…) para
o replay da fase 3 mapear parquet → request sem camada de tradução.

Regra de leakage vale no contrato: o request de /predict só aceita o que
existe no momento do embarque. A duração real só entra via /feedback.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    pickup_datetime: datetime = Field(
        description="Horário do embarque (naive, horário local de NY, "
                    "como no dataset da TLC)")
    PULocationID: int = Field(ge=1, le=265)
    DOLocationID: int = Field(ge=1, le=265)
    trip_type: Literal[1, 2] | None = Field(
        default=None,
        description="1=street-hail, 2=despacho; None vira a categoria 0 "
                    "(bloco MNAR, mesma convenção do treino)")


class PredictResponse(BaseModel):
    prediction_id: str
    predicted_duration_min: float
    model_version: str


class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_duration_min: float = Field(gt=0)


class FeedbackResponse(BaseModel):
    prediction_id: str
    status: Literal["ok"]