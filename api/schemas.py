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


# --- /monitoring/* ----------------------------------------------------------
# `month` refere-se sempre ao mês de pickup_datetime (relógio do mundo
# simulado, 2020-2021), nunca ao created_at do replay.


class DriftFeature(BaseModel):
    psi: float
    ks: float | None = Field(
        default=None, description="KS binado vs. treino; só para contínuas")
    level: Literal["stable", "watch", "alert"]


class DriftResponse(BaseModel):
    month: str
    n_predictions: int
    thresholds: dict[str, float]
    features: dict[str, DriftFeature]
    n_alert: int = Field(description="nº de features com level == alert")


class PerformanceResponse(BaseModel):
    month: str
    n_predictions: int
    n_with_feedback: int
    mae_min: float | None = Field(
        description="None enquanto nenhum feedback do mês chegou")
    rmse_min: float | None
    median_ae_min: float | None
    bias_min: float | None = Field(
        default=None,
        description="média de (predito - real); positivo = modelo "
                    "superestima a duração (assinatura de concept drift "
                    "com ruas vazias)")
    holdout_reference: dict[str, float] = Field(
        description="métricas do holdout de 2019 (metadata.json), baseline "
                    "para o dashboard alinhar as curvas")


class RetrainMonth(BaseModel):
    month: str
    n_alert: int
    alert_features: list[str]


class RetrainCheckResponse(BaseModel):
    month: str
    status: Literal["green", "yellow", "red"]
    rule: dict[str, float] = Field(
        description="parâmetros vigentes da regra (k, m, psi_alert)")
    months: list[RetrainMonth] = Field(
        description="janela avaliada (só meses com dados), em ordem "
                    "cronológica")


class TimelineMonth(BaseModel):
    """Um mês de produção com os três sinais lado a lado: drift (imediato),
    performance (atrasada) e o parecer da regra de retreino."""

    month: str
    n_predictions: int
    n_with_feedback: int
    features: dict[str, DriftFeature]
    n_alert: int
    alert_features: list[str]
    mae_min: float | None
    rmse_min: float | None
    median_ae_min: float | None
    bias_min: float | None
    retrain_status: Literal["green", "yellow", "red"]


class TimelineResponse(BaseModel):
    """Série completa para o dashboard desenhar de uma vez (uma chamada, sem
    N requests por mês). O que é constante — thresholds, regra, baseline do
    holdout — sai no topo, não repetido em cada mês."""

    months: list[TimelineMonth] = Field(description="ordem cronológica")
    feature_names: list[str] = Field(
        description="features monitoradas, na ordem do schema do modelo")
    thresholds: dict[str, float]
    rule: dict[str, float]
    holdout_reference: dict[str, float]