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


Status = Literal["green", "yellow", "red"]


class MonthVerdict(BaseModel):
    """O parecer de retreino de um mês: dois gatilhos e o pior dos dois.

    Os dois status são BRUTOS — o que os números dizem, sem desconto. O
    reconhecimento (o botão do dashboard) não muda o status: ele aparece ao
    lado, como `*_acknowledged_month`, e o dashboard é que desenha o alarme
    apagado. Assim o sinal continua auditável depois de alguém dizer "eu
    vi".

    O reconhecimento vale POR GATILHO, não sobre o parecer geral: na série
    real o drift esfria antes da performance, e um reconhecimento aplicado
    ao geral nunca expiraria.
    """

    status: Status = Field(description="o mais severo entre os dois gatilhos")
    drift_status: Status = Field(
        description="K features com PSI > alerta por M meses — alerta "
                    "precoce, não depende de ground truth")
    drift_acknowledged_month: str | None = Field(
        default=None,
        description="mês do reconhecimento que silencia este alarme; None "
                    "se não há reconhecimento vigente")
    performance_status: Status = Field(
        description="MAE acima do holdout por uma margem, por M meses — "
                    "confirmação atrasada, só existe com feedback")
    performance_acknowledged_month: str | None = None


class RetrainMonth(BaseModel):
    """Um mês da janela avaliada, com os insumos dos dois gatilhos."""

    month: str
    n_alert: int
    alert_features: list[str]
    mae_min: float | None


class RetrainCheckResponse(BaseModel):
    month: str
    verdict: MonthVerdict
    mae_threshold_min: float = Field(
        description="MAE a partir do qual o mês conta como degradado "
                    "(holdout × (1 + tolerância))")
    rule: dict[str, float] = Field(
        description="parâmetros vigentes (k, m, psi_alert, perf_tolerance, "
                    "perf_m, perf_min_feedback)")
    months: list[RetrainMonth] = Field(
        description="janela avaliada (só meses com dados), em ordem "
                    "cronológica")


class RetrainTriggerRequest(BaseModel):
    """Corpo do botão "disparar retreino" do dashboard."""

    month: str = Field(
        pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
        description="mês do mundo simulado cujo alarme está sendo "
                    "reconhecido")
    note: str | None = Field(
        default=None, max_length=500,
        description="justificativa livre de quem clicou")


class RetrainTriggerResponse(BaseModel):
    """O que foi registrado. Nenhum modelo é treinado: retreinar está fora
    do escopo deste projeto, que é de monitoramento. O evento existe para o
    dashboard distinguir "estourou" de "estourou e ninguém atendeu"."""

    id: int
    month: str
    created_at: str
    triggered_by: str = Field(
        description="qual gatilho estava aceso quando o botão foi clicado: "
                    "drift, performance, both ou none")
    retrained: Literal[False] = Field(
        default=False,
        description="sempre False — o pipeline de retreino é mock por "
                    "escopo, e o contrato diz isso em voz alta")


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
    verdict: MonthVerdict


class TimelineResponse(BaseModel):
    """Série completa para o dashboard desenhar de uma vez (uma chamada, sem
    N requests por mês). O que é constante — thresholds, regra, baseline do
    holdout — sai no topo, não repetido em cada mês."""

    months: list[TimelineMonth] = Field(description="ordem cronológica")
    feature_names: list[str] = Field(
        description="features monitoradas, na ordem do schema do modelo")
    thresholds: dict[str, float]
    rule: dict[str, float]
    mae_threshold_min: float = Field(
        description="limiar do gatilho de performance, para o dashboard "
                    "desenhar a linha sem recalcular a regra")
    holdout_reference: dict[str, float]
    retrain_events: list[dict] = Field(
        default_factory=list,
        description="reconhecimentos registrados, mais recente primeiro")