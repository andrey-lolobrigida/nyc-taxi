"""GET /monitoring/* — drift, performance e gatilho de retreino (fase 3).

O "setor de auditoria" da decisão #2: pode ser lento, nunca pode travar o
/predict. Toda conta roda sobre o log de predições no SQLite (leitura
concorrente com o replay graças ao WAL) e sobre os bins CONGELADOS do
metadata.json — nunca recalculados aqui (drift.py só compara).

O relógio do monitoramento é `pickup_datetime` (mundo simulado, 2020-2021);
`created_at` é relógio de parede do replay e não aparece aqui.

Narrativa alerta precoce vs. confirmação atrasada (CLAUDE.md):
- /monitoring/drift só olha INPUTS logados — disponível assim que o mês é
  reproduzido, sem esperar ground truth;
- /monitoring/performance depende do feedback ter chegado — é a
  confirmação atrasada (mae_min=None enquanto não há pares).
"""

import sqlite3

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api import db
from api.schemas import (
    DriftFeature,
    DriftResponse,
    PerformanceResponse,
    RetrainCheckResponse,
    RetrainMonth,
)
from drift import (
    PSI_ALERT,
    PSI_WATCH,
    RETRAIN_K,
    RETRAIN_M,
    compute_drift,
    retrain_status,
)
from train import metrics

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

MONTH_QUERY = Query(
    pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
    description="mês de pickup no mundo simulado, ex: 2020-04",
    examples=["2020-04"],
)

# Aliases: colunas do SQLite → nomes de features do treino, para o DataFrame
# sair pronto para os specs de psi_bins do metadata.
FEATURES_SQL = """
SELECT pu_location_id  AS "PULocationID",
       do_location_id  AS "DOLocationID",
       trip_type, haversine_km, pickup_hour, pickup_dayofweek,
       is_weekend, is_holiday
FROM predictions
WHERE pickup_datetime >= ? AND pickup_datetime < ?
"""

PERFORMANCE_SQL = """
SELECT predicted_duration_min, actual_duration_min
FROM predictions
WHERE pickup_datetime >= ? AND pickup_datetime < ?
"""


def month_bounds(month: str) -> tuple[str, str]:
    """[início, fim) do mês em ISO — comparação lexicográfica funciona
    contra o pickup_datetime armazenado como isoformat."""
    following = pd.Period(month, freq="M") + 1
    return f"{month}-01", f"{following.year:04d}-{following.month:02d}-01"


def load_month_features(conn: sqlite3.Connection, month: str) -> pd.DataFrame:
    return pd.read_sql_query(FEATURES_SQL, conn, params=month_bounds(month))


def psi_level(psi: float) -> str:
    if psi > PSI_ALERT:
        return "alert"
    if psi > PSI_WATCH:
        return "watch"
    return "stable"


@router.get("/drift", response_model=DriftResponse)
def drift(request: Request, month: str = MONTH_QUERY,
          conn: sqlite3.Connection = Depends(db.get_db)) -> DriftResponse:
    features_df = load_month_features(conn, month)
    if features_df.empty:
        raise HTTPException(404, f"nenhuma predição logada para {month}")

    report = compute_drift(features_df, request.app.state.metadata["psi_bins"])
    features = {
        name: DriftFeature(psi=entry["psi"], ks=entry.get("ks"),
                           level=psi_level(entry["psi"]))
        for name, entry in report.items()
    }
    return DriftResponse(
        month=month,
        n_predictions=len(features_df),
        thresholds={"watch": PSI_WATCH, "alert": PSI_ALERT},
        features=features,
        n_alert=sum(f.level == "alert" for f in features.values()),
    )


@router.get("/performance", response_model=PerformanceResponse)
def performance(request: Request, month: str = MONTH_QUERY,
                conn: sqlite3.Connection = Depends(db.get_db)
                ) -> PerformanceResponse:
    rows = pd.read_sql_query(PERFORMANCE_SQL, conn, params=month_bounds(month))
    if rows.empty:
        raise HTTPException(404, f"nenhuma predição logada para {month}")

    pairs = rows.dropna(subset=["actual_duration_min"])
    month_metrics: dict = {"mae_min": None, "rmse_min": None,
                           "median_ae_min": None, "bias_min": None}
    if not pairs.empty:
        actual = pairs["actual_duration_min"].to_numpy()
        predicted = pairs["predicted_duration_min"].to_numpy()
        month_metrics = metrics(actual, predicted)  # mesma conta do holdout
        month_metrics["bias_min"] = float((predicted - actual).mean())

    return PerformanceResponse(
        month=month,
        n_predictions=len(rows),
        n_with_feedback=len(pairs),
        holdout_reference=request.app.state.metadata["holdout_metrics"],
        **month_metrics,
    )


@router.get("/retrain-check", response_model=RetrainCheckResponse)
def retrain_check(request: Request, month: str = MONTH_QUERY,
                  conn: sqlite3.Connection = Depends(db.get_db)
                  ) -> RetrainCheckResponse:
    """Regra v1 (decisão #8): vermelho se >= K features com PSI > 0.2 em
    todos os últimos M meses (janela terminando no mês pedido). Mês sem
    dados não entra — e portanto impede a streak de fechar."""
    specs = request.app.state.metadata["psi_bins"]
    end = pd.Period(month, freq="M")
    window = [str(end - offset) for offset in range(RETRAIN_M - 1, -1, -1)]

    months = []
    for window_month in window:
        features_df = load_month_features(conn, window_month)
        if features_df.empty:
            continue
        report = compute_drift(features_df, specs)
        alert_features = sorted(
            name for name, entry in report.items() if entry["psi"] > PSI_ALERT
        )
        months.append(RetrainMonth(month=window_month,
                                   n_alert=len(alert_features),
                                   alert_features=alert_features))

    if not months or months[-1].month != month:
        raise HTTPException(404, f"nenhuma predição logada para {month}")

    return RetrainCheckResponse(
        month=month,
        status=retrain_status([m.n_alert for m in months]),
        rule={"k": RETRAIN_K, "m": RETRAIN_M, "psi_alert": PSI_ALERT},
        months=months,
    )