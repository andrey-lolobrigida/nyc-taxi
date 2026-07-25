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
    TimelineMonth,
    TimelineResponse,
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

# A timeline puxa TUDO de uma vez e agrupa em memória: 24 meses × 30k linhas
# saem em ~2,5 s numa passada só, contra 24×3 requests se o dashboard
# montasse a série chamando os endpoints mês a mês.
TIMELINE_SQL = """
SELECT substr(pickup_datetime, 1, 7) AS month,
       pu_location_id  AS "PULocationID",
       do_location_id  AS "DOLocationID",
       trip_type, haversine_km, pickup_hour, pickup_dayofweek,
       is_weekend, is_holiday,
       predicted_duration_min, actual_duration_min
FROM predictions
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


def drift_features(features_df: pd.DataFrame, specs: dict
                   ) -> dict[str, DriftFeature]:
    """compute_drift + classificação por threshold, no formato do contrato."""
    return {
        name: DriftFeature(psi=entry["psi"], ks=entry.get("ks"),
                           level=psi_level(entry["psi"]))
        for name, entry in compute_drift(features_df, specs).items()
    }


def alert_names(features: dict[str, DriftFeature]) -> list[str]:
    return sorted(name for name, f in features.items() if f.level == "alert")


def month_performance(rows: pd.DataFrame) -> dict:
    """MAE/RMSE/mediana/viés dos pares predição-feedback do mês.

    Tudo None enquanto nenhum feedback chegou — é a "confirmação atrasada"
    ainda em trânsito, não um erro.
    """
    pairs = rows.dropna(subset=["actual_duration_min"])
    if pairs.empty:
        return {"mae_min": None, "rmse_min": None,
                "median_ae_min": None, "bias_min": None}
    actual = pairs["actual_duration_min"].to_numpy()
    predicted = pairs["predicted_duration_min"].to_numpy()
    return metrics(actual, predicted) | {  # mesma conta do holdout
        "bias_min": float((predicted - actual).mean()),
    }


@router.get("/drift", response_model=DriftResponse)
def drift(request: Request, month: str = MONTH_QUERY,
          conn: sqlite3.Connection = Depends(db.get_db)) -> DriftResponse:
    features_df = load_month_features(conn, month)
    if features_df.empty:
        raise HTTPException(404, f"nenhuma predição logada para {month}")

    features = drift_features(features_df,
                              request.app.state.metadata["psi_bins"])
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

    return PerformanceResponse(
        month=month,
        n_predictions=len(rows),
        n_with_feedback=int(rows["actual_duration_min"].notna().sum()),
        holdout_reference=request.app.state.metadata["holdout_metrics"],
        **month_performance(rows),
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
        alerts = alert_names(drift_features(features_df, specs))
        months.append(RetrainMonth(month=window_month, n_alert=len(alerts),
                                   alert_features=alerts))

    if not months or months[-1].month != month:
        raise HTTPException(404, f"nenhuma predição logada para {month}")

    return RetrainCheckResponse(
        month=month,
        status=retrain_status([m.n_alert for m in months]),
        rule={"k": RETRAIN_K, "m": RETRAIN_M, "psi_alert": PSI_ALERT},
        months=months,
    )


@router.get("/timeline", response_model=TimelineResponse)
def timeline(request: Request, conn: sqlite3.Connection = Depends(db.get_db)
             ) -> TimelineResponse:
    """Série mês a mês de drift + performance + parecer de retreino.

    Existe para o dashboard: uma chamada devolve a página inteira, e o
    seletor de mês vira filtragem local em vez de round-trip por mês. Os
    endpoints por mês continuam sendo a fonte da verdade (mesmas funções);
    este aqui é a versão em lote.
    """
    metadata = request.app.state.metadata
    rows = pd.read_sql_query(TIMELINE_SQL, conn)
    if rows.empty:
        raise HTTPException(404, "nenhuma predição logada")

    by_month = {}
    for month, group in rows.groupby("month", sort=True):
        features = drift_features(group, metadata["psi_bins"])
        alerts = alert_names(features)
        by_month[month] = {
            "month": month,
            "n_predictions": len(group),
            "n_with_feedback": int(group["actual_duration_min"].notna().sum()),
            "features": features,
            "n_alert": len(alerts),
            "alert_features": alerts,
            **month_performance(group),
        }

    # Status de retreino: mesma janela de calendário do /retrain-check —
    # os M meses anteriores, contando só os que têm dados (mês sem medição
    # não confirma nem quebra a streak, apenas não entra).
    months = []
    for month, entry in by_month.items():
        end = pd.Period(month, freq="M")
        window = [str(end - offset) for offset in range(RETRAIN_M - 1, -1, -1)]
        counts = [by_month[m]["n_alert"] for m in window if m in by_month]
        months.append(TimelineMonth(**entry,
                                    retrain_status=retrain_status(counts)))

    return TimelineResponse(
        months=months,
        feature_names=metadata["features"],
        thresholds={"watch": PSI_WATCH, "alert": PSI_ALERT},
        rule={"k": RETRAIN_K, "m": RETRAIN_M, "psi_alert": PSI_ALERT},
        holdout_reference=metadata["holdout_metrics"],
    )