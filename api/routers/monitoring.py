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
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api import db
from api.schemas import (
    DriftFeature,
    DriftResponse,
    MonthVerdict,
    PerformanceResponse,
    RetrainCheckResponse,
    RetrainMonth,
    RetrainTriggerRequest,
    RetrainTriggerResponse,
    TimelineMonth,
    TimelineResponse,
)
from drift import (
    PERF_M,
    PERF_MIN_FEEDBACK,
    PERF_TOLERANCE,
    PSI_ALERT,
    PSI_WATCH,
    RETRAIN_K,
    RETRAIN_M,
    acknowledgement_flags,
    compute_drift,
    performance_status,
    performance_threshold,
    retrain_status,
    worst_status,
)
from train import metrics

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Parâmetros vigentes dos dois gatilhos, ecoados nas respostas: o dashboard
# desenha a regra em vez de repetir os números por conta própria.
RULE = {"k": RETRAIN_K, "m": RETRAIN_M, "psi_alert": PSI_ALERT,
        "perf_tolerance": PERF_TOLERANCE, "perf_m": PERF_M,
        "perf_min_feedback": PERF_MIN_FEEDBACK}

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


def has_confirmation(entry: dict) -> bool:
    """O mês tem feedback suficiente para julgar performance?

    Mês sem confirmação não entra na janela do gatilho de performance —
    mesma convenção que o gatilho de drift usa para mês sem predição: sem
    medição não se afirma nem se nega streak. O piso existe porque um mês
    com um punhado de feedbacks decidiria um alarme sozinho.
    """
    return (entry["mae_min"] is not None
            and entry["n_with_feedback"] >= PERF_MIN_FEEDBACK)


def calendar_window(month: str, size: int) -> list[str]:
    """Os `size` meses de CALENDÁRIO terminando em `month`.

    Janela de calendário, não "os últimos N meses com dados": um buraco no
    meio da série encurta a janela e impede a streak de fechar, em vez de
    costurar meses distantes como se fossem consecutivos.
    """
    end = pd.Period(month, freq="M")
    return [str(end - offset) for offset in range(size - 1, -1, -1)]


def monthly_signals(conn: sqlite3.Connection, metadata: dict) -> list[dict]:
    """A série mês a mês inteira: drift, performance e o parecer dos dois
    gatilhos, já com o reconhecimento aplicado.

    Fonte única dos dois endpoints de parecer. O veredito de um mês depende
    da série INTEIRA (o reconhecimento expira quando o gatilho volta a
    verde, o que só se sabe olhando os meses seguintes ao clique), então
    não dá para calcular um mês isolado — daí /retrain-check também passar
    por aqui. Custa uma varredura de ~2,5 s, o que o setor de auditoria da
    decisão #2 pode gastar à vontade.
    """
    rows = pd.read_sql_query(TIMELINE_SQL, conn)
    if rows.empty:
        raise HTTPException(404, "nenhuma predição logada")
    holdout_mae = metadata["holdout_metrics"]["mae_min"]

    months = []
    for month, group in rows.groupby("month", sort=True):
        features = drift_features(group, metadata["psi_bins"])
        alerts = alert_names(features)
        months.append({
            "month": month,
            "n_predictions": len(group),
            "n_with_feedback": int(group["actual_duration_min"].notna().sum()),
            "features": features,
            "n_alert": len(alerts),
            "alert_features": alerts,
            **month_performance(group),
        })

    by_month = {entry["month"]: entry for entry in months}
    drift_statuses, perf_statuses = [], []
    for entry in months:
        counts = [by_month[m]["n_alert"]
                  for m in calendar_window(entry["month"], RETRAIN_M)
                  if m in by_month]
        drift_statuses.append(retrain_status(counts))

        maes = [by_month[m]["mae_min"]
                for m in calendar_window(entry["month"], PERF_M)
                if m in by_month and has_confirmation(by_month[m])]
        perf_statuses.append(performance_status(maes, holdout_mae))

    # Um reconhecimento, aplicado SEPARADAMENTE a cada gatilho: aplicado ao
    # parecer geral ele nunca expiraria, porque o geral só volta a verde
    # quando os dois voltam (ver drift.acknowledgement_flags).
    labels = [entry["month"] for entry in months]
    acknowledged = db.acknowledged_months(conn)
    drift_ack = acknowledgement_flags(labels, drift_statuses, acknowledged)
    perf_ack = acknowledgement_flags(labels, perf_statuses, acknowledged)

    for entry, d_status, p_status, d_ack, p_ack in zip(
            months, drift_statuses, perf_statuses, drift_ack, perf_ack):
        entry["verdict"] = MonthVerdict(
            status=worst_status(d_status, p_status),
            drift_status=d_status, drift_acknowledged_month=d_ack,
            performance_status=p_status, performance_acknowledged_month=p_ack,
        )
    return months


@router.get("/retrain-check", response_model=RetrainCheckResponse)
def retrain_check(request: Request, month: str = MONTH_QUERY,
                  conn: sqlite3.Connection = Depends(db.get_db)
                  ) -> RetrainCheckResponse:
    """Parecer de retreino de um mês: os DOIS gatilhos, lado a lado.

    - drift: >= K features com PSI > 0.2 nos últimos M meses (decisão #8);
    - performance: MAE acima do holdout por uma margem, nos últimos M meses.

    O `status` é o pior dos dois. Eles discordam nas duas direções ao longo
    de 2020-2021, e é por isso que aparecem separados em vez de fundidos
    num número só (ver drift.py).
    """
    metadata = request.app.state.metadata
    months = monthly_signals(conn, metadata)
    by_month = {entry["month"]: entry for entry in months}
    if month not in by_month:
        raise HTTPException(404, f"nenhuma predição logada para {month}")

    window = calendar_window(month, max(RETRAIN_M, PERF_M))
    return RetrainCheckResponse(
        month=month,
        verdict=by_month[month]["verdict"],
        mae_threshold_min=performance_threshold(
            metadata["holdout_metrics"]["mae_min"]),
        rule=RULE,
        months=[RetrainMonth(**{k: by_month[m][k] for k in
                                ("month", "n_alert", "alert_features",
                                 "mae_min")})
                for m in window if m in by_month],
    )


@router.post("/retrain-trigger", response_model=RetrainTriggerResponse,
             status_code=201)
def retrain_trigger(request: Request, body: RetrainTriggerRequest,
                    conn: sqlite3.Connection = Depends(db.get_db)
                    ) -> RetrainTriggerResponse:
    """Registra que um humano viu o alarme e assumiu a ação. NÃO retreina.

    Retreinar está fora do escopo deste projeto (que é de monitoramento);
    o gatilho é mock e este endpoint é a forma honesta do mock: em vez de
    fingir um pipeline, ele grava o reconhecimento — que é justamente o que
    faltava para o dashboard distinguir "estourou" de "estourou e ninguém
    atendeu", e o que silencia o alarme até o episódio se resolver.

    Fica na API, não no dashboard, porque o dashboard nunca toca o banco
    (decisão #5): ele só consome esta API.
    """
    months = monthly_signals(conn, request.app.state.metadata)
    by_month = {entry["month"]: entry for entry in months}
    if body.month not in by_month:
        raise HTTPException(404, f"nenhuma predição logada para {body.month}")

    verdict = by_month[body.month]["verdict"]
    aceso = {"drift": verdict.drift_status != "green",
             "performance": verdict.performance_status != "green"}
    triggered_by = ("both" if all(aceso.values())
                    else next((name for name, on in aceso.items() if on),
                              "none"))

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    event_id = db.insert_retrain_event(conn, month=body.month,
                                       triggered_by=triggered_by,
                                       created_at=created_at, note=body.note)
    return RetrainTriggerResponse(id=event_id, month=body.month,
                                  created_at=created_at,
                                  triggered_by=triggered_by)


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
    months = monthly_signals(conn, metadata)
    return TimelineResponse(
        months=[TimelineMonth(**entry) for entry in months],
        feature_names=metadata["features"],
        thresholds={"watch": PSI_WATCH, "alert": PSI_ALERT},
        rule=RULE,
        mae_threshold_min=performance_threshold(
            metadata["holdout_metrics"]["mae_min"]),
        holdout_reference=metadata["holdout_metrics"],
        retrain_events=db.retrain_events(conn),
    )
