"""POST /predict — inferência + log da predição.

O caminho de features é o add_features do src/features.py, o MESMO código do
treino, aplicado a um DataFrame de uma linha. Qualquer transformação feita
aqui fora dele seria reintrodução do training-serving skew.
"""

import math
import sqlite3
import uuid
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, Depends, Request

from api import db
from api.schemas import PredictRequest, PredictResponse
from features import FEATURE_COLUMNS, add_features

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request,
            conn: sqlite3.Connection = Depends(db.get_db)) -> PredictResponse:
    state = request.app.state

    # O dataset da TLC é naive em horário local de NY; se o cliente mandar
    # offset explícito, descarta-se o fuso e fica o relógio de parede.
    pickup = pd.Timestamp(req.pickup_datetime).tz_localize(None)

    trips = pd.DataFrame({
        "lpep_pickup_datetime": [pickup],
        "PULocationID": [req.PULocationID],
        "DOLocationID": [req.DOLocationID],
        "trip_type": [float("nan") if req.trip_type is None else req.trip_type],
    })
    features = add_features(trips, state.centroids)[FEATURE_COLUMNS]

    predicted = float(state.booster.predict(features)[0])

    row = features.iloc[0]
    haversine = float(row["haversine_km"])
    prediction_id = str(uuid.uuid4())
    db.insert_prediction(conn, {
        "prediction_id": prediction_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pickup_datetime": pickup.isoformat(),
        "pu_location_id": int(row["PULocationID"]),
        "do_location_id": int(row["DOLocationID"]),
        "trip_type": int(row["trip_type"]),
        "haversine_km": None if math.isnan(haversine) else haversine,
        "pickup_hour": int(row["pickup_hour"]),
        "pickup_dayofweek": int(row["pickup_dayofweek"]),
        "is_weekend": int(row["is_weekend"]),
        "is_holiday": int(row["is_holiday"]),
        "predicted_duration_min": predicted,
    })

    return PredictResponse(
        prediction_id=prediction_id,
        predicted_duration_min=predicted,
        model_version=state.metadata["model_version"],
    )