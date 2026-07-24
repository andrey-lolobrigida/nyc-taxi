"""POST /feedback — ground truth atrasado.

Em produção real a duração verdadeira só existe quando a corrida termina;
o replay da fase 3 simula esse atraso mandando o feedback depois do predict.
Cada feedback preenche a linha já logada — nunca cria linha nova.
"""

import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from api import db
from api.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest,
             conn: sqlite3.Connection = Depends(db.get_db)) -> FeedbackResponse:
    result = db.apply_feedback(
        conn, req.prediction_id, req.actual_duration_min,
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if result == "not_found":
        raise HTTPException(404, f"prediction_id desconhecido: {req.prediction_id}")
    if result == "duplicate":
        raise HTTPException(409, f"feedback já registrado para {req.prediction_id}")
    return FeedbackResponse(prediction_id=req.prediction_id, status="ok")