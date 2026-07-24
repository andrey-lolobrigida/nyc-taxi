"""SQLite da API: log de predições e feedback.

WAL (decisão #7 do CLAUDE.md): o replay martela writes no /predict enquanto
o dashboard lê via /monitoring/*; sem WAL essas duas coisas brigam
(`database is locked`). journal_mode é persistente no arquivo, mas setar por
conexão é inócuo e garante o modo mesmo num banco recém-criado.

Uma tabela única `predictions`, uma linha por request de /predict:
- inputs crus do request (pickup_datetime, zonas, trip_type);
- as FEATURES COMPUTADAS pelo features.py — é sobre elas que o PSI da fase 3
  roda, então o log precisa registrar o que o modelo viu, não só o que o
  cliente mandou;
- a predição; feedback (duração real) chega depois e preenche
  actual_duration_min/feedback_at na mesma linha.

`pickup_datetime` é o relógio do mundo simulado (2020-2021) e é por ele que
o monitoramento agrupa por mês; `created_at` é o relógio de parede do
replay, útil só para debug.
"""

import sqlite3

from api.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id          TEXT PRIMARY KEY,
    created_at             TEXT NOT NULL,
    pickup_datetime        TEXT NOT NULL,
    pu_location_id         INTEGER NOT NULL,
    do_location_id         INTEGER NOT NULL,
    trip_type              INTEGER NOT NULL,
    haversine_km           REAL,
    pickup_hour            INTEGER NOT NULL,
    pickup_dayofweek       INTEGER NOT NULL,
    is_weekend             INTEGER NOT NULL,
    is_holiday             INTEGER NOT NULL,
    predicted_duration_min REAL NOT NULL,
    actual_duration_min    REAL,
    feedback_at            TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_pickup
    ON predictions (pickup_datetime);
"""


def get_connection() -> sqlite3.Connection:
    """Conexão nova (uma por request — sqlite3 não é thread-safe entre
    requests do threadpool do FastAPI). busy_timeout cobre a janela rara de
    dois escritores simultâneos, que WAL não elimina."""
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def get_db():
    """Dependência FastAPI: uma conexão por request, fechada ao final."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def insert_prediction(conn: sqlite3.Connection, row: dict) -> None:
    columns = ", ".join(row)
    placeholders = ", ".join(f":{k}" for k in row)
    with conn:
        conn.execute(
            f"INSERT INTO predictions ({columns}) VALUES ({placeholders})", row
        )


def apply_feedback(conn: sqlite3.Connection, prediction_id: str,
                   actual_duration_min: float, feedback_at: str) -> str:
    """Associa o ground truth a uma predição. Retorna 'ok', 'not_found' ou
    'duplicate' — o router traduz em status HTTP."""
    exists = conn.execute(
        "SELECT actual_duration_min FROM predictions WHERE prediction_id = ?",
        (prediction_id,),
    ).fetchone()
    if exists is None:
        return "not_found"
    if exists["actual_duration_min"] is not None:
        return "duplicate"
    with conn:
        conn.execute(
            "UPDATE predictions SET actual_duration_min = ?, feedback_at = ?"
            " WHERE prediction_id = ?",
            (actual_duration_min, feedback_at, prediction_id),
        )
    return "ok"