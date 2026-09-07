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

-- Reconhecimentos do gatilho de retreino: o botão do dashboard não treina
-- nada (o pipeline de retreino é mock por escopo — este projeto é de
-- MONITORAMENTO). O que ele faz é registrar que um humano viu o alarme e
-- assumiu a ação, que é o que silencia o alarme até o episódio se resolver.
-- Sem isso o dashboard não distingue "estourou" de "estourou e ninguém
-- atendeu" — e a série real fica 17 meses em vermelho sem essa distinção.
--
-- `month` é o mês do MUNDO SIMULADO reconhecido (2020-2021); `created_at` é
-- relógio de parede de quem clicou. Mesma separação de relógios do resto do
-- monitoramento.
CREATE TABLE IF NOT EXISTS retrain_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    month        TEXT NOT NULL,
    triggered_by TEXT NOT NULL,
    note         TEXT
);
CREATE INDEX IF NOT EXISTS idx_retrain_events_month
    ON retrain_events (month);
"""


def get_connection() -> sqlite3.Connection:
    """Conexão nova (uma por request). busy_timeout cobre a janela rara de
    dois escritores simultâneos, que WAL não elimina.

    check_same_thread=False: o FastAPI pode rodar o endpoint e o teardown
    da dependência em threads DIFERENTES do threadpool — o uso continua
    sequencial dentro do request (nunca duas threads ao mesmo tempo na
    mesma conexão), que é o cenário seguro. Sem isso, o replay concorrente
    da fase 3 estoura sqlite3.ProgrammingError no conn.close()."""
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
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


def insert_retrain_event(conn: sqlite3.Connection, month: str,
                         triggered_by: str, created_at: str,
                         note: str | None = None) -> int:
    """Registra um reconhecimento do gatilho. Retorna o id gravado."""
    with conn:
        cursor = conn.execute(
            "INSERT INTO retrain_events (created_at, month, triggered_by,"
            " note) VALUES (?, ?, ?, ?)",
            (created_at, month, triggered_by, note),
        )
    return int(cursor.lastrowid)


def acknowledged_months(conn: sqlite3.Connection) -> set[str]:
    """Meses do mundo simulado que têm ao menos um reconhecimento.

    Conjunto, não lista: clicar duas vezes no mesmo mês é a mesma
    informação. A validade de cada reconhecimento (quando expira) é conta
    de drift.acknowledgement_flags, não do banco.
    """
    return {row["month"] for row in
            conn.execute("SELECT DISTINCT month FROM retrain_events")}


def retrain_events(conn: sqlite3.Connection) -> list[dict]:
    """Histórico completo, mais recente primeiro (para auditoria)."""
    rows = conn.execute(
        "SELECT id, created_at, month, triggered_by, note FROM retrain_events"
        " ORDER BY created_at DESC, id DESC"
    ).fetchall()
    return [dict(row) for row in rows]
