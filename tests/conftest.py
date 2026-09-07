"""Fixtures dos testes da API: app de teste com modelo, metadata e
centroides FALSOS, e um banco SQLite descartável por teste.

Por que fakes e não o app de verdade: `models/*.pkl` e `data/processed/`
estão fora do git (só o `metadata.json` é versionado). Num clone limpo não
existe `model.pkl` nem o CSV de centroides, e o lifespan do `api.main` abre
os dois no startup — um teste que dependesse deles não rodaria em lugar
nenhum além da máquina de quem replayou os dados. A app daqui espelha o
`main.py` (mesmos routers, mesmo /health) e enche o `app.state` à mão; um
teste-guarda em test_api_predict.py compara as rotas das duas para a cópia
não divergir calada.

O metadata falso tem bins minúsculos de propósito: com dois bins de 50%
cada, uma feature 50/50 dá PSI 0 e uma feature 100% num bin só dá PSI ~4,6.
Isso deixa "quantas features estão em alerta neste mês" ser escolhido pelo
teste, que é o insumo do gatilho de drift.
"""

from contextlib import asynccontextmanager
from itertools import count

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import config, db
from api.routers import feedback, monitoring, predict
from features import FEATURE_COLUMNS

# --- os fakes ---------------------------------------------------------------

# Zonas 1-4 existem; 264/265 ficam de fora, como no shapefile real (elas não
# têm centroide e produzem haversine NaN — ver features.build_zone_centroids).
CENTROIDS = pd.DataFrame(
    {
        "LocationID": [1, 2, 3, 4],
        "lat": [40.70, 40.75, 40.80, 40.85],
        "lon": [-74.00, -73.99, -73.98, -73.97],
    }
).set_index("LocationID")


class FakeBooster:
    """Modelo determinístico: 5 min + 3 min por km, NaN de zona sem
    centroide contando como 0 km. Não imita o LightGBM real — imita só o
    contrato que o router usa (`predict(df) -> array`)."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        km = np.nan_to_num(features["haversine_km"].to_numpy(dtype=float))
        return 5.0 + 3.0 * km


# Valores (bin A, bin B) de cada feature no banco semeado. Alternar entre os
# dois dá 50/50 = PSI 0; fixar em A dá 100% num bin = PSI ~4,6 (alerta).
FEATURE_VALUES = {
    "PULocationID": (1, 2),
    "DOLocationID": (1, 2),
    "trip_type": (1, 2),
    "haversine_km": (1.0, 3.0),
    "pickup_hour": (8, 18),
    "pickup_dayofweek": (1, 5),
    "is_weekend": (0, 1),
    "is_holiday": (0, 1),
}

DB_COLUMNS = {"PULocationID": "pu_location_id", "DOLocationID": "do_location_id"}

HOLDOUT_MAE = 4.0  # limiar do gatilho de performance = 4,0 × 1,25 = 5,0


def _categorical(categories):
    return {"kind": "categorical", "categories": list(categories),
            "expected": [0.5, 0.5]}


def fake_metadata() -> dict:
    """Metadata no mesmo formato do que o train.py grava, com bins de dois
    compartimentos iguais. A ordem das chaves de psi_bins é a ordem em que
    as features saem nas respostas."""
    return {
        "model_version": "test-0.0.1",
        "trained_at": "2019-12-31T00:00:00",
        "features": list(FEATURE_COLUMNS),
        "holdout_metrics": {"mae_min": HOLDOUT_MAE, "rmse_min": 7.0,
                            "median_ae_min": 2.5},
        "psi_bins": {
            "PULocationID": _categorical([1, 2]),
            "DOLocationID": _categorical([1, 2]),
            "trip_type": _categorical([1, 2]),
            # única contínua, como no metadata real — é a única que ganha KS
            "haversine_km": {"kind": "continuous", "edges": [2.0],
                             "expected": [0.5, 0.5], "nan_expected": 0.0},
            "pickup_hour": _categorical([8, 18]),
            "pickup_dayofweek": _categorical([1, 5]),
            "is_weekend": _categorical([0, 1]),
            "is_holiday": _categorical([0, 1]),
        },
    }


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """Banco descartável. `settings` é instanciado no import do
    api.config, então variável de ambiente não pega mais — o jeito de
    redirecionar é o atributo do objeto, que é o mesmo que o api/db.py
    importou."""
    path = tmp_path / "api.db"
    monkeypatch.setattr(config.settings, "db_path", path)
    db.init_db()
    return path


@pytest.fixture
def metadata():
    return fake_metadata()


@pytest.fixture
def app(db_path, metadata) -> FastAPI:
    """Espelho do api.main.app com o state preenchido à mão."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.booster = FakeBooster()
        app.state.metadata = metadata
        app.state.centroids = CENTROIDS
        db.init_db()
        yield

    app = FastAPI(title="teste", lifespan=lifespan)
    app.include_router(predict.router)
    app.include_router(feedback.router)
    app.include_router(monitoring.router)

    @app.get("/health")
    def health():
        return {"status": "ok",
                "model_version": app.state.metadata["model_version"],
                "trained_at": app.state.metadata["trained_at"]}

    return app


@pytest.fixture
def client(app):
    # context manager: sem ele o lifespan não roda e o app.state fica vazio.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def conn(db_path):
    connection = db.get_connection()
    yield connection
    connection.close()


@pytest.fixture
def seed(conn):
    """Semeia um mês de predições direto no banco.

    Passar pelo /predict daria n requests para montar 24 meses e amarraria
    os testes de monitoramento ao caminho de inferência; o que o
    monitoramento lê é o log, e é o log que se escreve aqui.

    - `drifted`: features que ficam 100% no bin A (PSI ~4,6 = alerta). As
      demais alternam A/B e dão PSI 0.
    - `mae`: erro constante de cada par com feedback (predito = real + mae,
      logo viés = +mae). None = mês sem nenhum feedback.
    - `n_feedback`: quantos dos n recebem feedback (default: todos).
    - `pickup`: horário do embarque cravado (default: dia 15 do mês), para
      testar as bordas da janela do mês.
    """
    counter = count()

    def _seed(month, n=200, drifted=(), mae=None, n_feedback=None, pickup=None):
        n_feedback = n if n_feedback is None else n_feedback
        pickup = pickup or f"{month}-15T00:00:00"
        for i in range(n):
            values = {
                name: pair[0] if name in drifted else pair[i % 2]
                for name, pair in FEATURE_VALUES.items()
            }
            row = {
                "prediction_id": f"{month}-{next(counter):05d}",
                "created_at": "2026-01-01T00:00:00+00:00",
                "pickup_datetime": pickup,
                "predicted_duration_min": 10.0 + (mae or 0.0),
                "actual_duration_min": (
                    10.0 if mae is not None and i < n_feedback else None),
                "feedback_at": (
                    "2026-01-01T01:00:00+00:00"
                    if mae is not None and i < n_feedback else None),
            } | {DB_COLUMNS.get(name, name): value
                 for name, value in values.items()}
            db.insert_prediction(conn, row)

    return _seed
