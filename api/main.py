"""App FastAPI: inferência (/predict), feedback (/feedback) e, na fase 3,
monitoramento (/monitoring/*) — routers separados sobre o mesmo app
(decisão #2 do CLAUDE.md: caixa de banco vs. setor de auditoria).

Modelo, metadata e centroides são carregados UMA vez no lifespan e ficam em
app.state — o custo por request do /predict é só feature de uma linha +
predict do booster. No startup o schema de features do metadata é validado
contra o features.py importado: se divergirem, o processo morre na hora em
vez de servir predições silenciosamente erradas (fail fast contra skew de
versão entre modelo salvo e código).

Uso:
    uv run uvicorn api.main:app [--reload]
"""

import json
import pickle
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import settings
from api.db import init_db
from api.routers import feedback, predict
from features import FEATURE_COLUMNS, load_zone_centroids


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(settings.models_dir / "model.pkl", "rb") as f:
        app.state.booster = pickle.load(f)
    app.state.metadata = json.loads(
        (settings.models_dir / "metadata.json").read_text())
    app.state.centroids = load_zone_centroids(settings.centroids_path)

    if app.state.metadata["features"] != FEATURE_COLUMNS:
        raise RuntimeError(
            "schema de features do metadata.json difere do features.py: "
            f"{app.state.metadata['features']} != {FEATURE_COLUMNS} — "
            "modelo salvo e código estão fora de sincronia; retreine ou "
            "faça checkout da versão correspondente."
        )

    init_db()
    yield


app = FastAPI(title="NYC Green Taxi — duração de corrida", lifespan=lifespan)
app.include_router(predict.router)
app.include_router(feedback.router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": app.state.metadata["model_version"],
        "trained_at": app.state.metadata["trained_at"],
    }