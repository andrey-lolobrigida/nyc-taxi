"""Configuração de ambiente da API via pydantic-settings.

Só entra aqui o que é ambiente (caminhos que podem variar entre máquina
local e um deploy futuro). Constantes que são decisões de projeto —
thresholds de PSI, schema de features — vivem em Python nos módulos donos
(drift.py, features.py), não aqui.

Override por variável de ambiente com prefixo TAXI_API_, ex:
    TAXI_API_DB_PATH=/tmp/test.db uvicorn api.main:app
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAXI_API_")

    db_path: Path = ROOT / "data" / "api.db"
    models_dir: Path = ROOT / "models"
    centroids_path: Path = ROOT / "data" / "processed" / "zone_centroids.csv"


settings = Settings()