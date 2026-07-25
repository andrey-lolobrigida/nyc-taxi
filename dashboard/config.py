"""Ambiente do dashboard.

Mesma convenção da `api/config.py`: só entra aqui o que muda de máquina para
máquina. Decisões de projeto (paleta, thresholds, layout) ficam em Python nos
módulos donos — `theme.py`, `drift.py` —, não em configuração.

Override por variável de ambiente com prefixo TAXI_DASH_, ex:
    TAXI_DASH_API_URL=http://127.0.0.1:9000 streamlit run dashboard/app.py
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TAXI_DASH_")

    api_url: str = "http://127.0.0.1:8000"
    intents_path: Path = ROOT / "data" / "retrain_intents.log"


settings = Settings()
