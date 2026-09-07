"""Cliente da API de monitoramento — a única porta de entrada de dados.

Decisão #5 do CLAUDE.md: o dashboard nunca abre o SQLite. Tudo vem de
`GET /monitoring/timeline`, numa chamada só; o slider de mês é filtragem
local sobre o que já está em memória, sem round-trip por mês.

O `Timeline` abaixo é a timeline da API já em formato de trabalho
(DataFrames) mais os parâmetros vigentes (thresholds, regra de retreino,
baseline do holdout). Nada aqui reimplementa conta de drift: quem calcula é
o backend, isto aqui só arruma o formato.
"""

from dataclasses import dataclass

import pandas as pd
import requests
import streamlit as st


@dataclass(frozen=True)
class Timeline:
    drift: pd.DataFrame        # formato longo: mês × feature
    performance: pd.DataFrame  # uma linha por mês, com o veredito achatado
    features: list[str]
    thresholds: dict[str, float]
    rule: dict[str, float]
    holdout: dict[str, float]
    mae_threshold: float       # limiar do gatilho de performance
    events: list[dict]         # reconhecimentos, mais recente primeiro

    @property
    def months(self) -> list[str]:
        return self.performance["month"].tolist()

    def month(self, month: str) -> pd.Series:
        return self.performance[self.performance["month"] == month].iloc[0]

    def drift_of(self, month: str) -> pd.DataFrame:
        return self.drift[self.drift["month"] == month]

    def alert_streak(self, month: str) -> int:
        """Meses consecutivos, terminando no selecionado, com >= K features
        em alerta — o que a regra de retreino conta para fechar a streak."""
        k = int(self.rule["k"])
        history = self.performance[self.performance["month"] <= month]
        streak = 0
        for n_alert in reversed(history.sort_values("month")["n_alert"].tolist()):
            if n_alert < k:
                break
            streak += 1
        return streak


def _get(api_url: str, path: str) -> dict:
    response = requests.get(f"{api_url}{path}", timeout=120)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def health(api_url: str) -> dict:
    return _get(api_url, "/health")


@st.cache_data(ttl=300, show_spinner="consultando a API de monitoramento…")
def timeline(api_url: str) -> Timeline:
    payload = _get(api_url, "/monitoring/timeline")
    drift_rows, perf_rows = [], []
    for month in payload["months"]:
        date = pd.Timestamp(f"{month['month']}-01")
        for name in payload["feature_names"]:
            entry = month["features"][name]
            drift_rows.append({"date": date, "month": month["month"],
                               "feature": name, "psi": entry["psi"],
                               "ks": entry["ks"], "level": entry["level"]})
        # `verdict` é achatado na mesma linha: a série vira DataFrame, e
        # coluna aninhada não serve nem para filtro nem para gráfico.
        perf_rows.append({"date": date,
                          **{k: v for k, v in month.items()
                             if k not in ("features", "verdict")},
                          **month["verdict"]})
    return Timeline(
        drift=pd.DataFrame(drift_rows),
        performance=pd.DataFrame(perf_rows),
        features=payload["feature_names"],
        thresholds=payload["thresholds"],
        rule=payload["rule"],
        holdout=payload["holdout_reference"],
        mae_threshold=payload["mae_threshold_min"],
        events=payload["retrain_events"],
    )


def trigger_retrain(api_url: str, month: str, note: str | None = None) -> dict:
    """Registra o reconhecimento do alarme na API. Nada é treinado.

    Sem cache, ao contrário das leituras acima: é escrita, e o resultado
    muda a timeline (o alarme passa a aparecer silenciado). Quem chama
    precisa limpar o cache de `timeline` depois.
    """
    response = requests.post(f"{api_url}/monitoring/retrain-trigger",
                             json={"month": month, "note": note}, timeout=120)
    response.raise_for_status()
    return response.json()
