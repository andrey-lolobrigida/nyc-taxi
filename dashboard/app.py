"""Dashboard de diagnóstico do modelo em produção (fase 3) — a página.

Só o layout mora aqui; cada peça tem seu módulo vizinho:

    config.py       ambiente (URL da API, caminho do log de intenções)
    theme.py        paleta, tema do Altair, CSS
    monitoring.py   cliente da API — a única porta de entrada de dados
    seismograph.py  o gráfico de pistas
    panels.py       os três painéis do mês selecionado

Layout (design aprovado no CLAUDE.md — "sismógrafo + certificado"),
de cima para baixo:

1. Cabeçalho: versão do modelo (`/health`) e a agulha (slider de mês), que
   dirige a página inteira.
2. Sismógrafo: PSI por feature (alerta precoce) e, depois do divisor, MAE e
   viés (confirmação atrasada). A defasagem entre as duas metades é o
   argumento central da página.
3. Três painéis com os números exatos do mês selecionado.

Uso:
    make api          # a API precisa estar no ar, noutro terminal
    make dashboard
"""

import html

import requests
import streamlit as st

import monitoring
import panels
import seismograph
import theme
from config import settings

DEFAULT_MONTH = "2020-04"  # o mês em que a agulha salta pela primeira vez


def api_down(error: Exception) -> None:
    st.markdown(
        "### API de monitoramento fora do ar\n\n"
        f"O dashboard só lê a API (`{settings.api_url}`) — ele nunca abre o "
        "banco. Suba a API e recarregue:\n\n```bash\nmake api\n```")
    st.caption(f"{type(error).__name__}: {error}")
    st.stop()


def masthead(health: dict, timeline: monitoring.Timeline) -> None:
    st.markdown(
        f'<div class="masthead"><h1>Vigilância de drift — NYC Green Taxi</h1>'
        f'<span class="meta">modelo {html.escape(health["model_version"])} · '
        f'treinado em {health["trained_at"][:10]} · '
        f"{len(timeline.months)} meses de produção · "
        f'{theme.num(timeline.performance["n_predictions"].sum(), "{:,.0f}")}'
        " predições"
        "</span></div>", unsafe_allow_html=True)


def severity_key(thresholds: dict) -> None:
    st.markdown(
        '<div class="key">'
        f'<span><span class="dot" style="background:{theme.STABLE}"></span>'
        f'estável · PSI &lt; {theme.num(thresholds["watch"], "{:g}")}</span>'
        f'<span><span class="dot" style="background:{theme.WATCH}"></span>'
        "atenção</span>"
        f'<span><span class="dot" style="background:{theme.ALERT}"></span>'
        f'alerta · PSI &gt; {theme.num(thresholds["alert"], "{:g}")}</span>'
        f'<span style="color:{theme.NEEDLE}">│ agulha — mês selecionado</span>'
        "</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Drift — NYC Green Taxi", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(theme.CSS, unsafe_allow_html=True)

    try:
        health = monitoring.health(settings.api_url)
        timeline = monitoring.timeline(settings.api_url)
    except requests.RequestException as error:
        api_down(error)

    masthead(health, timeline)

    months = timeline.months
    default = DEFAULT_MONTH if DEFAULT_MONTH in months else months[-1]
    month = st.select_slider("agulha", options=months, value=default,
                             label_visibility="collapsed")
    row = timeline.month(month)

    severity_key(timeline.thresholds)
    st.altair_chart(seismograph.chart(timeline, month),
                    use_container_width=False)

    left, middle, right = st.columns([1.15, 1, 1], gap="medium")
    with left:
        st.markdown(panels.distribution(timeline, month),
                    unsafe_allow_html=True)
    with middle:
        st.markdown(panels.performance(row, timeline.holdout),
                    unsafe_allow_html=True)
    with right:
        st.markdown(panels.retrain(row, timeline.alert_streak(month),
                                   timeline.rule), unsafe_allow_html=True)
        if st.button(f"disparar retreino · {month}"):
            panels.record_intent(month, row)
            st.success(f"Intenção registrada em {settings.intents_path.name}. "
                       "Nenhum modelo foi treinado — o pipeline de retreino "
                       "ainda é um mock.")


if __name__ == "__main__":
    main()
