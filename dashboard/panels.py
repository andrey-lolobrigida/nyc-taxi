"""Os três painéis estilo formulário — o "certificado" do mês selecionado.

Enquanto o sismógrafo mostra a forma, aqui ficam os números exatos: nenhum
valor da página existe só como cor ou só como posição num gráfico. Cada
painel é HTML montado à mão (em vez de `st.dataframe`) porque o design pede
borda fina, cabeçalho em caixa alta e dado em monoespaçada — controle que a
tabela pronta do Streamlit não dá.
"""

import html
from datetime import datetime

import pandas as pd

from config import settings
from theme import LEVEL_COLOR, LEVEL_LABEL, STATUS_COLOR, STATUS_LABEL, STATUS_READING


def _panel(title: str, body: str) -> str:
    return (f'<div class="panel"><div class="panel-title">{title}</div>'
            f"{body}</div>")


def distribution(timeline, month: str) -> str:
    """Seção 1: PSI e KS por feature, com a leitura em palavras."""
    rows = []
    for row in timeline.drift_of(month).itertuples():
        ks = "—" if pd.isna(row.ks) else f"{row.ks:.3f}"
        rows.append(
            f'<tr><td>{html.escape(row.feature)}</td>'
            f'<td class="num">{row.psi:.3f}</td><td class="num">{ks}</td>'
            f'<td><span class="dot" style="background:'
            f'{LEVEL_COLOR[row.level]}"></span>{LEVEL_LABEL[row.level]}</td>'
            "</tr>")
    return _panel(
        "1 · exame de distribuição",
        '<table class="form"><thead><tr><th>feature</th><th class="num">PSI'
        '</th><th class="num">KS</th><th>leitura</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table>"
        '<p class="foot">KS só para contínuas. PSI contra os bins congelados '
        "de 2019 — a régua nunca muda de tamanho.</p>")


def performance(row: pd.Series, holdout: dict) -> str:
    """Seção 2: erro do mês contra o baseline do holdout de 2019."""
    coverage = row["n_with_feedback"] / row["n_predictions"]

    def line(label: str, value, fmt: str, reference: str = "") -> str:
        shown = "—" if value is None or pd.isna(value) else fmt.format(value)
        return (f'<tr><td>{label}</td><td class="num">{shown}</td>'
                f'<td class="num ref">{reference}</td></tr>')

    delta = (None if pd.isna(row["mae_min"])
             else row["mae_min"] - holdout["mae_min"])
    body = ['<table class="form"><thead><tr><th>métrica</th>'
            '<th class="num">mês</th><th class="num">holdout 2019</th>'
            "</tr></thead><tbody>",
            line("MAE (min)", row["mae_min"], "{:.2f}",
                 f"{holdout['mae_min']:.2f}"),
            line("RMSE (min)", row["rmse_min"], "{:.2f}",
                 f"{holdout['rmse_min']:.2f}"),
            line("erro mediano", row["median_ae_min"], "{:.2f}",
                 f"{holdout['median_ae_min']:.2f}"),
            line("viés (min)", row["bias_min"], "{:+.2f}", "0.00"),
            line("Δ MAE vs. holdout", delta, "{:+.2f}"),
            line("predições", row["n_predictions"], "{:,.0f}"),
            line("cobertura de feedback", coverage, "{:.1%}"),
            "</tbody></table>"]
    return _panel("2 · desempenho aferido", "".join(body) +
                  '<p class="foot">Viés positivo = o modelo superestima a '
                  "duração. Sem feedback do mês, tudo aqui fica em branco: é "
                  "a confirmação atrasada ainda em trânsito.</p>")


def retrain(row: pd.Series, streak: int, rule: dict) -> str:
    """Seção 3: o parecer da regra de retreino (decisão #8 do CLAUDE.md)."""
    status = row["retrain_status"]
    listed = ", ".join(row["alert_features"]) or "nenhuma"
    return _panel(
        "3 · parecer de retreino",
        f'<div class="verdict"><span class="pill" style="border-color:'
        f'{STATUS_COLOR[status]};color:{STATUS_COLOR[status]}">'
        f"{STATUS_LABEL[status]}</span>"
        f'<span class="verdict-text">{STATUS_READING[status]}</span></div>'
        '<table class="form"><tbody>'
        f'<tr><td>features em alerta</td>'
        f'<td class="num">{row["n_alert"]}</td></tr>'
        f'<tr><td>quais</td><td class="num">{html.escape(listed)}</td></tr>'
        f'<tr><td>streak (meses ≥ K)</td><td class="num">{streak}</td></tr>'
        f'<tr><td>regra vigente</td><td class="num">K={rule["k"]:.0f} · '
        f'M={rule["m"]:.0f} · PSI &gt; {rule["psi_alert"]:g}</td></tr>'
        "</tbody></table>"
        '<p class="foot">Vermelho = K features em alerta por M meses '
        "seguidos. Amarelo = o mês estourou, a streak não.</p>")


def record_intent(month: str, row: pd.Series) -> str:
    """Botão mock: não retreina nada, só registra a intenção junto com o
    contexto que a justificaria (decisão em aberto no CLAUDE.md)."""
    stamp = datetime.now().isoformat(timespec="seconds")
    line = (f"{stamp}\tmes={month}\tstatus={row['retrain_status']}\t"
            f"n_alert={row['n_alert']}\t"
            f"features={','.join(row['alert_features']) or '-'}\n")
    settings.intents_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.intents_path.open("a") as f:
        f.write(line)
    return line
