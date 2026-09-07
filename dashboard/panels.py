"""Os três painéis estilo formulário — o "certificado" do mês selecionado.

Enquanto o sismógrafo mostra a forma, aqui ficam os números exatos: nenhum
valor da página existe só como cor ou só como posição num gráfico. Cada
painel é HTML montado à mão (em vez de `st.dataframe`) porque o design pede
borda fina, cabeçalho em caixa alta e dado em monoespaçada — controle que a
tabela pronta do Streamlit não dá.
"""

import html

import pandas as pd

from theme import (
    LEVEL_COLOR,
    LEVEL_LABEL,
    STATUS_COLOR,
    STATUS_LABEL,
    STATUS_READING,
    TRIGGER_LABEL,
    VERDICT_READING,
    num,
)


def _panel(title: str, body: str) -> str:
    return (f'<div class="panel"><div class="panel-title">{title}</div>'
            f"{body}</div>")


def distribution(timeline, month: str) -> str:
    """Seção 1: PSI e KS por feature, com a leitura em palavras."""
    rows = []
    for row in timeline.drift_of(month).itertuples():
        ks = "—" if pd.isna(row.ks) else num(row.ks, "{:.3f}")
        rows.append(
            f'<tr><td>{html.escape(row.feature)}</td>'
            f'<td class="num">{num(row.psi, "{:.3f}")}</td>'
            f'<td class="num">{ks}</td>'
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
        shown = "—" if value is None or pd.isna(value) else num(value, fmt)
        return (f'<tr><td>{label}</td><td class="num">{shown}</td>'
                f'<td class="num ref">{reference}</td></tr>')

    delta = (None if pd.isna(row["mae_min"])
             else row["mae_min"] - holdout["mae_min"])
    body = ['<table class="form"><thead><tr><th>métrica</th>'
            '<th class="num">mês</th><th class="num">holdout 2019</th>'
            "</tr></thead><tbody>",
            line("MAE (min)", row["mae_min"], "{:.2f}",
                 num(holdout["mae_min"])),
            line("RMSE (min)", row["rmse_min"], "{:.2f}",
                 num(holdout["rmse_min"])),
            line("erro mediano", row["median_ae_min"], "{:.2f}",
                 num(holdout["median_ae_min"])),
            line("viés (min)", row["bias_min"], "{:+.2f}", "0,00"),
            line("Δ MAE vs. holdout", delta, "{:+.2f}"),
            line("predições", row["n_predictions"], "{:,.0f}"),
            line("cobertura de feedback", coverage, "{:.1%}"),
            "</tbody></table>"]
    return _panel("2 · desempenho aferido", "".join(body) +
                  '<p class="foot">Viés positivo = o modelo superestima a '
                  "duração. Sem feedback do mês, tudo aqui fica em branco: é "
                  "a confirmação atrasada ainda em trânsito.</p>")


def br_month(month: str | None) -> str:
    """2020-05 -> 05/2020."""
    year, m = month.split("-")
    return f"{m}/{year}"


def _pill(status: str) -> str:
    return (f'<span class="pill" style="border-color:{STATUS_COLOR[status]};'
            f'color:{STATUS_COLOR[status]}">{STATUS_LABEL[status]}</span>')


def _trigger_row(name: str, status: str, acknowledged: str | None) -> str:
    """Uma linha de gatilho. Reconhecido, a pílula fica apagada e ganha a
    tarja do mês — o status BRUTO continua o mesmo, porque reconhecer não
    muda o que os números dizem, só quem está de plantão."""
    acked = acknowledged is not None
    tail = (f'<span class="ack-tag">reconhecido em '
            f"{br_month(acknowledged)}</span>" if acked else
            f'<span class="hint">{STATUS_READING[status]}</span>')
    return (f'<div class="trigger{" acked" if acked else ""}">'
            f'<span class="name">{TRIGGER_LABEL[name]}</span>'
            f"{_pill(status)}{tail}</div>")


def retrain(row: pd.Series, streak: int, rule: dict,
            mae_threshold: float) -> str:
    """Seção 3: o parecer de retreino — DOIS gatilhos (decisão #8).

    O de drift só olha inputs e está disponível de imediato; o de
    performance depende do feedback ter chegado. Eles aparecem separados
    porque discordam nas duas direções ao longo de 2020-2021, e cada
    discordância diz uma coisa diferente sobre o modelo.
    """
    listed = ", ".join(row["alert_features"]) or "nenhuma"
    mae = "—" if pd.isna(row["mae_min"]) else num(row["mae_min"], "{:.2f}")
    perf_rule = (f'MAE &gt; holdout +{rule["perf_tolerance"]:.0%} · '
                 f'M={rule["perf_m"]:.0f}')

    triggers = [(name, row[f"{name}_status"],
                 row[f"{name}_acknowledged_month"])
                for name in ("drift", "performance")]
    # O parecer geral só apaga quando TODO gatilho aceso está reconhecido —
    # basta um sem plantonista para o alarme continuar gritando.
    pending = [name for name, status, ack in triggers
               if status != "green" and ack is None]
    handled = row["status"] != "green" and not pending
    reading = ("ação assumida — alarme silenciado até o gatilho voltar a "
               "verde" if handled else
               f'{VERDICT_READING[row["status"]]} — o pior dos dois gatilhos')

    return _panel(
        "3 · parecer de retreino",
        f'<div class="verdict{" acked" if handled else ""}">'
        f'{_pill(row["status"])}'
        f'<span class="verdict-text">{reading}</span></div>'
        + "".join(_trigger_row(*trigger) for trigger in triggers)
        + '<table class="form"><tbody>'
        f'<tr><td>features em alerta</td>'
        f'<td class="num">{num(row["n_alert"], "{:,.0f}")}</td>'
        f'<td class="num ref"></td></tr>'
        f'<tr><td>quais</td><td class="num" colspan="2">'
        f"{html.escape(listed)}</td></tr>"
        f'<tr><td>streak (meses ≥ K)</td><td class="num">{streak}</td>'
        f'<td class="num ref"></td></tr>'
        f'<tr><td>MAE do mês</td><td class="num">{mae}</td>'
        f'<td class="num ref">limiar {num(mae_threshold, "{:.2f}")}</td></tr>'
        f'<tr><td>regra · drift</td><td class="num" colspan="2">'
        f'K={rule["k"]:.0f} · M={rule["m"]:.0f} · '
        f'PSI &gt; {num(rule["psi_alert"], "{:g}")}</td></tr>'
        f'<tr><td>regra · performance</td><td class="num" colspan="2">'
        f"{perf_rule}</td></tr>"
        "</tbody></table>"
        '<p class="foot">Vermelho = a condição do gatilho se manteve por M '
        "meses. O drift dá o alerta precoce; a performance confirma — e em "
        "out–dez/2021 ela contradiz o drift, que já tinha dado alta. "
        "Reconhecer silencia o episódio até ele se resolver, e não treina "
        "nada: retreinar está fora do escopo deste projeto.</p>")
