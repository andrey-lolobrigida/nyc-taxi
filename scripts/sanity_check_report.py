"""Relatório persistido do sanity check do grupo de controle temporal.

Materializa em reports/monitoring/sanity_check_report.md a evidência de que
a régua de monitoramento é confiável: se o PSI estivesse alto em jan/fev
2020 (ainda "mundo normal"), o bug estaria na implementação (provavelmente
bins), não nos dados — e nenhum número da pandemia seria interpretável.

Fonte dos números: o SQLite alimentado pelo replay, lido VIA API
(/monitoring/*) — nunca queries diretas ao banco. Motivo: o relatório tem
que mostrar exatamente os mesmos números que o dashboard, e consumir os
endpoints reutiliza a implementação única (drift.py + monitoring.py) em vez
de reimplementar queries parecidas (mesma classe de bug do
training-serving skew).

Única exceção: a série 2019 de trip_type ausente vem dos parquets crus com
a limpeza do replay (load_month_trips) — o período de treino nunca passa
pelo /predict, então não existe no banco.

Uso (com a API de pé e os meses de controle já reproduzidos):
    uv run python scripts/sanity_check_report.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from simulate_production import api_call, load_month_trips

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "reports" / "monitoring" / "sanity_check_report.md"

CONTROL_MONTHS = ["2020-01", "2020-02"]
TREND_MONTHS = [f"2019-{m:02d}" for m in range(1, 13)] + CONTROL_MONTHS

LEVEL_LABEL = {"stable": "estável", "watch": "atenção", "alert": "alerta"}
STATUS_LABEL = {"green": "🟢 verde", "yellow": "🟡 amarelo", "red": "🔴 vermelho"}


def fetch(api_url: str, path: str) -> dict:
    status, body = api_call("GET", f"{api_url}{path}")
    if status != 200:
        sys.exit(f"GET {path} -> {status}: {body.get('detail')} "
                 "(os meses de controle já foram reproduzidos pelo replay?)")
    return body


def missing_trip_type_share(month: str) -> float:
    """Fração de trip_type ausente no mês, após a limpeza do replay.

    A amostragem do replay é estratificada e preserva proporções, então a
    fração do mês inteiro é comparável à do que foi logado no banco."""
    trips = load_month_trips(month)
    return float(trips["trip_type"].isna().mean())


def drift_table(drift_by_month: dict) -> list[str]:
    months = list(drift_by_month)
    lines = ["| Feature | " + " | ".join(f"PSI {m}" for m in months)
             + " | Leitura |",
             "|---|" + "---|" * (len(months) + 1)]
    features = drift_by_month[months[0]]["features"]
    for name in features:
        cells, worst = [], "stable"
        for month in months:
            entry = drift_by_month[month]["features"][name]
            cells.append(f"{entry['psi']:.4f}")
            if entry["level"] == "alert" or (entry["level"] == "watch"
                                             and worst == "stable"):
                worst = entry["level"]
        lines.append(f"| `{name}` | " + " | ".join(cells)
                     + f" | {LEVEL_LABEL[worst]} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000",
                        help="[%(default)s]")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="[%(default)s]")
    args = parser.parse_args()

    health = fetch(args.api_url, "/health")
    drift = {m: fetch(args.api_url, f"/monitoring/drift?month={m}")
             for m in CONTROL_MONTHS}
    perf = {m: fetch(args.api_url, f"/monitoring/performance?month={m}")
            for m in CONTROL_MONTHS}
    check = {m: fetch(args.api_url, f"/monitoring/retrain-check?month={m}")
             for m in CONTROL_MONTHS}
    thresholds = drift[CONTROL_MONTHS[0]]["thresholds"]
    rule = check[CONTROL_MONTHS[-1]]["rule"]

    metadata = json.loads((ROOT / "models" / "metadata.json").read_text())
    trained_share = dict(zip(metadata["psi_bins"]["trip_type"]["categories"],
                             metadata["psi_bins"]["trip_type"]["expected"]))[0]
    trend = {m: missing_trip_type_share(m) for m in TREND_MONTHS}

    # Critério do veredito: toda feature < 0.1 nos dois meses, exceto
    # trip_type (deriva secular real, tratada em seção própria).
    offenders = sorted({
        name
        for month in CONTROL_MONTHS
        for name, entry in drift[month]["features"].items()
        if name != "trip_type" and entry["psi"] >= thresholds["watch"]
    })
    passed = not offenders

    ks_cells = {m: drift[m]["features"]["haversine_km"]["ks"]
                for m in CONTROL_MONTHS}

    lines = [
        "# Sanity check do monitoramento — grupo de controle temporal",
        "",
        f"Gerado programaticamente por `scripts/sanity_check_report.py` em "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} — "
        f"modelo `{health['model_version']}` "
        f"(treinado em {health['trained_at']}).",
        "",
        "## Por que um grupo de controle",
        "",
        "Os bins de PSI foram congelados na distribuição de treino (2019) e",
        "nunca recalculados. Janeiro e fevereiro de 2020 ainda são \"mundo",
        "normal\": se o PSI aparecesse alto neles, a causa provável seria bug",
        "na implementação (bins, alinhamento de categorias, contagem de NaN),",
        "não drift — e nenhum número da pandemia seria confiável. Este",
        "relatório materializa esse teste.",
        "",
        "## Setup do replay",
        "",
    ]
    for month in CONTROL_MONTHS:
        p = perf[month]
        lines.append(f"- **{month}**: {p['n_predictions']} predições "
                     f"(amostra estratificada por dia, seed fixa), feedback "
                     f"em {p['n_with_feedback']} ({p['n_with_feedback'] / p['n_predictions']:.0%}).")
    lines += [
        "",
        "## Drift por feature (PSI, bins congelados de 2019)",
        "",
        f"Thresholds: < {thresholds['watch']} estável · "
        f"{thresholds['watch']}–{thresholds['alert']} atenção · "
        f"> {thresholds['alert']} alerta.",
        "",
        *drift_table(drift),
        "",
        f"KS binado de `haversine_km` vs. treino: "
        + " · ".join(f"{m}: {ks_cells[m]:.3f}" for m in CONTROL_MONTHS)
        + ".",
        "",
        "## Performance vs. holdout de 2019",
        "",
        "| Métrica (min) | Holdout 2019 | "
        + " | ".join(CONTROL_MONTHS) + " |",
        "|---|---|---|---|",
    ]
    holdout = perf[CONTROL_MONTHS[0]]["holdout_reference"]
    for key, label in [("mae_min", "MAE"), ("rmse_min", "RMSE"),
                       ("median_ae_min", "AE mediano")]:
        lines.append(f"| {label} | {holdout[key]:.2f} | "
                     + " | ".join(f"{perf[m][key]:.2f}"
                                  for m in CONTROL_MONTHS) + " |")
    lines.append("| Viés (pred − real) | — | "
                 + " | ".join(f"{perf[m]['bias_min']:+.2f}"
                              for m in CONTROL_MONTHS) + " |")
    lines += [
        "",
        "O erro nos meses de controle fica no nível do holdout — o modelo",
        "ainda opera no regime em que foi treinado.",
        "",
        "## Gatilho de retreino",
        "",
        f"Regra v1: ≥ {rule['k']:.0f} features com PSI > {rule['psi_alert']}"
        f" por {rule['m']:.0f} meses consecutivos.",
        "",
    ]
    for month in CONTROL_MONTHS:
        c = check[month]
        alert_feats = c["months"][-1]["alert_features"]
        detail = f" (em alerta: {', '.join(f'`{f}`' for f in alert_feats)})" \
            if alert_feats else ""
        lines.append(f"- **{month}**: {STATUS_LABEL[c['status']]}{detail}")
    lines += [
        "",
        "## O caso `trip_type`: drift real dentro do controle",
        "",
        f"`trip_type` é a exceção da tabela: PSI "
        f"{drift[CONTROL_MONTHS[0]]['features']['trip_type']['psi']:.3f} em "
        f"{CONTROL_MONTHS[0]}. Não é bug de bins — é deriva anterior à",
        "pandemia, visível nos dados de 2019 (fração de corridas SEM",
        "`trip_type`, a categoria 0/bloco MNAR, após a limpeza do replay):",
        "",
        "| Mês | trip_type ausente |",
        "|---|---|",
        *(f"| {month} | {share:.1%} |" for month, share in trend.items()),
        "",
        f"O treino congelou a média de 2019 ({trained_share:.1%}) como",
        "esperado da categoria 0; a série acima mostra que a fração já",
        "crescia de forma aproximadamente monotônica ao longo do próprio ano",
        "de treino — o PSI está acusando uma mudança verdadeira, não um erro",
        "de régua. A causa (mudança no canal de reporte/ingestão dos",
        "registros) é **hipótese**, não verificável nestes dados; o fato",
        "verificável é a tendência.",
        "",
        "Dois pontos que isso demonstra a favor da estratégia:",
        "",
        "1. o monitor detecta mudança real de distribuição mesmo fora de um",
        "   evento dramático;",
        f"2. a regra K={rule['k']:.0f}/M={rule['m']:.0f} segura o alarme: uma"
        " feature isolada em",
        "   alerta não dispara retreino (status verde nos dois meses).",
        "",
        "## Veredito",
        "",
    ]
    if passed:
        lines += [
            f"**Sanity check aprovado.** Todas as features, exceto o caso",
            f"documentado de `trip_type`, ficaram abaixo de "
            f"{thresholds['watch']} nos dois meses de controle. A régua",
            "(bins congelados + PSI/KS) é confiável e os números de",
            "2020-03 em diante podem ser interpretados como drift dos dados,",
            "não artefato da implementação.",
        ]
    else:
        lines += [
            f"**Sanity check reprovado.** Features fora de `trip_type` com "
            f"PSI ≥ {thresholds['watch']} no controle: "
            + ", ".join(f"`{f}`" for f in offenders) + ".",
            "Investigar a implementação (bins, alinhamento, NaN) antes de",
            "interpretar qualquer mês de pandemia.",
        ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"relatório escrito em {args.out.relative_to(ROOT)}"
          f" (veredito: {'aprovado' if passed else 'REPROVADO'})")


if __name__ == "__main__":
    main()