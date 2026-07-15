"""Validação final do modelo no holdout intocado (nov–dez/2019).

Gera reports/evaluation/: figuras + evaluation_report.md programático.

Papel deste script (complementar ao train.py, que só reporta agregados):
- métricas gerais vs. baseline ingênuo;
- **resíduos quebrados por trip_type** — o critério real da decisão
  "um modelo com a feature vs. dois modelos" (nota em aberto do CLAUDE.md).
  Tamanhos de efeito, não p-valores;
- erro por hora do dia, por distância e calibração por decil de predição.

Reusa o pipeline de dados do train.py (prepare/load_months) e o estilo de
gráficos do eda.py — mesmo código, nunca reimplementação parecida.

Uso:
    python src/evaluate.py [--out reports/evaluation]
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda import AQUA, BLUE, GRID, MUTED, SEC, setup_style
from features import FEATURE_COLUMNS, TARGET_COLUMN
from train import HOLDOUT_MONTHS, load_months, metrics, prepare
from features import load_zone_centroids

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DEFAULT_OUT = ROOT / "reports" / "evaluation"

TRIP_TYPE_LABELS = {
    0: "bloco MNAR (canal reduzido)",
    1: "street-hail",
    2: "despachada",
}


def load_model_and_metadata() -> tuple:
    with open(MODELS_DIR / "model.pkl", "rb") as f:
        booster = pickle.load(f)
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
    if metadata["features"] != FEATURE_COLUMNS:
        raise ValueError(
            "schema do metadata difere de features.FEATURE_COLUMNS — "
            "modelo e código fora de sincronia; re-treine antes de avaliar")
    return booster, metadata


def group_table(df: pd.DataFrame) -> pd.DataFrame:
    """Métricas de resíduo por grupo de trip_type (tamanhos de efeito)."""
    rows = []
    for value, group in df.groupby("trip_type"):
        res = group["residual_min"]
        rows.append({
            "trip_type": int(value),
            "grupo": TRIP_TYPE_LABELS[int(value)],
            "n": len(group),
            "share": len(group) / len(df),
            "dur_mediana_min": group[TARGET_COLUMN].median(),
            "vies_min": res.mean(),
            "mae_min": res.abs().mean(),
            "ae_mediano_min": res.abs().median(),
        })
    return pd.DataFrame(rows)


def fig_residual_hist(df: pd.DataFrame, path: Path) -> None:
    res = df["residual_min"]
    outside = (res.abs() > 30).mean()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(res[res.abs() <= 30], bins=121, color=BLUE)
    ax.axvline(0, color=SEC, linewidth=0.8)
    med = res.median()
    ax.axvline(med, color=SEC, linewidth=0.8, linestyle="--")
    ax.annotate(f"mediana {med:+.1f} min", xy=(med, 1), xycoords=("data", "axes fraction"),
                xytext=(6, -12), textcoords="offset points", fontsize=8, color=SEC)
    ax.set_title("Resíduos no holdout (previsto − real)")
    ax.set_xlabel(f"erro (min) — {outside:.1%} das corridas fora de ±30 (omitidas)")
    ax.set_ylabel("corridas")
    fig.savefig(path)
    plt.close(fig)


def fig_mae_by_hour(df: pd.DataFrame, path: Path) -> None:
    by_hour = df.groupby("pickup_hour")["residual_min"].agg(
        mae=lambda r: r.abs().mean(), vies="mean")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(by_hour.index, by_hour["mae"], color=BLUE, linewidth=2,
            marker="o", markersize=4, label="MAE")
    ax.plot(by_hour.index, by_hour["vies"], color=AQUA, linewidth=2,
            marker="o", markersize=4, label="viés (erro médio)")
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Erro por hora do dia")
    ax.set_xlabel("hora do embarque")
    ax.set_ylabel("minutos")
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def fig_mae_by_distance(df: pd.DataFrame, path: Path) -> None:
    known = df[df["haversine_km"].notna()].copy()
    known["decil"] = pd.qcut(known["haversine_km"], 10, duplicates="drop")
    by_bin = known.groupby("decil", observed=True).agg(
        mae=("residual_min", lambda r: r.abs().mean()),
        upper_km=("haversine_km", "max"))
    labels = [f"≤{u:.1f}" for u in by_bin["upper_km"]]

    nan_mae = df.loc[df["haversine_km"].isna(), "residual_min"].abs().mean()
    values = list(by_bin["mae"]) + [nan_mae]
    labels.append("zona s/\ncentroide")
    colors = [BLUE] * len(by_bin) + [AQUA]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(range(len(values)), values, color=colors, width=0.72)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.1f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.5, color=SEC)
    ax.set_xticks(range(len(values)), labels, fontsize=7.5)
    ax.set_title("MAE por decil de distância (haversine, km)")
    ax.set_xlabel("decil de distância entre centroides")
    ax.set_ylabel("MAE (min)")
    fig.savefig(path)
    plt.close(fig)


def fig_calibration(df: pd.DataFrame, path: Path) -> None:
    df = df.copy()
    df["decil"] = pd.qcut(df["pred_min"], 10, duplicates="drop")
    by_bin = df.groupby("decil", observed=True).agg(
        pred=("pred_min", "mean"), real=(TARGET_COLUMN, "mean"))
    lim = (0, by_bin.to_numpy().max() * 1.1)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot(lim, lim, color=GRID, linewidth=1)
    ax.plot(by_bin["pred"], by_bin["real"], color=BLUE, linewidth=2,
            marker="o", markersize=5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title("Calibração por decil de predição")
    ax.set_xlabel("duração prevista média (min)")
    ax.set_ylabel("duração real média (min)")
    fig.savefig(path)
    plt.close(fig)


def md_table(df: pd.DataFrame, formats: dict) -> str:
    header = "| " + " | ".join(formats) + " |"
    sep = "|" + "|".join("---" for _ in formats) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col, fmt in formats.items()) + " |")
    return "\n".join(lines)


def build_report(metadata: dict, overall: dict, baseline_mae: float,
                 groups: pd.DataFrame, df: pd.DataFrame, figures: dict) -> str:
    res = df["residual_min"]
    p90 = res.abs().quantile(0.9)
    frac_5min = (res.abs() <= 5).mean()
    nan_share = df["haversine_km"].isna().mean()
    nan_mae = res[df["haversine_km"].isna()].abs().mean()

    g = groups.set_index("trip_type")
    rel = g["mae_min"] / g["dur_mediana_min"]  # erro relativo à duração típica
    extra_bias_2 = g.loc[2, "vies_min"] - g.loc[1, "vies_min"]

    group_md = md_table(groups, {
        "grupo": str,
        "n": lambda v: f"{v:,}",
        "share": lambda v: f"{v:.1%}",
        "dur_mediana_min": lambda v: f"{v:.1f}",
        "vies_min": lambda v: f"{v:+.2f}",
        "mae_min": lambda v: f"{v:.2f}",
        "ae_mediano_min": lambda v: f"{v:.2f}",
    })

    return f"""# Validação do modelo — holdout {metadata['holdout_window'][0]} a {metadata['holdout_window'][1]}

Gerado programaticamente por `src/evaluate.py`. Modelo
`{metadata['model_version']}` ({metadata['model_library']}), treinado em
{metadata['training_window'][0]}..{metadata['training_window'][1]}
({metadata['n_train_rows']:,} corridas). Holdout **intocado no treino**:
{len(df):,} corridas.

## 1. Métricas gerais

| métrica | modelo | baseline (mediana do treino) |
|---|---|---|
| MAE | {overall['mae_min']:.2f} min | {baseline_mae:.2f} min |
| RMSE | {overall['rmse_min']:.2f} min | — |
| erro absoluto mediano | {overall['median_ae_min']:.2f} min | — |
| P90 do erro absoluto | {p90:.2f} min | — |
| corridas com erro ≤ 5 min | {frac_5min:.1%} | — |

## 2. Distribuição dos resíduos

![Resíduos]({figures['fig_hist']})

*Mediana do resíduo em {res.median():+.1f} min (o objetivo L1 mira a mediana
condicional), mas média em {res.mean():+.1f} min: a cauda esquerda é mais
pesada — quando o modelo erra muito, é mais frequentemente por **subestimar**
a duração.*

## 3. Resíduos por grupo (trip_type) — critério da decisão "um modelo ou dois"

{group_md}

*O viés dos dois grupos grandes é praticamente o mesmo
({g.loc[1, 'vies_min']:+.2f} min no street-hail vs
{g.loc[0, 'vies_min']:+.2f} no bloco MNAR): a subestimação é compartilhada
entre grupos, não específica de um deles. Em erro **relativo à duração
típica**, o bloco MNAR ({g.loc[0, 'mae_min']:.2f} min de MAE sobre mediana de
{g.loc[0, 'dur_mediana_min']:.0f} min ≈ {rel.loc[0]:.0%}) sai melhor que o
street-hail ({rel.loc[1]:.0%}) — o modelo único **com** a feature não falha
sistematicamente no grupo grande.*

*A exceção é a despachada explícita (trip_type=2): viés de
{g.loc[2, 'vies_min']:+.2f} min, {abs(extra_bias_2):.1f} min além do viés
compartilhado — subestimação sistemática real, mas num grupo com
{g.loc[2, 'share']:.1%} do holdout.*

***Veredito: um modelo único com a feature fica.** Dois modelos não se
justificam por {g.loc[2, 'share']:.1%} do tráfego; o ponto vira item de
monitoramento — se a fração de despachadas crescer (o PSI de trip_type vigia
exatamente essa mistura), a decisão é revisitada.*

## 4. Erro por hora do dia

![Erro por hora]({figures['fig_hour']})

*O MAE acompanha os picos de trânsito (7h e 16–17h). O viés só afunda de
verdade no pico da tarde (~−2 min às 16–17h): é o horário em que o modelo
mais subestima; no pico da manhã o erro cresce sem viés extra.*

## 5. Erro por distância

![MAE por distância]({figures['fig_dist']})

*O erro cresce monotonicamente com a distância. Corridas sem centroide
(zonas 264/265, haversine NaN; {nan_share:.1%} do holdout) têm MAE de
{nan_mae:.1f} min — o modelo degrada sem a feature principal, mas continua
respondendo; aceitável enquanto o grupo for marginal, e é mais um item que o
monitoramento de inputs enxerga (bin de NaN no PSI).*

## 6. Calibração

![Calibração]({figures['fig_calib']})

*Os decis andam paralelos à diagonal com deslocamento quase constante de
~{-res.mean():.0f} min (real acima do previsto): o desvio é aditivo e
uniforme, não um problema de inclinação. Hipótese (não testada aqui):
nov–dez é mais lento que a média de jan–out por sazonalidade — o replay de
2020–2021 dirá se o padrão persiste fora da estação.*
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()

    booster, metadata = load_model_and_metadata()
    centroids = load_zone_centroids()

    start, end = metadata["holdout_window"]
    df = prepare(load_months(start, end), centroids, "2019-11-01", "2020-01-01")
    df["pred_min"] = booster.predict(df[FEATURE_COLUMNS])
    df["residual_min"] = df["pred_min"] - df[TARGET_COLUMN]

    overall = metrics(df[TARGET_COLUMN].to_numpy(), df["pred_min"].to_numpy())
    groups = group_table(df)

    figures = {
        "fig_hist": "01_residual_hist.png",
        "fig_hour": "02_error_by_hour.png",
        "fig_dist": "03_mae_by_distance.png",
        "fig_calib": "04_calibration.png",
    }
    fig_residual_hist(df, args.out / figures["fig_hist"])
    fig_mae_by_hour(df, args.out / figures["fig_hour"])
    fig_mae_by_distance(df, args.out / figures["fig_dist"])
    fig_calibration(df, args.out / figures["fig_calib"])

    report = build_report(metadata, overall, metadata["baseline_median_mae_min"],
                          groups, df, figures)
    (args.out / "evaluation_report.md").write_text(report)

    print(f"relatório e {len(figures)} figuras salvos em {args.out}")
    print("\nresíduos por grupo:")
    print(groups.to_string(index=False))
    print("\ngeral:", {k: round(v, 3) for k, v in overall.items()})


if __name__ == "__main__":
    main()