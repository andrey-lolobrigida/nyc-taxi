"""Análise exploratória dos dados de treino (green taxi 2019).

Gera as imagens em reports/eda/ e o relatório programático eda_report.md.
Sem notebooks: o script é re-executável do zero e determinístico — nenhuma
etapa usa amostragem, todas as corridas de 2019 entram na análise.

A EDA olha só para 2019 (o mundo que o modelo vai aprender); a única exceção
é o gráfico de volume 2019–2021, contexto visual do drift que vem depois.

Uso:
    python src/eda.py [--out reports/eda]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from data_download import month_range

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "reports" / "eda"

# Cortes de limpeza do alvo, justificados na seção 1 do relatório.
# features.py e train.py devem herdar estes valores, não redefini-los.
DUR_MIN_MINUTES = 1.0
DUR_MAX_MINUTES = 180.0

# Paleta de referência do skill de dataviz (modo claro, já validada).
BLUE = "#2a78d6"        # série principal
BLUE_LIGHT = "#9ec5f4"  # banda interquartil / série secundária do mesmo dado
BLUE_XLIGHT = "#cde2fb"  # banda externa (P10–P90)
AQUA = "#1baf7a"        # segunda série categórica
INK = "#0b0b0b"
SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

PT_MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun",
             "jul", "ago", "set", "out", "nov", "dez"]
PT_WEEKDAYS = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]

COLUMNS = ["lpep_pickup_datetime", "lpep_dropoff_datetime",
           "PULocationID", "DOLocationID"]


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlesize": 11,
        "axes.labelcolor": SEC,
        "axes.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": SEC,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
    })


def month_axis(ax, interval: int = 1, year_on_jan: bool = False) -> None:
    """Ticks mensais com abreviações em português (locale-independente)."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))

    def fmt(x, _pos):
        d = mdates.num2date(x)
        label = PT_MONTHS[d.month - 1]
        if year_on_jan and d.month == 1:
            label += f"\n{d.year}"
        return label

    ax.xaxis.set_major_formatter(FuncFormatter(fmt))


def count_axis(ax, axis: str = "y") -> None:
    """Ticks de contagem legíveis (1.200 → '1,2k', 1e6 → '1M')."""

    def fmt(v, _pos):
        if v >= 1e6:
            return f"{v / 1e6:g}M"
        if v >= 1000:
            return f"{v / 1000:g}k"
        return f"{v:g}"

    getattr(ax, f"{axis}axis").set_major_formatter(FuncFormatter(fmt))


def load_2019() -> pd.DataFrame:
    files = [RAW_DIR / f"green_tripdata_{m}.parquet" for m in month_range("2019-01", "2019-12")]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"parquets ausentes em data/raw: {missing}")
    return pd.concat(
        (pd.read_parquet(f, columns=COLUMNS) for f in files), ignore_index=True
    )


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sanidade temporal + alvo + cortes de limpeza. Retorna (df limpo, stats)."""
    stats = {"n_total": len(df)}

    in_2019 = df["lpep_pickup_datetime"].dt.year == 2019
    stats["n_fora_2019"] = int((~in_2019).sum())
    df = df[in_2019].copy()

    dur = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    df["duration_min"] = dur
    stats["percentis"] = dur.quantile([0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999, 1])
    stats["n_nao_positiva"] = int((dur <= 0).sum())
    stats["n_acima_corte"] = int((dur > DUR_MAX_MINUTES).sum())

    mask = (dur >= DUR_MIN_MINUTES) & (dur <= DUR_MAX_MINUTES)
    stats["n_removidas"] = int((~mask).sum())
    stats["pct_removidas"] = 100 * (~mask).mean()
    df = df[mask].copy()
    stats["n_limpo"] = len(df)

    df["date"] = df["lpep_pickup_datetime"].dt.floor("D")
    df["hour"] = df["lpep_pickup_datetime"].dt.hour
    df["dow"] = df["lpep_pickup_datetime"].dt.dayofweek
    df["is_weekend"] = df["dow"] >= 5
    return df, stats


def fig_target_hist(dur_raw: pd.Series, dur_clean: pd.Series, out: Path) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    p995 = dur_raw.quantile(0.995)
    ax1.hist(dur_raw, bins=80, range=(0, p995), color=BLUE, rwidth=0.92)
    ax1.set_title(f"Bruto (recorte visual até P99,5 = {p995:.0f} min)")
    ax1.set_xlabel("duração (min)")
    ax1.set_ylabel("corridas")
    count_axis(ax1)

    ax2.hist(dur_clean, bins=90, range=(0, DUR_MAX_MINUTES), color=BLUE, rwidth=0.92)
    ax2.set_yscale("log")
    ax2.set_title(f"Após limpeza ({DUR_MIN_MINUTES:.0f}–{DUR_MAX_MINUTES:.0f} min), eixo y log")
    ax2.set_xlabel("duração (min)")
    count_axis(ax2)

    fig.suptitle("Distribuição do alvo: trip_duration (2019)",
                 color=INK, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    name = "01_target_hist.png"
    fig.savefig(out / name)
    plt.close(fig)
    return name


def fig_target_daily(df: pd.DataFrame, out: Path) -> str:
    q = df.groupby("date")["duration_min"].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()

    fig, ax = plt.subplots(figsize=(11.5, 4))
    ax.fill_between(q.index, q[0.10], q[0.90], color=BLUE_XLIGHT, lw=0, label="P10–P90")
    ax.fill_between(q.index, q[0.25], q[0.75], color=BLUE_LIGHT, lw=0, label="P25–P75")
    ax.plot(q.index, q[0.50], color=BLUE, lw=1.4, label="mediana")
    ax.set_title("Duração da corrida ao longo de 2019 — mediana diária e bandas de percentis")
    ax.set_ylabel("duração (min)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", ncols=3)
    month_axis(ax)

    name = "02_target_daily.png"
    fig.savefig(out / name)
    plt.close(fig)
    return name


def fig_duration_by_hour(df: pd.DataFrame, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for is_weekend, color, label in [(False, BLUE, "dia útil"), (True, AQUA, "fim de semana")]:
        sub = df[df["is_weekend"] == is_weekend]
        q = sub.groupby("hour")["duration_min"].quantile([0.25, 0.50, 0.75]).unstack()
        ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.15, lw=0)
        ax.plot(q.index, q[0.50], color=color, lw=2, label=label)
        ax.annotate(label, (q.index[-1], q[0.50].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=SEC)

    ax.set_title("Duração por hora do embarque — mediana e P25–P75 (2019)")
    ax.set_xlabel("hora do embarque")
    ax.set_ylabel("duração (min)")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(0, 23)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")

    name = "03_duration_by_hour.png"
    fig.savefig(out / name)
    plt.close(fig)
    return name


def _plot_daily_volume(ax, counts: pd.Series) -> None:
    smooth = counts.rolling(7, center=True).mean()
    ax.plot(counts.index, counts.values, color=BLUE_LIGHT, lw=0.9, label="corridas/dia")
    ax.plot(smooth.index, smooth.values, color=BLUE, lw=1.8, label="média móvel 7 dias")
    ax.set_ylabel("corridas por dia")
    ax.set_ylim(bottom=0)
    count_axis(ax)
    ax.legend(loc="upper right", ncols=2)


def fig_volume_2019(df: pd.DataFrame, out: Path) -> tuple[str, dict]:
    counts = df.groupby("date").size()
    counts = counts.reindex(pd.date_range("2019-01-01", "2019-12-31"), fill_value=0)

    monthly_mean = counts.resample("MS").mean()
    stats = {"queda_2019_pct": 100 * (1 - monthly_mean.iloc[-1] / monthly_mean.iloc[0])}

    fig, ax = plt.subplots(figsize=(11.5, 4))
    _plot_daily_volume(ax, counts)
    ax.set_title("Volume diário de corridas em 2019 (baseline de treino)")
    month_axis(ax)

    name = "04_volume_daily_2019.png"
    fig.savefig(out / name)
    plt.close(fig)
    return name, stats


def fig_volume_full(out: Path) -> tuple[str, dict]:
    """Volume diário 2019–2021: o contexto visual do drift da pandemia."""
    files = [RAW_DIR / f"green_tripdata_{m}.parquet" for m in month_range("2019-01", "2021-12")]
    pickups = pd.concat(
        (pd.read_parquet(f, columns=["lpep_pickup_datetime"]) for f in files),
        ignore_index=True,
    )["lpep_pickup_datetime"]
    pickups = pickups[(pickups >= "2019-01-01") & (pickups < "2022-01-01")]
    counts = pickups.dt.floor("D").value_counts().sort_index()
    counts = counts.reindex(pd.date_range("2019-01-01", "2021-12-31"), fill_value=0)

    monthly = counts.resample("MS").sum()
    stats = {
        "vol_fev2020": int(monthly.loc["2020-02-01"]),
        "vol_abr2020": int(monthly.loc["2020-04-01"]),
        "vol_dez2021": int(monthly.loc["2021-12-01"]),
    }
    stats["queda_pct"] = 100 * (1 - stats["vol_abr2020"] / stats["vol_fev2020"])

    fig, ax = plt.subplots(figsize=(11.5, 4))
    _plot_daily_volume(ax, counts)
    covid = pd.Timestamp("2020-03-01")
    ax.axvline(covid, color=SEC, lw=1, ls="--")
    ax.annotate("início da pandemia\n(mar/2020)", (covid, ax.get_ylim()[1] * 0.92),
                xytext=(8, 0), textcoords="offset points", fontsize=8, color=SEC)
    ax.annotate("treino (2019)", (pd.Timestamp("2019-06-15"), ax.get_ylim()[1] * 0.03),
                fontsize=8, color=MUTED, ha="center")
    ax.annotate("produção simulada (2020–21)", (pd.Timestamp("2021-01-01"), ax.get_ylim()[1] * 0.03),
                fontsize=8, color=MUTED, ha="center")
    ax.set_title("Volume diário 2019–2021 — o drift que o monitoramento vai medir")
    month_axis(ax, interval=3, year_on_jan=True)

    name = "05_volume_daily_2019_2021.png"
    fig.savefig(out / name)
    plt.close(fig)
    return name, stats


def fig_volume_by_hour_dow(df: pd.DataFrame, out: Path) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), width_ratios=(3, 2))

    by_hour = df.groupby("hour").size()
    ax1.bar(by_hour.index, by_hour.values, color=BLUE, width=0.82)
    ax1.set_title("Corridas por hora do embarque")
    ax1.set_xlabel("hora do embarque")
    ax1.set_ylabel("corridas")
    ax1.set_xticks(range(0, 24, 3))
    count_axis(ax1)

    by_dow = df.groupby("dow").size()
    ax2.bar(by_dow.index, by_dow.values, color=BLUE, width=0.82)
    ax2.set_title("Corridas por dia da semana")
    ax2.set_xticks(range(7), PT_WEEKDAYS)
    count_axis(ax2)

    fig.suptitle("Distribuição das features temporais candidatas (2019)",
                 color=INK, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    name = "06_volume_by_hour_dow.png"
    fig.savefig(out / name)
    plt.close(fig)
    return name


def fig_top_zones(df: pd.DataFrame, zone_names: pd.Series, out: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, col, title in [(axes[0], "PULocationID", "Top 15 zonas de embarque"),
                           (axes[1], "DOLocationID", "Top 15 zonas de desembarque")]:
        top = df[col].value_counts().head(15)
        labels = [zone_names.get(i, f"ID {i}") for i in top.index]
        ax.barh(range(len(top)), top.values, color=BLUE, height=0.82)
        ax.set_yticks(range(len(top)), labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("corridas")
        count_axis(ax, "x")
        ax.grid(axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", labelsize=7.5, colors=SEC)

    fig.tight_layout()
    name = "07_top_zones.png"
    fig.savefig(out / name)
    plt.close(fig)
    return name


def locationid_sanity(df: pd.DataFrame, lookup: pd.DataFrame) -> list[dict]:
    """Frequência das zonas 'Unknown' (264/265) e de IDs fora do lookup."""
    known = set(lookup["LocationID"])
    rows = []
    for col in ("PULocationID", "DOLocationID"):
        series = df[col]
        rows.append({
            "coluna": col,
            "n_264": int((series == 264).sum()),
            "n_265": int((series == 265).sum()),
            "n_fora_lookup": int((~series.isin(known)).sum()),
            "pct_desconhecida": 100 * series.isin([264, 265]).mean(),
        })
    return rows


def build_report(stats: dict, figures: dict, sanity: list[dict]) -> str:
    p = stats["percentis"]
    pct_labels = {0: "mín", 0.01: "P1", 0.05: "P5", 0.25: "P25", 0.50: "P50 (mediana)",
                  0.75: "P75", 0.95: "P95", 0.99: "P99", 0.999: "P99,9", 1: "máx"}
    pct_table = ["| percentil | duração (min) |", "|---|---|"]
    pct_table += [f"| {pct_labels[k]} | {v:,.1f} |" for k, v in p.items()]

    sanity_table = [
        "| coluna | zona 264 (Unknown) | zona 265 (Outside NYC) | fora do lookup | % desconhecida |",
        "|---|---|---|---|---|",
    ]
    sanity_table += [
        f"| `{r['coluna']}` | {r['n_264']:,} | {r['n_265']:,} "
        f"| {r['n_fora_lookup']:,} | {r['pct_desconhecida']:.2f}% |"
        for r in sanity
    ]

    return f"""# EDA — NYC Green Taxi 2019 (baseline de treino)

Gerado por `src/eda.py` a partir dos 12 parquets de 2019
({stats['n_total']:,} corridas brutas; {stats['n_limpo']:,} após limpeza).
Nenhuma etapa usa amostragem — o relatório cobre todas as corridas do ano.

## 1. Alvo: `trip_duration`

Definição: `lpep_dropoff_datetime − lpep_pickup_datetime`, em minutos.
Antes de qualquer corte, {stats['n_fora_2019']:,} corridas com pickup fora de
2019 (timestamps de 2008 a 2062) foram removidas por sanidade temporal.

Percentis da duração bruta:

{chr(10).join(pct_table)}

A cauda é extrema: {stats['n_nao_positiva']:,} corridas com duração ≤ 0
(taxímetro ligado/desligado no mesmo instante ou timestamps invertidos) e
{stats['n_acima_corte']:,} acima de {DUR_MAX_MINUTES:.0f} min (esquecimento de
desligar o taxímetro, não corridas reais).

**Cortes de limpeza adotados: {DUR_MIN_MINUTES:.0f} min ≤ duração ≤
{DUR_MAX_MINUTES:.0f} min**, removendo {stats['n_removidas']:,} corridas
({stats['pct_removidas']:.2f}% do ano). `features.py` e `train.py` devem
herdar estes valores (constantes `DUR_MIN_MINUTES`/`DUR_MAX_MINUTES`).

![Distribuição do alvo]({figures['hist']})

*À esquerda, a distribuição bruta (recorte visual no P99,5); à direita, a
distribuição limpa em escala log — assimetria à direita forte, mediana em
{p[0.5]:.1f} min. Confirma a escolha de mediana + quartis (e não média ± σ)
nas séries temporais abaixo.*

## 2. Alvo ao longo do tempo

![Série diária do alvo]({figures['daily']})

*Mediana diária com bandas P25–P75 e P10–P90 (dados limpos). A banda é
estável ao longo do ano — este é o "mundo normal" que o modelo vai aprender
e que o PSI vai usar como régua.*

## 3. Perfil por hora do dia

![Duração por hora]({figures['by_hour']})

*Em dias úteis a mediana sobe nos horários de pico e a banda alarga; no fim
de semana o perfil é mais chato. Esta é exatamente a relação features→alvo
que o concept drift da pandemia deve quebrar (ruas vazias → corridas mais
rápidas nos mesmos horários). O pico das 6h é em parte **efeito de
composição**, não só trânsito: a subpopulação de corridas despachadas (o
bloco MNAR do dicionário de dados) chega a ~25% das corridas na madrugada,
com medianas de ~30 min — excluindo-a, a mediana das 6h cai de ~20 para
~15 min.*

## 4. Volume de corridas

![Volume diário 2019]({figures['vol_2019']})

*Sazonalidade semanal clara (dia útil vs. fim de semana), mas o baseline
**não é estacionário**: o volume médio diário cai
{stats['queda_2019_pct']:.0f}% entre jan e dez de 2019 — o green taxi já
vinha encolhendo antes da pandemia (competição dos apps de transporte).
Implicação para o monitoramento: alguma deriva de volume é o estado normal
deste serviço; o que a pandemia adiciona é uma mudança de outra ordem de
grandeza, e é isso que os thresholds de PSI precisam separar.*

![Volume diário 2019–2021]({figures['vol_full']})

*A única espiada além de 2019: o penhasco de mar/2020. O volume mensal cai de
{stats['vol_fev2020']:,} corridas (fev/2020) para {stats['vol_abr2020']:,}
(abr/2020) — queda de {stats['queda_pct']:.0f}% — e termina 2021 em
{stats['vol_dez2021']:,}, sem recuperar o patamar pré-pandemia. É este evento
que as fases 2–3 vão detectar via PSI/KS sem olhar este gráfico.*

## 5. Features temporais candidatas

![Volume por hora e dia da semana]({figures['hour_dow']})

*Distribuições que os bins congelados do PSI vão medir: pico vespertino
(15h–19h) e volume maior em dias úteis.*

![Top zonas]({figures['zones']})

*Concentração geográfica alta (green taxi atende principalmente norte de
Manhattan, Brooklyn e Queens). PU/DO LocationID alimentam a feature de
distância haversine entre centroides (features.py).*

## 6. Sanidade dos LocationIDs

{chr(10).join(sanity_table)}

Zonas 264/265 não têm centroide no shapefile → a feature haversine será NaN
nessas corridas. A fração é pequena; decidir em `features.py` entre descartar
as linhas ou imputar (ex.: mediana da distância).

## 7. Implicações para as próximas fases

- **Limpeza herdada**: sanidade temporal (pickup dentro do período) +
  duração em [{DUR_MIN_MINUTES:.0f}, {DUR_MAX_MINUTES:.0f}] min.
- **Alvo assimétrico**: avaliar treino sobre `log(duration)` em `train.py`;
  reportar MAE/RMSE na escala original em `evaluate.py`.
- **Bins do PSI**: hora do dia e dia da semana têm distribuições bem
  comportadas; os bins congelados devem vir da distribuição 2019 completa
  (ver regra no CLAUDE.md).
- **Correlação distância↔duração**: pendente até `features.py` materializar o
  haversine entre centroides — a EDA dela entra no relatório de features.
- **Zonas 264/265**: tratar explicitamente em `features.py` (NaN de
  haversine).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="diretório de saída")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    setup_style()

    print("carregando 2019...")
    df_raw = load_2019()
    dur_raw = (
        (df_raw["lpep_dropoff_datetime"] - df_raw["lpep_pickup_datetime"]).dt.total_seconds() / 60
    )
    df, stats = prepare(df_raw)
    print(f"  {stats['n_total']:,} corridas brutas → {stats['n_limpo']:,} limpas")

    lookup = pd.read_csv(RAW_DIR / "taxi_zone_lookup.csv")
    zone_names = pd.Series(
        (lookup["Zone"] + " (" + lookup["Borough"] + ")").values,
        index=lookup["LocationID"].values,
    )

    figures = {
        "hist": fig_target_hist(dur_raw, df["duration_min"], args.out),
        "daily": fig_target_daily(df, args.out),
        "by_hour": fig_duration_by_hour(df, args.out),
        "hour_dow": fig_volume_by_hour_dow(df, args.out),
        "zones": fig_top_zones(df, zone_names, args.out),
    }
    figures["vol_2019"], vol_2019_stats = fig_volume_2019(df, args.out)
    stats.update(vol_2019_stats)
    print("gerando volume 2019–2021 (lê os 36 parquets)...")
    figures["vol_full"], vol_stats = fig_volume_full(args.out)
    stats.update(vol_stats)

    sanity = locationid_sanity(df, lookup)

    report_path = args.out / "eda_report.md"
    report_path.write_text(build_report(stats, figures, sanity), encoding="utf-8")
    print(f"relatório gerado em {report_path} + {len(figures)} imagens")


if __name__ == "__main__":
    main()
