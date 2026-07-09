"""Gera o dicionário de dados do green taxi (reports/data_dictionary.md).

Para cada coluna dos parquets de 2019 (período de treino), o relatório cruza:
- estatísticas observadas nos dados (dtype, NaNs, únicos, min/max/média,
  distribuição de valores para categóricos);
- a interpretação oficial da variável, segundo o data dictionary da TLC
  (data_dictionary_trip_records_green.pdf), embutida em COLUMN_DOCS;
- se a informação existe **no momento do embarque** — a flag que decide se a
  coluna pode ou não virar feature sem vazamento de dados.

Uso:
    python src/data_description.py [--start 2019-01] [--end 2019-12]
"""

import argparse
from pathlib import Path

import pandas as pd

from data_download import month_range

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "reports" / "data_dictionary.md"

# Interpretação oficial de cada coluna (TLC, data dictionary do green taxi).
# "at_pickup" marca se o valor já é conhecido no momento do embarque — só
# essas colunas são candidatas a feature; as demais são leakage.
COLUMN_DOCS: dict[str, dict] = {
    "VendorID": {
        "at_pickup": True,
        "interpretation": (
            "Código do fornecedor do sistema LPEP que gerou o registro. "
            "1 = Creative Mobile Technologies; 2 = VeriFone Inc. "
            "Metadado do equipamento, não da corrida — pouco útil como feature."
        ),
    },
    "lpep_pickup_datetime": {
        "at_pickup": True,
        "interpretation": (
            "Data/hora em que o taxímetro foi acionado (início da corrida). "
            "Base das features temporais: hora do dia, dia da semana, feriado. "
            "Junto com o dropoff, define o alvo `trip_duration`."
        ),
    },
    "lpep_dropoff_datetime": {
        "at_pickup": False,
        "interpretation": (
            "Data/hora em que o taxímetro foi desligado (fim da corrida). "
            "Só existe quando a corrida termina — usar como feature é leakage. "
            "Uso legítimo: derivar o alvo `trip_duration` no treino."
        ),
    },
    "store_and_fwd_flag": {
        "at_pickup": False,
        "interpretation": (
            "Y/N — indica se o registro ficou retido na memória do veículo "
            "('store and forward') por falta de conexão com o servidor antes "
            "do envio. Detalhe de telemetria, irrelevante para o modelo."
        ),
    },
    "RatecodeID": {
        "at_pickup": False,
        "interpretation": (
            "Código de tarifa **final** da corrida: 1 = Standard; 2 = JFK; "
            "3 = Newark; 4 = Nassau/Westchester; 5 = tarifa negociada; "
            "6 = corrida em grupo. Por ser o código final (pode mudar durante "
            "a corrida), tratamos como pós-corrida por segurança."
        ),
    },
    "PULocationID": {
        "at_pickup": True,
        "interpretation": (
            "Taxi zone de embarque (1–263, mapeável pelo shapefile/lookup). "
            "Conhecida no embarque. Origem do centroide usado na feature de "
            "distância haversine."
        ),
    },
    "DOLocationID": {
        "at_pickup": True,
        "interpretation": (
            "Taxi zone de desembarque informada como destino. No green taxi é "
            "registrada pelo sistema ao fim da corrida, mas em produção real o "
            "destino declarado existe no embarque — tratamos como disponível. "
            "Destino do centroide da feature haversine."
        ),
    },
    "passenger_count": {
        "at_pickup": True,
        "interpretation": (
            "Número de passageiros, digitado manualmente pelo motorista. "
            "Conhecido no embarque, mas ruidoso (erros de digitação, zeros)."
        ),
    },
    "trip_distance": {
        "at_pickup": False,
        "interpretation": (
            "⚠️ **Distância efetivamente percorrida** (milhas) registrada pelo "
            "taxímetro — só existe ao fim da corrida. É a armadilha de leakage "
            "central do projeto: **nunca usar como feature**. Substituto: "
            "haversine entre centroides das zonas (features.py)."
        ),
    },
    "fare_amount": {
        "at_pickup": False,
        "interpretation": (
            "Tarifa de tempo-e-distância calculada pelo taxímetro. "
            "Resultado da corrida — leakage."
        ),
    },
    "extra": {
        "at_pickup": False,
        "interpretation": (
            "Extras e sobretaxas diversas (inclui $0.50 e $1 de rush hour e "
            "noturno). Componente do valor final — leakage."
        ),
    },
    "mta_tax": {
        "at_pickup": False,
        "interpretation": "Taxa MTA de $0.50 acionada pela tarifa em uso. Leakage.",
    },
    "tip_amount": {
        "at_pickup": False,
        "interpretation": (
            "Gorjeta — só registra gorjetas em cartão; gorjetas em dinheiro "
            "não entram. Pós-corrida, leakage."
        ),
    },
    "tolls_amount": {
        "at_pickup": False,
        "interpretation": "Total de pedágios pagos na corrida. Pós-corrida, leakage.",
    },
    "ehail_fee": {
        "at_pickup": False,
        "interpretation": (
            "Taxa de e-hail (chamada por app). Praticamente 100% nula no "
            "período — candidata a descarte imediato."
        ),
    },
    "improvement_surcharge": {
        "at_pickup": False,
        "interpretation": (
            "Sobretaxa de melhoria de $0.30 cobrada na bandeirada (desde 2015). "
            "Componente do valor final — leakage."
        ),
    },
    "total_amount": {
        "at_pickup": False,
        "interpretation": (
            "Valor total cobrado (não inclui gorjetas em dinheiro). "
            "Resultado da corrida — leakage."
        ),
    },
    "payment_type": {
        "at_pickup": False,
        "interpretation": (
            "Forma de pagamento: 1 = cartão de crédito; 2 = dinheiro; "
            "3 = sem cobrança; 4 = disputa; 5 = desconhecido; 6 = anulada. "
            "Só se sabe ao pagar — pós-corrida."
        ),
    },
    "trip_type": {
        "at_pickup": True,
        "interpretation": (
            "1 = street-hail (parada na rua); 2 = dispatch (despachada). "
            "Definido no início da corrida; atribuído automaticamente, mas "
            "editável pelo motorista."
        ),
    },
    "congestion_surcharge": {
        "at_pickup": False,
        "interpretation": (
            "Sobretaxa de congestionamento (introduzida em 2019) aplicada "
            "conforme a corrida. Componente do valor final — leakage."
        ),
    },
}


def load_months(start: str, end: str) -> pd.DataFrame:
    files = [RAW_DIR / f"green_tripdata_{m}.parquet" for m in month_range(start, end)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"parquets ausentes em data/raw: {missing}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def describe_column(series: pd.Series) -> dict:
    stats = {
        "dtype": str(series.dtype),
        "n_nan": int(series.isna().sum()),
        "pct_nan": 100 * series.isna().mean(),
        "n_unique": int(series.nunique()),
    }
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        stats["min"] = series.min()
        stats["max"] = series.max()
        if pd.api.types.is_numeric_dtype(series):
            stats["mean"] = series.mean()
    return stats


def value_counts_table(series: pd.Series, max_values: int = 12) -> str | None:
    """Tabela markdown de distribuição, só para colunas de baixa cardinalidade."""
    if series.nunique() > max_values:
        return None
    counts = series.value_counts(dropna=False).sort_index()
    lines = ["| valor | corridas | % |", "|---|---|---|"]
    for value, count in counts.items():
        label = "NaN" if pd.isna(value) else str(value)
        lines.append(f"| {label} | {count:,} | {100 * count / len(series):.2f}% |")
    return "\n".join(lines)


# Achados de investigação manual sobre os dados de 2019, registrados aqui para
# entrarem no relatório gerado. Ver conversa/commits para a análise completa.
QUALITY_NOTES = """\
## Notas de qualidade (investigação sobre 2019)

**O bloco de NaNs idênticos é uma subpopulação, não ruído.** As mesmas
673.138 linhas (10,7% de 2019) têm NaN simultâneo em `store_and_fwd_flag`,
`RatecodeID`, `passenger_count`, `payment_type` e `trip_type`. O perfil delas
é distinto do resto:

| característica | bloco NaN | demais corridas |
|---|---|---|
| VendorID | 100% VeriFone (2) | mistura de 1 e 2 |
| distância mediana | 6,2 mi | 1,8 mi |
| tarifa mediana | $29,45 | $9,50 |
| duração mediana | 28 min | 11 min |
| pico de horário | 9h–13h | 15h–19h |
| congestion_surcharge | 100% NaN | ~10% NaN |
| fração do ano | cresce de 6% (jan) para 21% (dez) | — |

O padrão (sem dados de taxímetro/pagamento, corridas longas, diurnas, caras,
via um único vendor) indica corridas **despachadas/pré-arranjadas ingeridas
por um canal de reporte reduzido**, e não falha aleatória de coleta — é
missingness informativo (MNAR).

Implicações para o projeto:
- **Não descartar essas linhas às cegas**: `lpep_pickup_datetime`,
  `PULocationID` e `DOLocationID` (nossas features centrais) estão íntegros
  nelas, e a fração cresce ao longo do tempo — remover mudaria a distribuição
  de forma correlacionada com o tempo, contaminando o baseline do PSI.
- `passenger_count` NaN nessas linhas: tratar na engenharia de features
  (imputação ou categoria própria), ou descartar a feature.
- Outros achados menores: 273 corridas com pickup fora de 2019 (2008–2062,
  todas fora do bloco) → filtro de sanidade temporal na limpeza;
  `RatecodeID = 99` em 50 linhas (código não documentado); `VendorID = 5`
  em 9 linhas; ~0,3% de valores monetários negativos (estornos).
"""


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def build_report(df: pd.DataFrame, start: str, end: str) -> str:
    out = [
        "# Dicionário de dados — NYC Green Taxi",
        "",
        f"Gerado por `src/data_description.py` a partir de "
        f"`green_tripdata_{start}` a `green_tripdata_{end}` "
        f"({len(df):,} corridas, {df.shape[1]} colunas).",
        "",
        "Interpretações conforme o data dictionary oficial da TLC "
        "([fonte](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)).",
        "",
        "**Disponível no embarque?** é a coluna mais importante da tabela: "
        "só variáveis com ✅ podem virar features do modelo. As demais só "
        "existem depois que a corrida termina — usá-las é vazamento de dados "
        "(ver CLAUDE.md).",
        "",
        "## Visão geral",
        "",
        "| coluna | dtype | NaN | % NaN | únicos | no embarque? |",
        "|---|---|---|---|---|---|",
    ]

    details = ["", "## Detalhes por coluna", ""]
    documented = set(COLUMN_DOCS)

    for col in df.columns:
        series = df[col]
        stats = describe_column(series)
        doc = COLUMN_DOCS.get(col)
        flag = "—" if doc is None else ("✅" if doc["at_pickup"] else "❌")
        out.append(
            f"| `{col}` | {stats['dtype']} | {stats['n_nan']:,} "
            f"| {stats['pct_nan']:.2f}% | {stats['n_unique']:,} | {flag} |"
        )

        details.append(f"### `{col}`")
        details.append("")
        if doc is not None:
            details.append(doc["interpretation"])
        else:
            details.append(
                "⚠️ Coluna presente nos parquets mas ausente do data dictionary "
                "oficial da TLC — investigar antes de usar."
            )
        details.append("")
        range_bits = []
        for key in ("min", "max", "mean"):
            if key in stats:
                range_bits.append(f"{key} = {fmt(stats[key])}")
        summary = (
            f"dtype `{stats['dtype']}` · {stats['n_nan']:,} NaN "
            f"({stats['pct_nan']:.2f}%) · {stats['n_unique']:,} valores únicos"
        )
        if range_bits:
            summary += " · " + " · ".join(range_bits)
        details.append(summary)
        details.append("")
        vc = value_counts_table(series)
        if vc is not None:
            details.append(vc)
            details.append("")

    not_in_data = documented - set(df.columns)
    if not_in_data:
        details.append("## Colunas documentadas ausentes dos dados")
        details.append("")
        details.append(", ".join(f"`{c}`" for c in sorted(not_in_data)))
        details.append("")

    details.append(QUALITY_NOTES)

    return "\n".join(out + details)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01", help="primeiro mês (YYYY-MM)")
    parser.add_argument("--end", default="2019-12", help="último mês (YYYY-MM)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="arquivo de saída")
    args = parser.parse_args()

    df = load_months(args.start, args.end)
    report = build_report(df, args.start, args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"dicionário gerado em {args.out} ({len(df):,} corridas analisadas)")


if __name__ == "__main__":
    main()