"""Engenharia de features compartilhada entre treino e serving (API).

Regra de ouro do projeto: o código que gera features no treino e em produção
é ESTE módulo, importado dos dois lados — nunca uma reimplementação parecida
(evita training-serving skew).

Regra de leakage: só entram features disponíveis no momento do embarque.
`trip_distance` (taxímetro) nunca é usada; o substituto é a distância
haversine entre os centroides das zonas de origem e destino.

Os centroides são materializados uma única vez em
data/processed/zone_centroids.csv por este script (geopandas só é importado
nesse passo de build — o caminho de serving depende apenas de pandas/numpy).
O shapefile da TLC vem em EPSG:2263 (NY State Plane, pés); o centroide é
calculado nessa projeção e só depois reprojetado para lat/lon (EPSG:4326).

Uso (build da tabela de centroides):
    python src/features.py [--raw data/raw] [--out data/processed/zone_centroids.csv]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_CENTROIDS_PATH = ROOT / "data" / "processed" / "zone_centroids.csv"

# Cortes de limpeza do alvo, justificados na seção 1 do relatório de EDA.
# eda.py e train.py importam daqui — fonte única.
DUR_MIN_MINUTES = 1.0
DUR_MAX_MINUTES = 180.0

# Schema de features do modelo, na ordem esperada. train.py grava esta lista
# no metadata.json e a API valida contra ela.
FEATURE_COLUMNS = [
    "PULocationID",
    "DOLocationID",
    "trip_type",
    "haversine_km",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_holiday",
]

TARGET_COLUMN = "duration_min"

EARTH_RADIUS_KM = 6371.0088


def build_zone_centroids(raw_dir: Path = DEFAULT_RAW_DIR,
                         out_path: Path = DEFAULT_CENTROIDS_PATH) -> pd.DataFrame:
    """Lê o shapefile de taxi zones (dentro do zip) e materializa os
    centroides em lat/lon num CSV pequeno (~263 linhas).

    Zonas 264 (Unknown) e 265 (Outside NYC) não existem no shapefile e
    portanto ficam fora da tabela — corridas com essas zonas recebem
    haversine NaN em add_features (ver seção 6 do relatório de EDA).
    """
    import geopandas as gpd  # import tardio: dependência só do build

    shp_uri = f"zip://{raw_dir / 'taxi_zones.zip'}!taxi_zones/taxi_zones.shp"
    gdf = gpd.read_file(shp_uri)
    if gdf.crs is None or not gdf.crs.is_projected:
        raise ValueError(f"CRS inesperado no shapefile: {gdf.crs} (esperava projetado)")

    # Centroide na projeção nativa (EPSG:2263, em pés) e só então reprojetado;
    # centroide calculado direto em lat/lon é geometricamente errado.
    centroids = gdf.geometry.centroid.to_crs(epsg=4326)

    table = pd.DataFrame({
        "LocationID": gdf["LocationID"],
        "zone": gdf["zone"],
        "borough": gdf["borough"],
        "lat": centroids.y,
        "lon": centroids.x,
    }).sort_values("LocationID").reset_index(drop=True)

    if table["LocationID"].duplicated().any():
        raise ValueError("LocationID duplicado no shapefile")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    return table


def load_zone_centroids(path: Path = DEFAULT_CENTROIDS_PATH) -> pd.DataFrame:
    """Tabela de centroides indexada por LocationID (colunas lat/lon)."""
    return pd.read_csv(path).set_index("LocationID")


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância haversine em km. Aceita escalares ou arrays (vetorizado)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float))
                              for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    """Adiciona as FEATURE_COLUMNS a um DataFrame de corridas.

    Espera as colunas `lpep_pickup_datetime` (datetime), `PULocationID`,
    `DOLocationID` e `trip_type` (NaN permitido). Funciona igualmente para um
    mês inteiro (treino) ou uma única corrida (API) — mesmo código nos dois
    caminhos.

    Zonas sem centroide (264/265) resultam em haversine_km NaN; a decisão de
    descartar ou imputar fica com quem consome (train.py mantém — LightGBM
    trata NaN nativamente, e produção recebe essas zonas também).
    """
    out = df.copy()
    pickup = out["lpep_pickup_datetime"]

    pu = centroids.reindex(out["PULocationID"])
    do = centroids.reindex(out["DOLocationID"])
    out["haversine_km"] = haversine_km(
        pu["lat"].to_numpy(), pu["lon"].to_numpy(),
        do["lat"].to_numpy(), do["lon"].to_numpy(),
    )

    # trip_type: 1 = street-hail, 2 = despacho. NaN identifica o bloco MNAR
    # (corridas ingeridas por canal de reporte reduzido — ver notas de
    # qualidade da EDA): subpopulação com perfil próprio, então NaN vira a
    # categoria 0 em vez de imputação.
    out["trip_type"] = out["trip_type"].fillna(0).astype("int8")

    out["pickup_hour"] = pickup.dt.hour.astype("int8")
    out["pickup_dayofweek"] = pickup.dt.dayofweek.astype("int8")
    out["is_weekend"] = (out["pickup_dayofweek"] >= 5).astype("int8")

    holidays = USFederalHolidayCalendar().holidays(
        start=pickup.min().normalize(), end=pickup.max().normalize())
    out["is_holiday"] = pickup.dt.normalize().isin(holidays).astype("int8")

    return out


def compute_duration_minutes(df: pd.DataFrame) -> pd.Series:
    """Alvo: duração da corrida em minutos (dropoff − pickup)."""
    delta = df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    return delta.dt.total_seconds() / 60.0


def clean_duration_mask(duration_min: pd.Series) -> pd.Series:
    """Máscara dos cortes de limpeza do alvo (herdados da EDA)."""
    return (duration_min >= DUR_MIN_MINUTES) & (duration_min <= DUR_MAX_MINUTES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR,
                        help="diretório com taxi_zones.zip")
    parser.add_argument("--out", type=Path, default=DEFAULT_CENTROIDS_PATH,
                        help="CSV de saída com os centroides")
    args = parser.parse_args()

    table = build_zone_centroids(args.raw, args.out)
    print(f"{len(table)} centroides salvos em {args.out}")
    print(f"lat [{table['lat'].min():.4f}, {table['lat'].max():.4f}], "
          f"lon [{table['lon'].min():.4f}, {table['lon'].max():.4f}]")


if __name__ == "__main__":
    main()