"""Treino do modelo de duração de corrida (green taxi, 2019).

Decisões deste script (ver CLAUDE.md para o racional do projeto):

- **Split temporal**, não aleatório: treina em jan–out/2019 e reserva
  nov–dez/2019 como holdout *intocado* para o evaluate.py — imita o deploy
  real (modelo velho encontra dados novos). Este script só reporta MAE/RMSE
  agregados do holdout como sanity check; a análise de resíduos (inclusive
  por trip_type) é papel do evaluate.py.
- **LightGBM com API nativa**: categóricas declaradas por nome sobre colunas
  int (sem dtype `category` do pandas — o mapeamento categoria→código do
  pandas é um risco de training-serving skew no predict da API).
- **NaN de haversine mantido** (~0,4%, zonas 264/265): LightGBM trata NaN
  nativamente e produção recebe essas zonas também — descartar no treino e
  encontrar em produção seria skew.
- **Bins de PSI congelados aqui**: calculados na distribuição de treino
  (jan–out/2019) e gravados no metadata.json com as proporções esperadas.
  A API de monitoramento só lê — nunca recalcula (régua não muda de tamanho).

Determinístico: seed fixa + parâmetros deterministic/force_row_wise do
LightGBM. Re-executável do zero.

Uso:
    python src/train.py [--out models/]
"""

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_download import month_range
from features import (
    DUR_MAX_MINUTES,
    DUR_MIN_MINUTES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_features,
    clean_duration_mask,
    compute_duration_minutes,
    load_zone_centroids,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "models"

SEED = 42

TRAIN_MONTHS = ("2019-01", "2019-10")
HOLDOUT_MONTHS = ("2019-11", "2019-12")

RAW_COLUMNS = ["lpep_pickup_datetime", "lpep_dropoff_datetime",
               "PULocationID", "DOLocationID", "trip_type"]

CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID", "trip_type"]

# Alvo em minutos com cauda direita mesmo após a limpeza (1–180 min);
# objetivo L1 otimiza o MAE direto e é menos refém da cauda que o L2.
LGB_PARAMS = {
    "objective": "regression_l1",
    "metric": ["l1", "rmse"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}
NUM_BOOST_ROUND = 500

PSI_N_BINS = 10
# Features de baixa cardinalidade viram bins categóricos (um bin por valor);
# só haversine_km é contínua de verdade e usa bins por quantil.
PSI_CATEGORICAL = [c for c in FEATURE_COLUMNS if c != "haversine_km"]


def load_months(start: str, end: str) -> pd.DataFrame:
    files = [RAW_DIR / f"green_tripdata_{m}.parquet" for m in month_range(start, end)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"parquets ausentes em data/raw: {missing}")
    return pd.concat(
        (pd.read_parquet(f, columns=RAW_COLUMNS) for f in files), ignore_index=True
    )


def prepare(df: pd.DataFrame, centroids: pd.DataFrame,
            start: str, end_exclusive: str) -> pd.DataFrame:
    """Limpeza + features. Devolve DataFrame com FEATURE_COLUMNS e o alvo.

    Filtros (justificados na EDA / notas de qualidade):
    - pickup dentro da janela esperada (os parquets contêm timestamps
      espúrios de 2008–2062);
    - duração em [DUR_MIN_MINUTES, DUR_MAX_MINUTES].
    Linhas com haversine NaN ficam (ver docstring do módulo).
    """
    pickup = df["lpep_pickup_datetime"]
    in_window = (pickup >= pd.Timestamp(start)) & (pickup < pd.Timestamp(end_exclusive))

    duration = compute_duration_minutes(df)
    keep = in_window & clean_duration_mask(duration)

    out = add_features(df.loc[keep], centroids)
    out[TARGET_COLUMN] = duration.loc[keep]
    return out[FEATURE_COLUMNS + [TARGET_COLUMN]].reset_index(drop=True)


def continuous_bin_spec(series: pd.Series, n_bins: int = PSI_N_BINS) -> dict:
    """Bins por quantil da distribuição de treino, com proporções esperadas.

    `edges` são os cortes internos (as pontas são abertas: -inf/+inf).
    NaN é um bin próprio (`nan_expected`) — em produção as zonas 264/265
    continuam chegando e precisam ser contadas, não ignoradas.
    As proporções somam 1 sobre todas as linhas (bins + NaN).
    """
    finite = series.dropna().to_numpy()
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1]))
    counts, _ = np.histogram(finite, bins=np.concatenate([[-np.inf], edges, [np.inf]]))
    return {
        "kind": "continuous",
        "edges": edges.tolist(),
        "expected": (counts / len(series)).tolist(),
        "nan_expected": float(series.isna().mean()),
    }


def categorical_bin_spec(series: pd.Series) -> dict:
    """Um bin por valor observado no treino, com proporções esperadas.

    Valores não vistos no treino (ex: zona nova) ficam a cargo do drift.py
    (bin "outros" com esperado zero + suavização epsilon).
    """
    counts = series.value_counts().sort_index()
    return {
        "kind": "categorical",
        "categories": [int(v) for v in counts.index],
        "expected": (counts / len(series)).tolist(),
    }


def psi_bin_specs(df: pd.DataFrame) -> dict:
    return {
        col: categorical_bin_spec(df[col]) if col in PSI_CATEGORICAL
        else continuous_bin_spec(df[col])
        for col in FEATURE_COLUMNS
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "mae_min": float(np.mean(np.abs(err))),
        "rmse_min": float(np.sqrt(np.mean(err ** 2))),
        "median_ae_min": float(np.median(np.abs(err))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="diretório de saída (model.pkl + metadata.json)")
    args = parser.parse_args()

    centroids = load_zone_centroids()

    print(f"carregando treino {TRAIN_MONTHS[0]}..{TRAIN_MONTHS[1]}")
    train_df = prepare(load_months(*TRAIN_MONTHS), centroids,
                       "2019-01-01", "2019-11-01")
    print(f"carregando holdout {HOLDOUT_MONTHS[0]}..{HOLDOUT_MONTHS[1]}")
    holdout_df = prepare(load_months(*HOLDOUT_MONTHS), centroids,
                         "2019-11-01", "2020-01-01")
    print(f"treino: {len(train_df):,} corridas | holdout: {len(holdout_df):,}")

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    dataset = lgb.Dataset(x_train, label=y_train,
                          categorical_feature=CATEGORICAL_FEATURES,
                          free_raw_data=False)
    booster = lgb.train(LGB_PARAMS, dataset, num_boost_round=NUM_BOOST_ROUND)

    holdout_pred = booster.predict(holdout_df[FEATURE_COLUMNS])
    holdout_metrics = metrics(holdout_df[TARGET_COLUMN].to_numpy(), holdout_pred)

    # Baseline ingênuo (mediana do treino) para dar escala ao MAE do modelo.
    baseline = float(y_train.median())
    baseline_mae = float(np.mean(np.abs(holdout_df[TARGET_COLUMN] - baseline)))

    importance = dict(zip(FEATURE_COLUMNS,
                          booster.feature_importance("gain").round(1).tolist()))

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(booster, f)

    metadata = {
        "model_version": "0.1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_library": f"lightgbm=={lgb.__version__}",
        "target": TARGET_COLUMN,
        "target_cleaning_minutes": [DUR_MIN_MINUTES, DUR_MAX_MINUTES],
        "features": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "training_window": list(TRAIN_MONTHS),
        "holdout_window": list(HOLDOUT_MONTHS),
        "n_train_rows": len(train_df),
        "n_holdout_rows": len(holdout_df),
        "hyperparameters": {**LGB_PARAMS, "num_boost_round": NUM_BOOST_ROUND},
        "holdout_metrics": holdout_metrics,
        "baseline_median_mae_min": baseline_mae,
        "feature_importance_gain": importance,
        # Régua congelada do PSI (decisão #6 do CLAUDE.md): distribuição de
        # treino jan–out/2019. Atenção ao interpretar o grupo de controle
        # (jan–fev/2020): a fração do bloco MNAR (trip_type=0) cresceu de 6%
        # para 21% ao longo de 2019 (ver EDA) — PSI elevado só em trip_type
        # no controle é deriva secular real, não bug de bins.
        "psi_bins": psi_bin_specs(x_train),
    }
    metadata_path = args.out / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"\nholdout ({HOLDOUT_MONTHS[0]}..{HOLDOUT_MONTHS[1]}):")
    for name, value in holdout_metrics.items():
        print(f"  {name:>15}: {value:.3f}")
    print(f"  baseline (mediana={baseline:.1f} min) MAE: {baseline_mae:.3f}")
    print("\nimportância (gain):")
    for name, value in sorted(importance.items(), key=lambda kv: -kv[1]):
        print(f"  {name:>17}: {value:,.0f}")
    print(f"\nsalvo: {model_path} e {metadata_path}")


if __name__ == "__main__":
    main()