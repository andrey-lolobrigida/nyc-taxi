"""Testes de features.py — haversine, features temporais e limpeza do alvo."""

import numpy as np
import pandas as pd
import pytest

from features import (
    DEFAULT_RAW_DIR,
    DUR_MAX_MINUTES,
    DUR_MIN_MINUTES,
    FEATURE_COLUMNS,
    add_features,
    build_zone_centroids,
    clean_duration_mask,
    compute_duration_minutes,
    haversine_km,
)


class TestHaversine:
    def test_distancia_zero_no_mesmo_ponto(self):
        assert haversine_km(40.75, -73.98, 40.75, -73.98) == 0.0

    def test_distancia_conhecida_times_square_jfk(self):
        # Times Square → JFK: ~21,8 km em linha reta
        # (Δlat ≈ 13,0 km; Δlon ≈ 17,5 km ajustado por cos(40,7°)).
        d = haversine_km(40.7580, -73.9855, 40.6413, -73.7781)
        assert d == pytest.approx(21.8, abs=0.2)

    def test_um_grau_de_latitude(self):
        # 1° de latitude ≈ 111,2 km em qualquer longitude.
        d = haversine_km(40.0, -74.0, 41.0, -74.0)
        assert d == pytest.approx(111.2, abs=0.3)

    def test_simetria(self):
        ida = haversine_km(40.75, -73.98, 40.64, -73.77)
        volta = haversine_km(40.64, -73.77, 40.75, -73.98)
        assert ida == pytest.approx(volta)

    def test_vetorizado(self):
        lat1 = np.array([40.75, 40.0])
        lon1 = np.array([-73.98, -74.0])
        lat2 = np.array([40.75, 41.0])
        lon2 = np.array([-73.98, -74.0])
        d = haversine_km(lat1, lon1, lat2, lon2)
        assert d.shape == (2,)
        assert d[0] == 0.0
        assert d[1] == pytest.approx(111.2, abs=0.3)


@pytest.fixture
def centroids():
    """Tabela mínima de centroides, no formato de load_zone_centroids."""
    return pd.DataFrame(
        {"lat": [40.70, 40.80], "lon": [-74.00, -73.90]},
        index=pd.Index([1, 2], name="LocationID"),
    )


@pytest.fixture
def trips():
    return pd.DataFrame({
        # ter 01/01/2019 (feriado), sáb 06/07/2019 14:30, seg 02/12/2019 23:59
        "lpep_pickup_datetime": pd.to_datetime(
            ["2019-01-01 08:00:00", "2019-07-06 14:30:00", "2019-12-02 23:59:00"]),
        "lpep_dropoff_datetime": pd.to_datetime(
            ["2019-01-01 08:20:00", "2019-07-06 14:30:30", "2019-12-03 03:30:00"]),
        "PULocationID": [1, 2, 1],
        "DOLocationID": [2, 2, 264],  # 264 não tem centroide
        "trip_type": [1.0, np.nan, 2.0],  # NaN = bloco MNAR (canal reduzido)
    })


class TestAddFeatures:
    def test_schema_completo(self, trips, centroids):
        out = add_features(trips, centroids)
        assert all(col in out.columns for col in FEATURE_COLUMNS)

    def test_nao_mutila_o_input(self, trips, centroids):
        colunas_antes = list(trips.columns)
        add_features(trips, centroids)
        assert list(trips.columns) == colunas_antes

    def test_haversine_entre_zonas(self, trips, centroids):
        out = add_features(trips, centroids)
        esperado = haversine_km(40.70, -74.00, 40.80, -73.90)
        assert out["haversine_km"].iloc[0] == pytest.approx(esperado)
        assert out["haversine_km"].iloc[1] == 0.0  # mesma zona

    def test_zona_sem_centroide_vira_nan(self, trips, centroids):
        out = add_features(trips, centroids)
        assert np.isnan(out["haversine_km"].iloc[2])

    def test_features_temporais(self, trips, centroids):
        out = add_features(trips, centroids)
        assert out["pickup_hour"].tolist() == [8, 14, 23]
        assert out["pickup_dayofweek"].tolist() == [1, 5, 0]  # ter, sáb, seg
        assert out["is_weekend"].tolist() == [0, 1, 0]

    def test_trip_type_nan_vira_categoria_zero(self, trips, centroids):
        out = add_features(trips, centroids)
        assert out["trip_type"].tolist() == [1, 0, 2]

    def test_feriado_federal(self, trips, centroids):
        out = add_features(trips, centroids)
        assert out["is_holiday"].tolist() == [1, 0, 0]  # só 01/jan

    def test_corrida_unica_como_na_api(self, trips, centroids):
        # O caminho da API: DataFrame de uma linha, mesmo código do treino.
        out = add_features(trips.iloc[[0]], centroids)
        assert len(out) == 1
        assert out["is_holiday"].iloc[0] == 1


class TestAlvo:
    def test_duracao_em_minutos(self, trips):
        dur = compute_duration_minutes(trips)
        assert dur.tolist() == pytest.approx([20.0, 0.5, 211.0])

    def test_mascara_de_limpeza(self, trips):
        dur = compute_duration_minutes(trips)
        assert clean_duration_mask(dur).tolist() == [True, False, False]

    def test_limites_inclusivos(self):
        dur = pd.Series([DUR_MIN_MINUTES, DUR_MAX_MINUTES, 0.99, 180.01])
        assert clean_duration_mask(dur).tolist() == [True, True, False, False]


@pytest.mark.skipif(not (DEFAULT_RAW_DIR / "taxi_zones.zip").exists(),
                    reason="taxi_zones.zip não baixado")
class TestBuildZoneCentroids:
    def test_build_do_zip_real(self, tmp_path):
        table = build_zone_centroids(DEFAULT_RAW_DIR, tmp_path / "centroids.csv")
        assert len(table) == 263
        assert table["LocationID"].is_unique
        # Caixa geográfica de NYC (com folga para EWR).
        assert table["lat"].between(40.4, 41.0).all()
        assert table["lon"].between(-74.3, -73.6).all()
        assert (tmp_path / "centroids.csv").exists()