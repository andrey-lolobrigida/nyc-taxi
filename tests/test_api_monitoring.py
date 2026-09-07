"""Testes de GET /monitoring/drift e /monitoring/performance.

Os bins do metadata falso têm dois compartimentos de 50% cada (ver
conftest), então os dois extremos são conferíveis de cabeça:

- feature 50/50 → PSI 0, "stable";
- feature 100% num bin → PSI = 0,5·ln2 + (ε−0,5)·ln(2ε) ≈ 4,6043, "alert".

Só `haversine_km` é contínua no metadata falso — de propósito, como no
metadata real — então é a única com KS.
"""

import json
from pathlib import Path

import pytest

from drift import PSI_ALERT, PSI_WATCH
from features import FEATURE_COLUMNS

PSI_100_PCT_NUM_BIN = 4.6043  # 0.5*ln(2) + (1e-4 - 0.5)*ln(2e-4)


class TestDrift:
    def test_mes_sem_deriva_da_psi_zero_em_todas_as_features(self, client, seed):
        seed("2020-01", n=200)

        body = client.get("/monitoring/drift", params={"month": "2020-01"}).json()

        assert body["month"] == "2020-01"
        assert body["n_predictions"] == 200
        assert body["n_alert"] == 0
        assert body["thresholds"] == {"watch": PSI_WATCH, "alert": PSI_ALERT}
        assert set(body["features"]) == set(FEATURE_COLUMNS)
        assert all(f["psi"] == pytest.approx(0.0) and f["level"] == "stable"
                   for f in body["features"].values())

    def test_feature_toda_num_bin_so_vira_alerta(self, client, seed):
        seed("2020-04", n=200, drifted=("PULocationID", "pickup_hour"))

        body = client.get("/monitoring/drift", params={"month": "2020-04"}).json()

        assert body["n_alert"] == 2
        assert body["features"]["PULocationID"]["psi"] == pytest.approx(
            PSI_100_PCT_NUM_BIN, abs=1e-4)
        assert body["features"]["PULocationID"]["level"] == "alert"
        assert body["features"]["pickup_hour"]["level"] == "alert"
        assert body["features"]["DOLocationID"]["level"] == "stable"

    def test_ks_so_existe_para_a_feature_continua(self, client, seed):
        seed("2020-01", n=200)

        features = client.get("/monitoring/drift",
                              params={"month": "2020-01"}).json()["features"]

        assert features["haversine_km"]["ks"] == pytest.approx(0.0)
        assert all(features[name]["ks"] is None
                   for name in FEATURE_COLUMNS if name != "haversine_km")

    def test_ks_acompanha_o_deslocamento_da_continua(self, client, seed):
        """Todas as corridas abaixo do corte: CDF de produção 1,0 contra
        0,5 do treino no único corte congelado."""
        seed("2020-04", n=200, drifted=("haversine_km",))

        body = client.get("/monitoring/drift", params={"month": "2020-04"}).json()

        assert body["features"]["haversine_km"]["ks"] == pytest.approx(0.5)

    def test_meses_vizinhos_nao_vazam_um_no_outro(self, client, seed):
        seed("2020-01", n=200)
        seed("2020-02", n=100, drifted=("trip_type",))

        janeiro = client.get("/monitoring/drift",
                             params={"month": "2020-01"}).json()
        fevereiro = client.get("/monitoring/drift",
                               params={"month": "2020-02"}).json()

        assert (janeiro["n_predictions"], fevereiro["n_predictions"]) == (200, 100)
        assert janeiro["features"]["trip_type"]["level"] == "stable"
        assert fevereiro["features"]["trip_type"]["level"] == "alert"

    def test_janela_do_mes_pega_as_duas_bordas(self, client, seed):
        """[início, fim) sobre o isoformat: o último instante de janeiro
        entra, a virada de fevereiro não."""
        seed("2020-01", n=2, pickup="2020-01-01T00:00:00")
        seed("2020-01", n=2, pickup="2020-01-31T23:59:59")
        seed("2020-02", n=2, pickup="2020-02-01T00:00:00")

        body = client.get("/monitoring/drift", params={"month": "2020-01"}).json()

        assert body["n_predictions"] == 4

    def test_mes_sem_predicao_da_404(self, client, seed):
        seed("2020-01")
        response = client.get("/monitoring/drift", params={"month": "2020-07"})
        assert response.status_code == 404

    @pytest.mark.parametrize("month", ["2020-13", "2020-00", "2020-4", "abril",
                                       "2020-04-15", ""])
    def test_mes_malformado_da_422(self, client, month):
        assert client.get("/monitoring/drift",
                          params={"month": month}).status_code == 422

    def test_mes_obrigatorio(self, client):
        assert client.get("/monitoring/drift").status_code == 422


class TestPerformance:
    def test_metricas_dos_pares_predicao_feedback(self, client, seed, metadata):
        # erro constante de +2 min em todos os pares: MAE = RMSE = mediana = 2,
        # e o viés é positivo porque o modelo superestima.
        seed("2020-04", n=200, mae=2.0)

        body = client.get("/monitoring/performance",
                          params={"month": "2020-04"}).json()

        assert body["n_predictions"] == 200
        assert body["n_with_feedback"] == 200
        assert body["mae_min"] == pytest.approx(2.0)
        assert body["rmse_min"] == pytest.approx(2.0)
        assert body["median_ae_min"] == pytest.approx(2.0)
        assert body["bias_min"] == pytest.approx(2.0)
        assert body["holdout_reference"] == metadata["holdout_metrics"]

    def test_conta_so_as_linhas_que_ja_receberam_feedback(self, client, seed):
        seed("2020-04", n=200, mae=2.0, n_feedback=50)

        body = client.get("/monitoring/performance",
                          params={"month": "2020-04"}).json()

        assert (body["n_predictions"], body["n_with_feedback"]) == (200, 50)
        assert body["mae_min"] == pytest.approx(2.0)

    def test_mes_sem_feedback_devolve_metricas_nulas_e_nao_erro(self, client, seed):
        """A confirmação atrasada ainda em trânsito não é falha: o mês
        existe, tem predições, e as métricas chegam depois."""
        seed("2020-04", n=200)

        body = client.get("/monitoring/performance",
                          params={"month": "2020-04"}).json()

        assert body["n_predictions"] == 200
        assert body["n_with_feedback"] == 0
        assert (body["mae_min"], body["rmse_min"], body["median_ae_min"],
                body["bias_min"]) == (None, None, None, None)

    def test_mes_sem_predicao_da_404(self, client, seed):
        seed("2020-01")
        assert client.get("/monitoring/performance",
                          params={"month": "2020-07"}).status_code == 404


class TestMetadataReal:
    def test_bins_congelados_cobrem_o_schema_de_features(self):
        """O metadata.json é o único artefato do modelo versionado, e a
        régua do PSI vem dele: se uma feature entrar no schema sem bin
        congelado, o /monitoring/drift quebra em produção. Mesmo fail-fast
        do lifespan, virado teste."""
        metadata = json.loads(
            (Path(__file__).resolve().parents[1] / "models" / "metadata.json")
            .read_text())

        assert metadata["features"] == FEATURE_COLUMNS
        assert list(metadata["psi_bins"]) == FEATURE_COLUMNS
        assert set(metadata["holdout_metrics"]) >= {"mae_min", "rmse_min",
                                                    "median_ae_min"}
