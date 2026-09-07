"""Testes de /health, /predict e /feedback — o caminho de inferência.

O que importa aqui não é o número que o modelo devolve (o booster é falso),
e sim que o request vira uma linha de log com as FEATURES COMPUTADAS: é
sobre essas colunas que o PSI da fase 3 roda, então um /predict que
respondesse certo mas logasse errado quebraria o monitoramento inteiro sem
aparecer na resposta.
"""

import pandas as pd
import pytest

import api.main
from conftest import CENTROIDS
from features import haversine_km

TRIP = {
    "pickup_datetime": "2020-04-15T08:30:00",  # quarta-feira, 08h, dia útil
    "PULocationID": 1,
    "DOLocationID": 3,
    "trip_type": 1,
}


def routes(app):
    return {(route.path, tuple(sorted(route.methods)))
            for route in app.routes if hasattr(route, "methods")}


class TestAppDeTeste:
    def test_espelha_as_rotas_do_main(self, app):
        """Guarda contra a fixture divergir do app real sem ninguém notar —
        um router novo no main.py tem que quebrar isto."""
        assert routes(app) == routes(api.main.app)


class TestHealth:
    def test_devolve_versao_do_modelo_carregado(self, client, metadata):
        body = client.get("/health").json()
        assert body == {"status": "ok",
                        "model_version": metadata["model_version"],
                        "trained_at": metadata["trained_at"]}


class TestPredict:
    def test_responde_com_a_predicao_do_booster(self, client):
        body = client.post("/predict", json=TRIP).json()

        km = haversine_km(*CENTROIDS.loc[1, ["lat", "lon"]],
                          *CENTROIDS.loc[3, ["lat", "lon"]])
        assert body["predicted_duration_min"] == pytest.approx(5.0 + 3.0 * km)
        assert body["model_version"] == "test-0.0.1"
        assert body["prediction_id"]

    def test_loga_as_features_computadas_e_nao_so_o_request(self, client, conn):
        prediction_id = client.post("/predict", json=TRIP).json()["prediction_id"]

        row = conn.execute("SELECT * FROM predictions WHERE prediction_id = ?",
                           (prediction_id,)).fetchone()
        km = haversine_km(*CENTROIDS.loc[1, ["lat", "lon"]],
                          *CENTROIDS.loc[3, ["lat", "lon"]])
        assert dict(row) | {"created_at": None} == {
            "prediction_id": prediction_id,
            "created_at": None,  # relógio de parede, conferido à parte
            "pickup_datetime": "2020-04-15T08:30:00",
            "pu_location_id": 1,
            "do_location_id": 3,
            "trip_type": 1,
            "haversine_km": pytest.approx(km),
            "pickup_hour": 8,
            "pickup_dayofweek": 2,
            "is_weekend": 0,
            "is_holiday": 0,
            "predicted_duration_min": pytest.approx(5.0 + 3.0 * km),
            "actual_duration_min": None,
            "feedback_at": None,
        }

    def test_trip_type_ausente_vira_a_categoria_zero(self, client, conn):
        """O bloco MNAR do dataset tem perfil próprio: NaN vira categoria 0
        no treino, e o log tem que registrar a mesma convenção."""
        body = client.post("/predict", json={k: v for k, v in TRIP.items()
                                             if k != "trip_type"}).json()

        assert conn.execute(
            "SELECT trip_type FROM predictions WHERE prediction_id = ?",
            (body["prediction_id"],)).fetchone()["trip_type"] == 0

    def test_zona_sem_centroide_loga_haversine_nulo(self, client, conn):
        """264 (Unknown) e 265 (Outside NYC) não têm centroide no shapefile.
        São zonas válidas do contrato: a corrida entra, a distância é
        desconhecida (~0,4% do tráfego real)."""
        body = client.post("/predict", json=TRIP | {"DOLocationID": 264}).json()

        assert conn.execute(
            "SELECT haversine_km FROM predictions WHERE prediction_id = ?",
            (body["prediction_id"],)).fetchone()["haversine_km"] is None
        assert body["predicted_duration_min"] == pytest.approx(5.0)

    @pytest.mark.parametrize("pickup,esperado", [
        # Natal de 2020 caiu numa sexta: feriado no próprio dia.
        ("2020-12-25T14:00:00", (1, 0, 4)),
        # 4 de julho de 2020 caiu num sábado e o calendário federal o
        # OBSERVA na sexta anterior — o dia 4 em si não é feriado.
        ("2020-07-03T14:00:00", (1, 0, 4)),
        ("2020-07-04T14:00:00", (0, 1, 5)),
    ])
    def test_feriado_e_fim_de_semana_saem_do_calendario(self, client, conn,
                                                        pickup, esperado):
        body = client.post(
            "/predict", json=TRIP | {"pickup_datetime": pickup}).json()

        row = conn.execute(
            "SELECT is_holiday, is_weekend, pickup_dayofweek FROM predictions"
            " WHERE prediction_id = ?", (body["prediction_id"],)).fetchone()
        assert (row["is_holiday"], row["is_weekend"],
                row["pickup_dayofweek"]) == esperado

    @pytest.mark.parametrize("campo,valor", [
        ("PULocationID", 0), ("PULocationID", 266),
        ("DOLocationID", 0), ("DOLocationID", 266),
        ("trip_type", 3),
    ])
    def test_recusa_valores_fora_do_dominio(self, client, campo, valor):
        assert client.post("/predict", json=TRIP | {campo: valor}).status_code == 422

    def test_recusa_request_sem_o_horario_do_embarque(self, client):
        assert client.post(
            "/predict", json={k: v for k, v in TRIP.items()
                              if k != "pickup_datetime"}).status_code == 422

    def test_nao_aceita_a_duracao_real_no_request(self, client, conn):
        """Regra de leakage no contrato: a duração só existe quando a
        corrida acaba, então ela entra por /feedback e nunca no /predict.
        Campo extra é ignorado, não vira coluna."""
        body = client.post("/predict",
                           json=TRIP | {"actual_duration_min": 42.0}).json()

        assert conn.execute(
            "SELECT actual_duration_min FROM predictions WHERE prediction_id = ?",
            (body["prediction_id"],)).fetchone()["actual_duration_min"] is None


class TestFeedback:
    def test_ida_e_volta_preenche_a_linha_ja_logada(self, client, conn):
        prediction_id = client.post("/predict", json=TRIP).json()["prediction_id"]

        response = client.post("/feedback", json={
            "prediction_id": prediction_id, "actual_duration_min": 12.5})

        assert response.status_code == 200
        assert response.json() == {"prediction_id": prediction_id, "status": "ok"}
        rows = conn.execute("SELECT * FROM predictions").fetchall()
        assert len(rows) == 1, "feedback tem que preencher a linha, não criar outra"
        assert rows[0]["actual_duration_min"] == 12.5
        assert rows[0]["feedback_at"] is not None

    def test_prediction_id_desconhecido_da_404(self, client):
        response = client.post("/feedback", json={
            "prediction_id": "nunca-existiu", "actual_duration_min": 12.5})
        assert response.status_code == 404

    def test_feedback_repetido_da_409_e_mantem_o_primeiro(self, client, conn):
        """O replay pode reenviar; o ground truth de uma corrida é um só."""
        prediction_id = client.post("/predict", json=TRIP).json()["prediction_id"]
        payload = {"prediction_id": prediction_id, "actual_duration_min": 12.5}
        client.post("/feedback", json=payload)

        response = client.post("/feedback",
                               json=payload | {"actual_duration_min": 99.0})

        assert response.status_code == 409
        assert conn.execute(
            "SELECT actual_duration_min FROM predictions WHERE prediction_id = ?",
            (prediction_id,)).fetchone()["actual_duration_min"] == 12.5

    def test_recusa_duracao_nao_positiva(self, client):
        prediction_id = client.post("/predict", json=TRIP).json()["prediction_id"]
        assert client.post("/feedback", json={
            "prediction_id": prediction_id,
            "actual_duration_min": 0.0}).status_code == 422
