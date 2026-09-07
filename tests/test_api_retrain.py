"""Testes do parecer de retreino: /retrain-check, /retrain-trigger e
/timeline — ou seja, o `monthly_signals` e os DOIS gatilhos da decisão #8.

É a lógica nova do projeto e a que tem mais regra por linha: janela de
calendário (buraco na série encurta a janela em vez de costurar meses
distantes), piso de feedback do gatilho de performance, e o reconhecimento,
que não muda o status bruto, vale por gatilho e expira sozinho quando aquele
gatilho volta ao verde.

Convenções do metadata falso (ver conftest): MAE do holdout 4,0 → limiar do
gatilho de performance 5,0. K=2, M=2, piso de 100 feedbacks.
"""

import pytest

from drift import PERF_MIN_FEEDBACK, PSI_ALERT, PSI_WATCH, RETRAIN_K, RETRAIN_M

DUAS_FEATURES = ("PULocationID", "DOLocationID")  # n_alert = K
MAE_ACIMA = 6.0   # > 5,0
MAE_ABAIXO = 4.5  # < 5,0


def verdict(client, month):
    response = client.get("/monitoring/retrain-check", params={"month": month})
    assert response.status_code == 200, response.text
    return response.json()["verdict"]


class TestGatilhoDeDrift:
    def test_k_features_por_m_meses_seguidos_fecha_a_streak(self, client, seed):
        seed("2020-03", drifted=DUAS_FEATURES)
        seed("2020-04", drifted=DUAS_FEATURES)

        assert verdict(client, "2020-03")["drift_status"] == "yellow"
        assert verdict(client, "2020-04")["drift_status"] == "red"

    def test_um_mes_isolado_fica_em_atencao(self, client, seed):
        seed("2020-03", drifted=DUAS_FEATURES)
        seed("2020-04")

        assert verdict(client, "2020-03")["drift_status"] == "yellow"
        assert verdict(client, "2020-04")["drift_status"] == "green"

    def test_menos_de_k_features_nao_conta(self, client, seed):
        seed("2020-03", drifted=("PULocationID",))
        seed("2020-04", drifted=("PULocationID",))

        assert verdict(client, "2020-04")["drift_status"] == "green"

    def test_mes_faltando_encurta_a_janela_e_impede_a_streak(self, client, seed):
        """Janela de CALENDÁRIO: sem fevereiro, a janela de março tem um mês
        medido só, e um mês não fecha uma streak de dois — mesmo que o mês
        anterior COM dados também estivesse em alerta."""
        seed("2020-01", drifted=DUAS_FEATURES)
        seed("2020-03", drifted=DUAS_FEATURES)

        assert verdict(client, "2020-03")["drift_status"] == "yellow"

    def test_alarme_desliga_quando_o_drift_passa(self, client, seed):
        seed("2020-03", drifted=DUAS_FEATURES)
        seed("2020-04", drifted=DUAS_FEATURES)
        seed("2020-05")

        assert verdict(client, "2020-05")["drift_status"] == "green"


class TestGatilhoDePerformance:
    def test_mae_acima_do_limiar_por_m_meses_fecha_a_streak(self, client, seed):
        seed("2021-10", mae=MAE_ACIMA)
        seed("2021-11", mae=MAE_ACIMA)

        assert verdict(client, "2021-10")["performance_status"] == "yellow"
        assert verdict(client, "2021-11")["performance_status"] == "red"

    def test_mae_abaixo_do_limiar_fica_verde(self, client, seed):
        seed("2021-10", mae=MAE_ABAIXO)
        seed("2021-11", mae=MAE_ABAIXO)

        assert verdict(client, "2021-11")["performance_status"] == "green"

    def test_mes_sem_feedback_nao_julga_performance(self, client, seed):
        """Sem ground truth não se afirma nem se nega degradação: o mês sai
        da janela, como um mês sem dados sai da janela do drift."""
        seed("2021-10", mae=MAE_ACIMA)
        seed("2021-11")  # sem nenhum feedback

        assert verdict(client, "2021-11")["performance_status"] == "yellow"

    def test_poucos_feedbacks_nao_decidem_um_alarme(self, client, seed):
        """O piso existe para um punhado de pares não fechar uma streak
        sozinho: com 99 pares o mês não entra na janela."""
        seed("2021-10", n=300, mae=MAE_ACIMA)
        seed("2021-11", n=300, mae=MAE_ACIMA, n_feedback=PERF_MIN_FEEDBACK - 1)

        assert verdict(client, "2021-11")["performance_status"] == "yellow"

    def test_no_piso_exato_o_mes_conta(self, client, seed):
        seed("2021-10", n=300, mae=MAE_ACIMA)
        seed("2021-11", n=300, mae=MAE_ACIMA, n_feedback=PERF_MIN_FEEDBACK)

        assert verdict(client, "2021-11")["performance_status"] == "red"


class TestParecerGeral:
    def test_e_o_pior_dos_dois_gatilhos(self, client, seed):
        """A discordância é o ponto do projeto: inputs longe de 2019 com o
        modelo acertando, e inputs de volta ao normal com o erro alto."""
        # drift vermelho, performance verde (sem feedback)
        seed("2020-06", drifted=DUAS_FEATURES)
        seed("2020-07", drifted=DUAS_FEATURES)
        # drift verde, performance vermelha
        seed("2021-10", mae=MAE_ACIMA)
        seed("2021-11", mae=MAE_ACIMA)

        julho = verdict(client, "2020-07")
        novembro = verdict(client, "2021-11")

        assert (julho["drift_status"], julho["performance_status"]) == ("red", "green")
        assert julho["status"] == "red"
        assert (novembro["drift_status"],
                novembro["performance_status"]) == ("green", "red")
        assert novembro["status"] == "red"


class TestRetrainCheck:
    def test_devolve_a_regra_o_limiar_e_a_janela_avaliada(self, client, seed):
        seed("2020-02", drifted=DUAS_FEATURES, mae=MAE_ACIMA)
        seed("2020-03", drifted=DUAS_FEATURES, mae=MAE_ACIMA)

        body = client.get("/monitoring/retrain-check",
                          params={"month": "2020-03"}).json()

        assert body["month"] == "2020-03"
        assert body["mae_threshold_min"] == pytest.approx(5.0)  # holdout 4,0 × 1,25
        assert body["rule"]["k"] == RETRAIN_K
        assert body["rule"]["m"] == RETRAIN_M
        assert body["rule"]["psi_alert"] == PSI_ALERT
        assert [m["month"] for m in body["months"]] == ["2020-02", "2020-03"]
        assert body["months"][-1]["n_alert"] == 2
        assert body["months"][-1]["alert_features"] == sorted(DUAS_FEATURES)
        assert body["months"][-1]["mae_min"] == pytest.approx(MAE_ACIMA)

    def test_janela_omite_meses_sem_dados(self, client, seed):
        seed("2020-03", drifted=DUAS_FEATURES)

        body = client.get("/monitoring/retrain-check",
                          params={"month": "2020-03"}).json()

        assert [m["month"] for m in body["months"]] == ["2020-03"]

    def test_mes_sem_predicao_da_404(self, client, seed):
        seed("2020-03")
        assert client.get("/monitoring/retrain-check",
                          params={"month": "2020-07"}).status_code == 404

    def test_banco_vazio_da_404(self, client):
        assert client.get("/monitoring/retrain-check",
                          params={"month": "2020-03"}).status_code == 404

    def test_mes_malformado_da_422(self, client, seed):
        seed("2020-03")
        assert client.get("/monitoring/retrain-check",
                          params={"month": "2020-3"}).status_code == 422


class TestReconhecimento:
    def test_registra_o_evento_e_diz_qual_gatilho_estava_aceso(self, client,
                                                              seed, conn):
        seed("2020-03", drifted=DUAS_FEATURES)
        seed("2020-04", drifted=DUAS_FEATURES)

        response = client.post("/monitoring/retrain-trigger",
                               json={"month": "2020-04", "note": "vi o alarme"})

        assert response.status_code == 201
        body = response.json()
        assert body["month"] == "2020-04"
        assert body["triggered_by"] == "drift"
        assert body["retrained"] is False, "retreinar está fora do escopo"
        assert body["created_at"]
        row = conn.execute("SELECT * FROM retrain_events WHERE id = ?",
                           (body["id"],)).fetchone()
        assert (row["month"], row["triggered_by"], row["note"]) == (
            "2020-04", "drift", "vi o alarme")

    @pytest.mark.parametrize("drifted,mae,esperado", [
        (DUAS_FEATURES, None, "drift"),
        ((), MAE_ACIMA, "performance"),
        (DUAS_FEATURES, MAE_ACIMA, "both"),
        ((), MAE_ABAIXO, "none"),
    ])
    def test_triggered_by_reflete_o_estado_do_mes(self, client, seed,
                                                  drifted, mae, esperado):
        seed("2020-03", drifted=drifted, mae=mae)
        seed("2020-04", drifted=drifted, mae=mae)

        body = client.post("/monitoring/retrain-trigger",
                           json={"month": "2020-04"}).json()

        assert body["triggered_by"] == esperado

    def test_silencia_o_alarme_sem_mudar_o_status_bruto(self, client, seed):
        seed("2020-03", drifted=DUAS_FEATURES)
        seed("2020-04", drifted=DUAS_FEATURES)
        client.post("/monitoring/retrain-trigger", json={"month": "2020-04"})

        parecer = verdict(client, "2020-04")

        assert parecer["drift_status"] == "red", "o sinal continua auditável"
        assert parecer["drift_acknowledged_month"] == "2020-04"

    def test_vale_dos_meses_seguintes_em_diante(self, client, seed):
        seed("2020-03", drifted=DUAS_FEATURES)
        seed("2020-04", drifted=DUAS_FEATURES)
        seed("2020-05", drifted=DUAS_FEATURES)
        client.post("/monitoring/retrain-trigger", json={"month": "2020-04"})

        assert verdict(client, "2020-03")["drift_acknowledged_month"] is None
        assert verdict(client, "2020-05")["drift_acknowledged_month"] == "2020-04"

    def test_expira_quando_o_gatilho_volta_a_verde(self, client, seed):
        seed("2020-03", drifted=DUAS_FEATURES)
        seed("2020-04", drifted=DUAS_FEATURES)
        seed("2020-05")                          # episódio resolvido
        seed("2020-06", drifted=DUAS_FEATURES)   # episódio novo
        seed("2020-07", drifted=DUAS_FEATURES)
        client.post("/monitoring/retrain-trigger", json={"month": "2020-04"})

        assert verdict(client, "2020-05")["drift_acknowledged_month"] is None
        julho = verdict(client, "2020-07")
        assert julho["drift_status"] == "red"
        assert julho["drift_acknowledged_month"] is None, "alarme rearmado"

    def test_vale_por_gatilho_e_nao_sobre_o_parecer_geral(self, client, seed):
        """Reconhecer o episódio de drift de 2020 não pode silenciar a
        degradação de performance de 2021 — na série real o pior-dos-dois
        nunca volta a verde, e um clique só apagaria o resto da história."""
        seed("2020-03", drifted=DUAS_FEATURES, mae=MAE_ABAIXO)
        seed("2020-04", drifted=DUAS_FEATURES, mae=MAE_ABAIXO)
        seed("2020-05", mae=MAE_ACIMA)
        seed("2020-06", mae=MAE_ACIMA)
        client.post("/monitoring/retrain-trigger", json={"month": "2020-04"})

        abril = verdict(client, "2020-04")
        junho = verdict(client, "2020-06")

        assert abril["drift_acknowledged_month"] == "2020-04"
        assert abril["performance_acknowledged_month"] is None
        assert junho["performance_status"] == "red"
        assert junho["performance_acknowledged_month"] is None

    def test_reconhecer_mes_verde_nao_ativa_nada(self, client, seed):
        seed("2020-03")
        seed("2020-04")
        client.post("/monitoring/retrain-trigger", json={"month": "2020-04"})

        assert verdict(client, "2020-04")["drift_acknowledged_month"] is None

    def test_mes_sem_predicao_da_404(self, client, seed):
        seed("2020-03")
        assert client.post("/monitoring/retrain-trigger",
                           json={"month": "2020-07"}).status_code == 404

    def test_mes_malformado_da_422(self, client, seed):
        seed("2020-03")
        assert client.post("/monitoring/retrain-trigger",
                           json={"month": "marco"}).status_code == 422

    def test_nota_longa_demais_da_422(self, client, seed):
        seed("2020-03")
        assert client.post("/monitoring/retrain-trigger",
                           json={"month": "2020-03",
                                 "note": "x" * 501}).status_code == 422


class TestTimeline:
    def test_traz_a_serie_inteira_em_ordem_com_o_que_e_constante_no_topo(
            self, client, seed, metadata):
        seed("2020-04", n=200, drifted=DUAS_FEATURES, mae=MAE_ACIMA)
        seed("2020-03", n=100)  # semeado fora de ordem de propósito

        body = client.get("/monitoring/timeline").json()

        assert [m["month"] for m in body["months"]] == ["2020-03", "2020-04"]
        assert body["feature_names"] == metadata["features"]
        assert body["thresholds"] == {"watch": PSI_WATCH, "alert": PSI_ALERT}
        assert body["mae_threshold_min"] == pytest.approx(5.0)
        assert body["holdout_reference"] == metadata["holdout_metrics"]
        assert body["retrain_events"] == []

    def test_cada_mes_traz_drift_performance_e_parecer(self, client, seed,
                                                       metadata):
        seed("2020-04", n=200, drifted=DUAS_FEATURES, mae=MAE_ACIMA,
             n_feedback=150)

        mes = client.get("/monitoring/timeline").json()["months"][0]

        assert mes["n_predictions"] == 200
        assert mes["n_with_feedback"] == 150
        assert mes["n_alert"] == 2
        assert mes["alert_features"] == sorted(DUAS_FEATURES)
        assert mes["mae_min"] == pytest.approx(MAE_ACIMA)
        assert mes["bias_min"] == pytest.approx(MAE_ACIMA)
        assert set(mes["features"]) == set(metadata["features"])
        assert mes["verdict"]["status"] == "yellow"

    def test_bate_com_o_retrain_check_do_mesmo_mes(self, client, seed):
        """Os dois endpoints têm que sair da mesma função — se divergirem,
        o dashboard mostra um parecer e a API responde outro."""
        seed("2020-03", drifted=DUAS_FEATURES, mae=MAE_ACIMA)
        seed("2020-04", drifted=DUAS_FEATURES, mae=MAE_ACIMA)

        por_mes = {m["month"]: m["verdict"]
                   for m in client.get("/monitoring/timeline").json()["months"]}

        assert por_mes["2020-04"] == verdict(client, "2020-04")

    def test_lista_os_reconhecimentos_mais_recente_primeiro(self, client, seed):
        seed("2020-03", drifted=DUAS_FEATURES)
        seed("2020-04", drifted=DUAS_FEATURES)
        client.post("/monitoring/retrain-trigger", json={"month": "2020-03"})
        client.post("/monitoring/retrain-trigger",
                    json={"month": "2020-04", "note": "segundo"})

        eventos = client.get("/monitoring/timeline").json()["retrain_events"]

        assert [e["month"] for e in eventos] == ["2020-04", "2020-03"]
        assert eventos[0]["note"] == "segundo"

    def test_banco_vazio_da_404(self, client):
        assert client.get("/monitoring/timeline").status_code == 404
