"""Testes de drift.py — PSI (bins congelados) e KS-test manuais.

Os specs de bins usados aqui são construídos à mão no mesmo formato que o
train.py grava no metadata.json (ver docstring de drift.py).
"""

import numpy as np
import pandas as pd
import pytest

from drift import (
    compute_drift,
    compute_psi,
    ks_statistic,
    ks_vs_training,
    observed_props_categorical,
    observed_props_continuous,
    psi_from_props,
    retrain_status,
)

RNG_SEED = 42


def categorical_spec(categories, expected):
    return {"kind": "categorical", "categories": categories, "expected": expected}


def continuous_spec(edges, expected, nan_expected=0.0):
    return {
        "kind": "continuous",
        "edges": edges,
        "expected": expected,
        "nan_expected": nan_expected,
    }


class TestPsiFromProps:
    def test_distribuicoes_identicas_dao_zero(self):
        props = [0.25, 0.25, 0.25, 0.25]
        assert psi_from_props(props, props) == pytest.approx(0.0)

    def test_valor_conhecido_calculado_a_mao(self):
        # (0.7-0.5)ln(0.7/0.5) + (0.3-0.5)ln(0.3/0.5)
        #   = 0.2*0.336472 + (-0.2)*(-0.510826) = 0.169460
        psi = psi_from_props([0.7, 0.3], [0.5, 0.5])
        assert psi == pytest.approx(0.169460, abs=1e-5)

    def test_simetria(self):
        # A fórmula do PSI é simétrica em (observado, esperado).
        a, b = [0.6, 0.3, 0.1], [0.2, 0.5, 0.3]
        assert psi_from_props(a, b) == pytest.approx(psi_from_props(b, a))

    def test_bin_zerado_nao_da_infinito(self):
        psi = psi_from_props([0.0, 1.0], [0.5, 0.5])
        assert np.isfinite(psi)
        assert psi > 0.2  # colapso total de um bin é drift significativo

    def test_sempre_nao_negativo(self):
        # Cada termo (o-e)ln(o/e) é >= 0 (o-e e ln(o/e) têm o mesmo sinal).
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(20):
            o = rng.dirichlet(np.ones(8))
            e = rng.dirichlet(np.ones(8))
            assert psi_from_props(o, e) >= 0.0


class TestPsiCategorical:
    SPEC = categorical_spec(categories=[1, 2, 3], expected=[0.5, 0.3, 0.2])

    def test_mesma_distribuicao_do_treino_da_psi_baixo(self):
        series = pd.Series([1] * 50 + [2] * 30 + [3] * 20)
        assert compute_psi(series, self.SPEC) == pytest.approx(0.0, abs=1e-9)

    def test_alinhamento_com_categorias_do_treino(self):
        # Categoria 3 ausente na produção: o bin dela tem que aparecer com 0.
        series = pd.Series([1] * 70 + [2] * 30)
        observed, expected = observed_props_categorical(series, self.SPEC)
        assert observed == pytest.approx([0.7, 0.3, 0.0, 0.0])
        assert expected == pytest.approx([0.5, 0.3, 0.2, 0.0])

    def test_categoria_nova_cai_no_bin_outros(self):
        series = pd.Series([1] * 40 + [2] * 30 + [3] * 20 + [99] * 10)
        observed, _ = observed_props_categorical(series, self.SPEC)
        assert observed[-1] == pytest.approx(0.10)  # bin "outros"

    def test_categoria_nova_gera_psi_alto(self):
        # 10% da massa numa categoria com esperado ~0 tem que estourar o alarme.
        no_drift = pd.Series([1] * 50 + [2] * 30 + [3] * 20)
        with_new = pd.Series([1] * 45 + [2] * 27 + [3] * 18 + [99] * 10)
        assert compute_psi(with_new, self.SPEC) > 0.2
        assert compute_psi(with_new, self.SPEC) > compute_psi(no_drift, self.SPEC)

    def test_nan_cai_no_bin_outros(self):
        series = pd.Series([1] * 50 + [2] * 30 + [3] * 10 + [np.nan] * 10)
        observed, _ = observed_props_categorical(series, self.SPEC)
        assert observed[-1] == pytest.approx(0.10)

    def test_serie_vazia_levanta_erro(self):
        with pytest.raises(ValueError, match="vazia"):
            compute_psi(pd.Series([], dtype=float), self.SPEC)


class TestPsiContinuous:
    # Treino sintético: uniforme em [0, 4) com cortes nos inteiros;
    # 5 bins (contando as pontas abertas), pontas com massa zero.
    SPEC = continuous_spec(
        edges=[1.0, 2.0, 3.0],
        expected=[0.25, 0.25, 0.25, 0.25],
        nan_expected=0.0,
    )

    def test_mesma_distribuicao_do_treino_da_psi_baixo(self):
        rng = np.random.default_rng(RNG_SEED)
        series = pd.Series(rng.uniform(0, 4, size=20_000))
        assert compute_psi(series, self.SPEC) < 0.01

    def test_deslocamento_forte_da_psi_alto(self):
        rng = np.random.default_rng(RNG_SEED)
        series = pd.Series(rng.uniform(2, 6, size=20_000))
        assert compute_psi(series, self.SPEC) > 0.2

    def test_massa_fora_das_pontas_e_contada(self):
        series = pd.Series([-5.0] * 10 + [0.5] * 40 + [1.5] * 40 + [10.0] * 10)
        observed, expected = observed_props_continuous(series, self.SPEC)
        assert observed == pytest.approx([0.5, 0.4, 0.0, 0.1, 0.0])
        assert expected == pytest.approx([0.25, 0.25, 0.25, 0.25, 0.0])

    def test_nan_e_bin_proprio(self):
        spec = continuous_spec(
            edges=[1.0, 2.0, 3.0],
            expected=[0.24, 0.24, 0.24, 0.24],
            nan_expected=0.04,
        )
        series = pd.Series([0.5] * 24 + [1.5] * 24 + [2.5] * 24 + [3.5] * 24
                           + [np.nan] * 4)
        assert compute_psi(series, spec) == pytest.approx(0.0, abs=1e-9)

    def test_aumento_da_taxa_de_nan_sobe_o_psi(self):
        spec = continuous_spec(
            edges=[1.0, 2.0, 3.0],
            expected=[0.24, 0.24, 0.24, 0.24],
            nan_expected=0.04,
        )
        base = pd.Series([0.5, 1.5, 2.5, 3.5] * 24 + [np.nan] * 4)
        more_nan = pd.Series([0.5, 1.5, 2.5, 3.5] * 18 + [np.nan] * 28)
        assert compute_psi(more_nan, spec) > compute_psi(base, spec)


class TestKsStatistic:
    def test_amostras_identicas_dao_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert ks_statistic(a, a) == 0.0

    def test_amostras_disjuntas_dao_um(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        assert ks_statistic(a, b) == 1.0

    def test_valor_conhecido_calculado_a_mao(self):
        # a=[1,2,3,4], b=[3,4,5,6]: em x=2, ECDF_a=0.5 e ECDF_b=0 => D=0.5.
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([3.0, 4.0, 5.0, 6.0])
        assert ks_statistic(a, b) == pytest.approx(0.5)

    def test_simetria(self):
        rng = np.random.default_rng(RNG_SEED)
        a, b = rng.normal(0, 1, 500), rng.normal(0.5, 1, 400)
        assert ks_statistic(a, b) == pytest.approx(ks_statistic(b, a))

    def test_deslocamento_uniforme_conhecido(self):
        # U[0,1] vs U[0.5,1.5]: KS teórico = 0.5.
        rng = np.random.default_rng(RNG_SEED)
        a = rng.uniform(0, 1, 20_000)
        b = rng.uniform(0.5, 1.5, 20_000)
        assert ks_statistic(a, b) == pytest.approx(0.5, abs=0.02)

    def test_amostra_vazia_levanta_erro(self):
        with pytest.raises(ValueError, match="vazia"):
            ks_statistic(np.array([]), np.array([1.0]))


class TestKsVsTraining:
    SPEC = continuous_spec(
        edges=[1.0, 2.0, 3.0],
        expected=[0.25, 0.25, 0.25, 0.25],
        nan_expected=0.0,
    )

    def test_mesma_distribuicao_da_ks_baixo(self):
        rng = np.random.default_rng(RNG_SEED)
        series = pd.Series(rng.uniform(0, 4, size=20_000))
        assert ks_vs_training(series, self.SPEC) < 0.02

    def test_deslocamento_conhecido(self):
        # U[1,5] vs U[0,4]: nos cortes 1/2/3 a diferença de CDF é 0.25 no
        # corte 2 (treino 0.5, prod 0.25) e no corte 3 (0.75 vs 0.5).
        rng = np.random.default_rng(RNG_SEED)
        series = pd.Series(rng.uniform(1, 5, size=50_000))
        assert ks_vs_training(series, self.SPEC) == pytest.approx(0.25, abs=0.02)

    def test_ks_binado_e_lower_bound_do_ks_exato(self):
        rng = np.random.default_rng(RNG_SEED)
        train = rng.uniform(0, 4, size=50_000)
        prod = rng.uniform(1, 5, size=50_000)
        exact = ks_statistic(train, prod)
        binned = ks_vs_training(pd.Series(prod), self.SPEC)
        assert binned <= exact + 1e-9

    def test_ignora_nan_condicional(self):
        # KS é condicional a não-NaN: NaN não muda o resultado (PSI cuida).
        rng = np.random.default_rng(RNG_SEED)
        values = rng.uniform(0, 4, size=10_000)
        without_nan = pd.Series(values)
        with_nan = pd.Series(np.concatenate([values, [np.nan] * 1_000]))
        assert ks_vs_training(with_nan, self.SPEC) == pytest.approx(
            ks_vs_training(without_nan, self.SPEC)
        )


class TestRetrainStatus:
    # Regra v1: K=2 features com PSI > 0.2 por M=2 meses consecutivos.

    def test_sem_alerta_e_verde(self):
        assert retrain_status([0, 0, 1], k=2, m=2) == "green"

    def test_um_mes_em_alerta_e_amarelo(self):
        # Outlier de um mês só não dispara retreino — vira atenção.
        assert retrain_status([0, 0, 3], k=2, m=2) == "yellow"

    def test_streak_completa_e_vermelho(self):
        assert retrain_status([0, 2, 3], k=2, m=2) == "red"

    def test_streak_interrompida_reseta(self):
        # Mês em alerta, mês limpo, mês em alerta: não há M consecutivos.
        assert retrain_status([3, 0, 3], k=2, m=2) == "yellow"

    def test_menos_features_que_k_nao_conta(self):
        assert retrain_status([1, 1, 1], k=2, m=2) == "green"

    def test_historico_curto_nao_fecha_streak(self):
        # Primeiro mês monitorado: mesmo em alerta, não há histórico para M=2.
        assert retrain_status([5], k=2, m=2) == "yellow"

    def test_historico_vazio_e_verde(self):
        assert retrain_status([], k=2, m=2) == "green"

    def test_m_maior(self):
        assert retrain_status([2, 2, 2], k=2, m=3) == "red"
        assert retrain_status([0, 2, 2], k=2, m=3) == "yellow"


class TestComputeDrift:
    def test_controle_sem_drift_fica_abaixo_do_watch(self):
        # Grupo de controle sintético: produção amostrada da mesma
        # distribuição do treino => tudo < 0.1 (mesmo racional do sanity
        # check com jan/fev 2020).
        rng = np.random.default_rng(RNG_SEED)
        specs = {
            "zona": categorical_spec([1, 2, 3], [0.5, 0.3, 0.2]),
            "dist": continuous_spec([1.0, 2.0, 3.0],
                                    [0.25, 0.25, 0.25, 0.25]),
        }
        df = pd.DataFrame({
            "zona": rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2], size=20_000),
            "dist": rng.uniform(0, 4, size=20_000),
        })
        report = compute_drift(df, specs)
        assert set(report) == {"zona", "dist"}
        assert report["zona"]["psi"] < 0.1
        assert report["dist"]["psi"] < 0.1
        assert "ks" not in report["zona"]  # KS só para contínuas
        assert report["dist"]["ks"] < 0.02

    def test_specs_reais_do_metadata(self):
        # Smoke test com os bins congelados de verdade do models/metadata.json
        # (pulado se o artefato não existir, ex: CI sem modelo treinado).
        import json
        from pathlib import Path

        metadata_path = (Path(__file__).resolve().parents[1]
                         / "models" / "metadata.json")
        if not metadata_path.exists():
            pytest.skip("models/metadata.json não existe")
        specs = json.loads(metadata_path.read_text())["psi_bins"]

        rng = np.random.default_rng(RNG_SEED)
        n = 5_000
        df = pd.DataFrame({
            "PULocationID": rng.choice(specs["PULocationID"]["categories"], n),
            "DOLocationID": rng.choice(specs["DOLocationID"]["categories"], n),
            "trip_type": rng.choice([1, 2], n),
            "haversine_km": rng.exponential(3.0, n),
            "pickup_hour": rng.integers(0, 24, n),
            "pickup_dayofweek": rng.integers(0, 7, n),
            "is_weekend": rng.integers(0, 2, n),
            "is_holiday": rng.integers(0, 2, n),
        })
        report = compute_drift(df, specs)
        for feature, entry in report.items():
            assert np.isfinite(entry["psi"]), feature
            assert entry["psi"] >= 0.0, feature