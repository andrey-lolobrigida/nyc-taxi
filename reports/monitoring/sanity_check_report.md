# Sanity check do monitoramento — grupo de controle temporal

Gerado programaticamente por `scripts/sanity_check_report.py` em 2026-07-24 17:30 UTC — modelo `0.1.0` (treinado em 2026-07-15T22:31:30+00:00).

## Por que um grupo de controle

Os bins de PSI foram congelados na distribuição de treino (2019) e
nunca recalculados. Janeiro e fevereiro de 2020 ainda são "mundo
normal": se o PSI aparecesse alto neles, a causa provável seria bug
na implementação (bins, alinhamento de categorias, contagem de NaN),
não drift — e nenhum número da pandemia seria confiável. Este
relatório materializa esse teste.

## Setup do replay

- **2020-01**: 30001 predições (amostra estratificada por dia, seed fixa), feedback em 30001 (100%).
- **2020-02**: 29999 predições (amostra estratificada por dia, seed fixa), feedback em 29999 (100%).

## Drift por feature (PSI, bins congelados de 2019)

Thresholds: < 0.1 estável · 0.1–0.2 atenção · > 0.2 alerta.

| Feature | PSI 2020-01 | PSI 2020-02 | Leitura |
|---|---|---|---|
| `PULocationID` | 0.0333 | 0.0279 | estável |
| `DOLocationID` | 0.0179 | 0.0158 | estável |
| `trip_type` | 0.2372 | 0.1185 | alerta |
| `haversine_km` | 0.0021 | 0.0022 | estável |
| `pickup_hour` | 0.0053 | 0.0017 | estável |
| `pickup_dayofweek` | 0.0143 | 0.0038 | estável |
| `is_weekend` | 0.0078 | 0.0008 | estável |
| `is_holiday` | 0.0312 | 0.0027 | estável |

KS binado de `haversine_km` vs. treino: 2020-01: 0.016 · 2020-02: 0.020.

## Performance vs. holdout de 2019

| Métrica (min) | Holdout 2019 | 2020-01 | 2020-02 |
|---|---|---|---|
| MAE | 3.99 | 3.71 | 3.59 |
| RMSE | 7.15 | 6.11 | 6.25 |
| AE mediano | 2.45 | 2.45 | 2.34 |
| Viés (pred − real) | — | +0.18 | -0.25 |

O erro nos meses de controle fica no nível do holdout — o modelo
ainda opera no regime em que foi treinado.

## Gatilho de retreino

Regra v1: ≥ 2 features com PSI > 0.2 por 2 meses consecutivos.

- **2020-01**: 🟢 verde (em alerta: `trip_type`)
- **2020-02**: 🟢 verde

## O caso `trip_type`: drift real dentro do controle

`trip_type` é a exceção da tabela: PSI 0.237 em 2020-01. Não é bug de bins — é deriva anterior à
pandemia, visível nos dados de 2019 (fração de corridas SEM
`trip_type`, a categoria 0/bloco MNAR, após a limpeza do replay):

| Mês | trip_type ausente |
|---|---|
| 2019-01 | 6.4% |
| 2019-02 | 6.8% |
| 2019-03 | 6.8% |
| 2019-04 | 9.7% |
| 2019-05 | 7.7% |
| 2019-06 | 7.1% |
| 2019-07 | 8.2% |
| 2019-08 | 11.5% |
| 2019-09 | 13.4% |
| 2019-10 | 19.0% |
| 2019-11 | 19.7% |
| 2019-12 | 21.3% |
| 2020-01 | 26.3% |
| 2020-02 | 20.7% |

O treino congelou a média de 2019 (9.3%) como
esperado da categoria 0; a série acima mostra que a fração já
crescia de forma aproximadamente monotônica ao longo do próprio ano
de treino — o PSI está acusando uma mudança verdadeira, não um erro
de régua. A causa (mudança no canal de reporte/ingestão dos
registros) é **hipótese**, não verificável nestes dados; o fato
verificável é a tendência.

Dois pontos que isso demonstra a favor da estratégia:

1. o monitor detecta mudança real de distribuição mesmo fora de um
   evento dramático;
2. a regra K=2/M=2 segura o alarme: uma feature isolada em
   alerta não dispara retreino (status verde nos dois meses).

## Veredito

**Sanity check aprovado.** Todas as features, exceto o caso
documentado de `trip_type`, ficaram abaixo de 0.1 nos dois meses de controle. A régua
(bins congelados + PSI/KS) é confiável e os números de
2020-03 em diante podem ser interpretados como drift dos dados,
não artefato da implementação.
