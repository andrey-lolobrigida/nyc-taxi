"""Métricas de drift implementadas na unha: PSI e KS-test (decisão #6).

Regra dos bins congelados: os bins vêm do `metadata.json` gravado pelo
train.py (specs de `psi_bin_specs`) e NUNCA são recalculados aqui — os bins
são a régua, e régua que muda de tamanho não mede nada. Este módulo só
computa a metade "observada" da conta e compara com o `expected` congelado.

Formato dos specs (um por feature, chave `psi_bins` do metadata):
- categórico: {"kind": "categorical", "categories": [...], "expected": [...]}
- contínuo:   {"kind": "continuous", "edges": [cortes internos],
               "expected": [...], "nan_expected": p}
  As pontas são abertas (-inf/+inf); `expected` soma 1 junto com
  `nan_expected`.

Decisões de implementação:
- Categorias não vistas no treino (ex: zona nova em produção) caem num bin
  "outros" com esperado 0 — com a suavização epsilon isso gera contribuição
  alta de PSI, que é o comportamento desejado: categoria nova É drift.
  NaN em série categórica também cai em "outros" (no treino não havia NaN
  nas categóricas; ver features.py).
- NaN em contínua é um bin próprio (zonas 264/265 sem centroide continuam
  chegando em produção e precisam ser contadas, não ignoradas).
- Epsilon (1e-4) entra por clip nos dois lados antes do log, convenção
  comum de mercado; sem ele, bin zerado => log(0) => PSI infinito.
- KS contra o treino: não guardamos os valores crus de 2019, só os bins.
  `ks_vs_training` avalia |CDF_prod - CDF_treino| apenas nos cortes
  congelados, onde a CDF de treino é exata (cumsum dos expected). É um KS
  "binado": lower bound do KS verdadeiro, suficiente como segundo sinal ao
  lado do PSI. A versão clássica de duas amostras (`ks_statistic`) fica
  disponível para quando existirem duas amostras cruas.
  O KS binado é condicional a não-NaN dos dois lados — a massa de NaN já é
  vigiada pelo bin próprio do PSI.
"""

import numpy as np
import pandas as pd

PSI_EPSILON = 1e-4

# Thresholds (convenção de mercado — tabela da decisão #8 do CLAUDE.md).
PSI_WATCH = 0.1   # atenção
PSI_ALERT = 0.2   # drift significativo

# Regra de retreino v1 (decisão #8): dispara se RETRAIN_K ou mais features
# tiverem PSI > PSI_ALERT por RETRAIN_M meses consecutivos. K e M são
# candidatos a recalibração depois de ver os números reais da pandemia
# (decisão em aberto no CLAUDE.md).
RETRAIN_K = 2
RETRAIN_M = 2


def psi_from_props(observed: np.ndarray, expected: np.ndarray,
                   epsilon: float = PSI_EPSILON) -> float:
    """PSI = Σ (o_i - e_i) * ln(o_i / e_i), com clip epsilon nos dois lados.

    Espera duas listas de proporções alinhadas bin a bin (cada uma somando
    ~1). O clip evita log(0); a pequena massa extra que ele injeta em bins
    zerados não é renormalizada (convenção usual — o erro é << 0.01 mesmo
    com centenas de bins).
    """
    o = np.clip(np.asarray(observed, dtype=float), epsilon, None)
    e = np.clip(np.asarray(expected, dtype=float), epsilon, None)
    return float(np.sum((o - e) * np.log(o / e)))


def observed_props_categorical(series: pd.Series, spec: dict) -> tuple[list, list]:
    """Proporções observadas alinhadas às categorias do treino + bin "outros".

    "Outros" = toda massa fora das categorias congeladas (categoria nova,
    NaN), com esperado 0.
    """
    n = len(series)
    if n == 0:
        raise ValueError("série vazia — sem dados para calcular PSI")
    counts = series.value_counts()
    observed = [counts.get(c, 0) / n for c in spec["categories"]]
    other = 1.0 - sum(observed)
    return observed + [other], list(spec["expected"]) + [0.0]


def observed_props_continuous(series: pd.Series, spec: dict) -> tuple[list, list]:
    """Proporções observadas nos cortes congelados + bin de NaN.

    Mesma construção do train.py: histograma em [-inf, *edges, +inf],
    proporções sobre TODAS as linhas (bins + NaN somam 1).
    """
    n = len(series)
    if n == 0:
        raise ValueError("série vazia — sem dados para calcular PSI")
    finite = series.dropna().to_numpy(dtype=float)
    bins = np.concatenate([[-np.inf], spec["edges"], [np.inf]])
    counts, _ = np.histogram(finite, bins=bins)
    observed = (counts / n).tolist() + [(n - len(finite)) / n]
    expected = list(spec["expected"]) + [spec["nan_expected"]]
    return observed, expected


def compute_psi(series: pd.Series, spec: dict) -> float:
    """PSI de uma feature de produção contra o spec congelado do treino."""
    if spec["kind"] == "categorical":
        observed, expected = observed_props_categorical(series, spec)
    else:
        observed, expected = observed_props_continuous(series, spec)
    return psi_from_props(observed, expected)


def ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """KS clássico de duas amostras: sup |ECDF_a(x) - ECDF_b(x)|.

    Implementação manual: avalia as duas ECDFs em todos os pontos das duas
    amostras (o sup só muda em pontos observados).
    """
    a = np.sort(np.asarray(sample_a, dtype=float))
    b = np.sort(np.asarray(sample_b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        raise ValueError("amostra vazia — sem dados para calcular KS")
    points = np.concatenate([a, b])
    ecdf_a = np.searchsorted(a, points, side="right") / len(a)
    ecdf_b = np.searchsorted(b, points, side="right") / len(b)
    return float(np.max(np.abs(ecdf_a - ecdf_b)))


def ks_vs_training(series: pd.Series, spec: dict) -> float:
    """KS binado contra o treino: max |CDF_prod - CDF_treino| nos cortes.

    CDF de treino nos cortes = cumsum dos expected (exata, pois os bins
    foram contados nesses mesmos cortes), renormalizada para a massa
    não-NaN. Lado da produção idem: ECDF dos valores finitos. Lower bound
    do KS verdadeiro (só enxerga os cortes congelados).
    """
    finite = np.sort(series.dropna().to_numpy(dtype=float))
    if len(finite) == 0:
        raise ValueError("série sem valores finitos — sem dados para calcular KS")
    edges = np.asarray(spec["edges"], dtype=float)
    expected = np.asarray(spec["expected"], dtype=float)
    # histograma usa bins [e_i, e_{i+1}): cumsum até o bin k-1 é a massa
    # estritamente abaixo do corte k => avaliar as duas CDFs com side="left".
    train_cdf = np.cumsum(expected)[:-1] / expected.sum()
    prod_cdf = np.searchsorted(finite, edges, side="left") / len(finite)
    return float(np.max(np.abs(prod_cdf - train_cdf)))


def retrain_status(alert_counts: list[int], k: int = RETRAIN_K,
                   m: int = RETRAIN_M) -> str:
    """Aplica a regra de retreino a uma janela de meses.

    `alert_counts`: nº de features com PSI > PSI_ALERT por mês, em ordem
    cronológica (último elemento = mês avaliado). Meses sem dados não
    entram na lista — sem medição não há como afirmar streak.

    - "red": os últimos `m` meses todos têm >= k features em alerta
      (streak completa => retreinar);
    - "yellow": o mês avaliado tem >= k features em alerta, mas a streak
      ainda não fechou (pode ser outlier de um mês só — monitorar);
    - "green": caso contrário.
    """
    recent = alert_counts[-m:]
    if len(recent) == m and all(count >= k for count in recent):
        return "red"
    if alert_counts and alert_counts[-1] >= k:
        return "yellow"
    return "green"


def compute_drift(df: pd.DataFrame, specs: dict) -> dict:
    """Drift de um lote de produção contra os specs congelados.

    Retorna {feature: {"psi": ..., "ks": ...}}; "ks" só para contínuas.
    """
    out = {}
    for col, spec in specs.items():
        entry = {"psi": compute_psi(df[col], spec)}
        if spec["kind"] == "continuous":
            entry["ks"] = ks_vs_training(df[col], spec)
        out[col] = entry
    return out