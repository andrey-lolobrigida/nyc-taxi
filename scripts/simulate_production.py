"""Replay de produção: reproduz 2020-2021 mês a mês contra a API (decisão #4).

Para cada mês:
1. Carrega o parquet cru, aplica a MESMA limpeza do treino (janela de pickup
   + cortes de duração do features.py — mantém as métricas de performance
   comparáveis com o holdout de 2019);
2. Amostra estratificada por dia (proporção diária preservada, seed fixa
   derivada do mês — decisão #4: preserva as distribuições que PSI/KS
   enxergam e mantém o replay rápido);
3. Envia cada corrida para POST /predict, em ordem cronológica;
4. Depois de TODOS os predicts do mês, envia a duração real para
   POST /feedback — o "delay" simulado é estrutural: nenhum ground truth
   chega antes do mês inteiro ter sido previsto;
5. Imprime o relatório de monitoramento do mês (drift, performance,
   retrain-check) consumindo a própria API — o script é só mais um cliente,
   nunca toca no SQLite.

Sanity check obrigatório (CLAUDE.md): rodar PRIMEIRO os meses de controle
    uv run python scripts/simulate_production.py --months 2020-01 2020-02
e conferir PSI < 0.1 antes de interpretar qualquer mês de pandemia.
PSI alto no controle = bug na implementação (provavelmente bins), não drift.

Uso típico (API de pé em outro terminal: uv run uvicorn api.main:app):
    uv run python scripts/simulate_production.py --months 2020-01 2020-02
    uv run python scripts/simulate_production.py --start 2020-03 --end 2021-12
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_download import month_range                                # noqa: E402
from features import clean_duration_mask, compute_duration_minutes   # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
RAW_COLUMNS = ["lpep_pickup_datetime", "lpep_dropoff_datetime",
               "PULocationID", "DOLocationID", "trip_type"]

DEFAULT_SAMPLE_SIZE = 30_000  # decisão em aberto do CLAUDE.md: 30-50k/mês
CONTROL_MONTHS = {"2020-01", "2020-02"}  # grupo de controle temporal

LEVEL_MARK = {"stable": " ", "watch": "~", "alert": "!"}
STATUS_MARK = {"green": "🟢", "yellow": "🟡", "red": "🔴"}


def api_call(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            # 500 do uvicorn vem como texto puro ("Internal Server Error")
            return e.code, {"detail": body}


def load_month_trips(month: str) -> pd.DataFrame:
    path = RAW_DIR / f"green_tripdata_{month}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"parquet ausente: {path} (rode data_download.py)")
    df = pd.read_parquet(path, columns=RAW_COLUMNS)

    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    pickup = df["lpep_pickup_datetime"]
    duration = compute_duration_minutes(df)
    keep = (pickup >= start) & (pickup < end) & clean_duration_mask(duration)

    out = df.loc[keep, ["lpep_pickup_datetime", "PULocationID",
                        "DOLocationID", "trip_type"]].copy()
    out["actual_duration_min"] = duration.loc[keep]
    return out.reset_index(drop=True)


def stratified_sample(trips: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Amostra ~n corridas preservando a proporção diária (frac constante
    por dia). Mês menor que n vai inteiro. Sempre em ordem cronológica —
    o replay imita requests chegando na sequência real."""
    if len(trips) > n:
        rng = np.random.default_rng(seed)
        trips = (trips.groupby(trips["lpep_pickup_datetime"].dt.date,
                               group_keys=False)
                      .sample(frac=n / len(trips), random_state=rng))
    return trips.sort_values("lpep_pickup_datetime").reset_index(drop=True)


def replay_month(api_url: str, trips: pd.DataFrame, workers: int) -> dict:
    """Fase 1: todos os predicts (em paralelo, ordem de submissão
    cronológica). Fase 2: todos os feedbacks. Retorna contadores."""

    def do_predict(row) -> tuple[str | None, float]:
        status, body = api_call("POST", f"{api_url}/predict", {
            "pickup_datetime": row.lpep_pickup_datetime.isoformat(),
            "PULocationID": int(row.PULocationID),
            "DOLocationID": int(row.DOLocationID),
            "trip_type": None if pd.isna(row.trip_type) else int(row.trip_type),
        })
        if status != 200:
            return None, 0.0
        return body["prediction_id"], row.actual_duration_min

    def do_feedback(pair) -> bool:
        prediction_id, actual = pair
        status, _ = api_call("POST", f"{api_url}/feedback", {
            "prediction_id": prediction_id,
            "actual_duration_min": round(float(actual), 2),
        })
        return status == 200

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(do_predict, trips.itertuples(index=False)))
    predict_elapsed = time.perf_counter() - started

    ok_pairs = [(pid, actual) for pid, actual in results if pid is not None]

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        feedback_ok = sum(pool.map(do_feedback, ok_pairs))
    feedback_elapsed = time.perf_counter() - started

    return {
        "predict_ok": len(ok_pairs),
        "predict_fail": len(results) - len(ok_pairs),
        "feedback_ok": feedback_ok,
        "feedback_fail": len(ok_pairs) - feedback_ok,
        "predict_elapsed": predict_elapsed,
        "feedback_elapsed": feedback_elapsed,
    }


def print_month_report(api_url: str, month: str) -> None:
    _, drift = api_call("GET", f"{api_url}/monitoring/drift?month={month}")
    _, perf = api_call("GET", f"{api_url}/monitoring/performance?month={month}")
    _, check = api_call("GET", f"{api_url}/monitoring/retrain-check?month={month}")

    control = "  (grupo de controle — esperado PSI < 0.1)" \
        if month in CONTROL_MONTHS else ""
    print(f"\n--- monitoramento {month}{control}")
    for name, entry in drift["features"].items():
        ks = f"  ks={entry['ks']:.3f}" if entry["ks"] is not None else ""
        print(f"  {LEVEL_MARK[entry['level']]} {name:<18} "
              f"psi={entry['psi']:.4f}{ks}")
    print(f"  performance: mae={perf['mae_min']:.2f} min "
          f"(holdout {perf['holdout_reference']['mae_min']:.2f})  "
          f"rmse={perf['rmse_min']:.2f}  bias={perf['bias_min']:+.2f}  "
          f"n={perf['n_with_feedback']}/{perf['n_predictions']}")
    alerts = ", ".join(f"{m['month']}:{m['n_alert']}" for m in check["months"])
    print(f"  retrain-check: {STATUS_MARK[check['status']]} "
          f"{check['status']}  (features em alerta por mês: {alerts})")


def month_already_replayed(api_url: str, month: str) -> bool:
    status, _ = api_call("GET", f"{api_url}/monitoring/drift?month={month}")
    return status == 200


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--months", nargs="+", metavar="YYYY-MM",
                       help="meses explícitos, ex: --months 2020-01 2020-02")
    group.add_argument("--start", metavar="YYYY-MM",
                       help="início do intervalo (use com --end)")
    parser.add_argument("--end", metavar="YYYY-MM",
                        help="fim do intervalo, inclusivo")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="corridas/mês (estratificado por dia; mês menor "
                             "vai inteiro) [%(default)s]")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000",
                        help="[%(default)s]")
    parser.add_argument("--workers", type=int, default=8,
                        help="requests concorrentes [%(default)s]")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed base da amostragem (derivada por mês) "
                             "[%(default)s]")
    parser.add_argument("--force", action="store_true",
                        help="reenvia mês que já tem predições logadas "
                             "(CUIDADO: duplica linhas e distorce métricas)")
    args = parser.parse_args()
    if bool(args.start) != bool(args.end):
        parser.error("--start e --end andam juntos")
    months = args.months or month_range(args.start, args.end)

    status, health = api_call("GET", f"{args.api_url}/health")
    print(f"API ok — modelo {health['model_version']} "
          f"(treinado em {health['trained_at']})")

    for month in months:
        if month_already_replayed(args.api_url, month) and not args.force:
            print(f"\n### {month}: já tem predições logadas — pulando "
                  "(use --force para duplicar mesmo assim)")
            print_month_report(args.api_url, month)
            continue

        trips = load_month_trips(month)
        # Seed determinística por mês: rodar 2020-04 isolado ou no meio do
        # intervalo produz a mesma amostra.
        sample = stratified_sample(trips, args.sample_size,
                                   args.seed + int(month.replace("-", "")))
        print(f"\n### {month}: {len(trips)} corridas limpas -> "
              f"replay de {len(sample)}")

        counters = replay_month(args.api_url, sample, args.workers)
        rate = counters["predict_ok"] / max(counters["predict_elapsed"], 1e-9)
        print(f"  predict:  {counters['predict_ok']} ok, "
              f"{counters['predict_fail']} falhas "
              f"({counters['predict_elapsed']:.0f}s, {rate:.0f} req/s)")
        print(f"  feedback: {counters['feedback_ok']} ok, "
              f"{counters['feedback_fail']} falhas "
              f"({counters['feedback_elapsed']:.0f}s)")

        print_month_report(args.api_url, month)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        sys.exit(f"API inacessível ({e.reason}) — suba com: "
                 "uv run uvicorn api.main:app")