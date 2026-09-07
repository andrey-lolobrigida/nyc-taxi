"""Renderiza o sismógrafo do dashboard em PNG, para o README e para revisão.

Existe por dois motivos:

1. O README precisa da imagem-tese (a defasagem entre drift e performance)
   sem depender de alguém lembrar de tirar print. Decisão #1 do CLAUDE.md
   vale aqui também: output materializado por script re-executável.
2. É o único jeito de conferir o gráfico sem navegador. O que este caminho
   NÃO alcança é o CSS dos painéis — isso só no navegador.

Reaproveita as MESMAS peças do dashboard (`monitoring.timeline` e
`seismograph.chart`), nunca uma reimplementação parecida: um PNG que
divergisse da tela seria pior que nenhum PNG.

Uso (com a API de pé e o replay já rodado):
    uv run --with vl-convert-python python scripts/render_seismograph.py
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

DEFAULT_OUT = ROOT / "reports" / "monitoring" / "seismograph.png"
DEFAULT_MONTH = "2021-12"  # a agulha no mês em que os dois sinais discordam


def drop_nan(node):
    """Troca NaN por None recursivamente na spec do Vega-Lite.

    `chart.to_json()` do Altair emite o literal `NaN`, que não é JSON
    válido: o vl-convert recusa a spec inteira. O navegador engole porque
    ali a spec é objeto JS, não texto. A coluna que dispara isso é `ks`,
    nula para as features categóricas.
    """
    if isinstance(node, float) and math.isnan(node):
        return None
    if isinstance(node, dict):
        return {key: drop_nan(value) for key, value in node.items()}
    if isinstance(node, list):
        return [drop_nan(value) for value in node]
    return node


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000",
                        help="URL da API de monitoramento")
    parser.add_argument("--month", default=DEFAULT_MONTH,
                        help="mês da agulha (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scale", type=float, default=1.5,
                        help="fator de resolução do PNG")
    args = parser.parse_args()

    try:
        import vl_convert
    except ImportError:
        sys.exit("falta o vl-convert-python. Rode com:\n"
                 "  uv run --with vl-convert-python python "
                 "scripts/render_seismograph.py")

    import monitoring
    import seismograph

    try:
        # .__wrapped__ contorna o cache do Streamlit, que fora de uma sessão
        # só existe para emitir avisos.
        timeline = monitoring.timeline.__wrapped__(args.api_url)
    except Exception as error:  # noqa: BLE001 — a causa importa mais que o tipo
        sys.exit(f"não consegui ler {args.api_url}/monitoring/timeline: "
                 f"{type(error).__name__}: {error}\n"
                 "A API está no ar e o replay já rodou?")

    if args.month not in timeline.months:
        sys.exit(f"{args.month} não está na timeline "
                 f"({timeline.months[0]}..{timeline.months[-1]})")

    spec = json.dumps(drop_nan(seismograph.chart(timeline, args.month).to_dict()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(vl_convert.vegalite_to_png(spec, scale=args.scale))
    print(f"{args.out.relative_to(ROOT)} — {len(timeline.months)} meses, "
          f"agulha em {args.month}")


if __name__ == "__main__":
    main()
