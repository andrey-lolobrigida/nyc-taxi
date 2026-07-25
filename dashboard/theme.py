"""Paleta, tipografia e as duas peles que elas vestem: o tema do Altair e o
CSS da página.

Fonte da verdade da identidade visual (design aprovado no CLAUDE.md). O
`.streamlit/config.toml` repete só as cores que o Streamlit desenha sozinho
(chrome e widgets), porque ele lê o tema antes de qualquer Python rodar.

A severidade é sempre codificada em mais de um canal — cor E tamanho do
marcador no gráfico, cor E palavra ("estável"/"atenção"/"alerta") nas
tabelas. Cor sozinha não informa quem não distingue âmbar de vermelho.
"""

import altair as alt

BG = "#131417"
PANEL = "#1B1D21"
INK = "#E6E6E2"
MUTED = "#8A8F98"
HAIRLINE = "#2A2D33"
STABLE = "#8A8F98"
WATCH = "#D9A013"
ALERT = "#C4442C"
NEEDLE = "#4C8DFF"

SANS = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"

LEVEL_COLOR = {"stable": STABLE, "watch": WATCH, "alert": ALERT}
LEVEL_LABEL = {"stable": "estável", "watch": "atenção", "alert": "alerta"}
# As pistas de MAE/viés não têm severidade — os thresholds do projeto são de
# PSI. Marcador neutro ali, e a interpretação fica com a linha do holdout.
MARK_COLOR = {**LEVEL_COLOR, "neutro": MUTED}
MARK_SIZE = {"stable": 36, "watch": 90, "alert": 130, "neutro": 40}

STATUS_LABEL = {"green": "verde", "yellow": "amarelo", "red": "vermelho"}
STATUS_COLOR = {"green": STABLE, "yellow": WATCH, "red": ALERT}
STATUS_READING = {
    "green": "sem streak — nenhuma ação",
    "yellow": "mês em alerta, streak não fechou — monitorar",
    "red": "streak fechada — retreinar",
}


@alt.theme.register("taxi_dark", enable=True)
def taxi_dark() -> alt.theme.ThemeConfig:
    """Tema dos gráficos: grade e eixos recessivos, rótulos em mono."""
    return {
        "config": {
            "background": BG,
            "font": SANS,
            "view": {"stroke": None},
            "axis": {
                "domain": False,
                "grid": False,
                "tickColor": HAIRLINE,
                "labelColor": MUTED,
                "labelFont": MONO,
                "labelFontSize": 10,
                "titleColor": MUTED,
            },
            "header": {
                "labelColor": INK,
                "labelFont": MONO,
                "labelFontSize": 11,
                "titleColor": MUTED,
            },
        }
    }


CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  h1, h2, h3, p, td, th {{ color: {INK}; }}
  .masthead {{ display:flex; align-items:baseline; gap:14px;
               border-bottom:1px solid {HAIRLINE}; padding-bottom:10px; }}
  .masthead h1 {{ font: 600 21px/1.2 {SANS}; letter-spacing:-.01em; margin:0; }}
  .masthead .meta {{ font:11px/1.4 {MONO}; color:{MUTED}; margin-left:auto; }}
  .key {{ display:flex; gap:16px; font:10px/1 {MONO}; color:{MUTED};
          margin:10px 0 2px; }}
  .dot {{ display:inline-block; width:7px; height:7px; border-radius:50%;
          margin-right:6px; vertical-align:middle; }}
  .panel {{ border:1px solid {HAIRLINE}; background:{PANEL};
            padding:14px 16px 4px; height:100%; }}
  .panel-title {{ font:10px/1 {SANS}; letter-spacing:.16em;
                  text-transform:uppercase; color:{MUTED};
                  border-bottom:1px solid {HAIRLINE}; padding-bottom:9px;
                  margin-bottom:10px; }}
  table.form {{ width:100%; border-collapse:collapse; font:12px/1.5 {MONO}; }}
  table.form th {{ font:9px/1 {SANS}; letter-spacing:.1em;
                   text-transform:uppercase; color:{MUTED}; font-weight:500;
                   text-align:left; padding:0 0 6px; }}
  table.form td {{ padding:3px 0; border-top:1px solid #212429;
                   color:{INK}; }}
  table.form .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  table.form .ref {{ color:{MUTED}; }}
  .foot {{ font:10px/1.5 {SANS}; color:{MUTED}; margin:10px 0 12px; }}
  .verdict {{ display:flex; align-items:center; gap:10px; margin-bottom:12px; }}
  .pill {{ font:10px/1 {SANS}; letter-spacing:.14em; text-transform:uppercase;
           border:1px solid; border-radius:2px; padding:5px 9px; }}
  .verdict-text {{ font:11px/1.4 {SANS}; color:{MUTED}; }}
  div[data-testid="stButton"] button {{ width:100%; border-radius:2px;
      border:1px solid {HAIRLINE}; background:{PANEL}; color:{INK};
      font:10px/1 {SANS}; letter-spacing:.14em; text-transform:uppercase;
      padding:12px 0; }}
  div[data-testid="stButton"] button:hover {{ border-color:{NEEDLE};
      color:{NEEDLE}; }}
</style>
"""
