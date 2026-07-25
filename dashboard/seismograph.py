"""O sismógrafo: 11 pistas finas dividindo o mesmo eixo do tempo.

As 8 primeiras são PSI (alerta precoce — só olham inputs, não esperam ground
truth); depois de um divisor vêm MAE e viés (confirmação atrasada — só
existem depois que o feedback chega). Ver a defasagem entre as duas metades
é o argumento central da página.

Duas decisões de construção que explicam o resto do módulo:

1. **Um gráfico só, facetado por pista** — não 11 gráficos empilhados. Assim
   o eixo do tempo, a faixa da COVID e a agulha ficam alinhados por
   construção; gráficos separados desalinham porque cada um dimensiona seu
   próprio eixo.
2. **Toda anotação mora na mesma tabela das pistas**, marcada por uma coluna
   `kind`. Só a tabela do facet é fatiada por pista: uma camada com dados
   próprios apareceria repetida em todas as células, e o rótulo do holdout
   acabaria dentro da pista do `is_holiday`.

Sem eixo y: em 46 px o eixo vira ruído. A escala de cada pista vem das
linhas de referência anotadas (PSI 0.2, MAE do holdout, viés zero) e do pico
no canto direito; o valor exato está no tooltip e nas tabelas abaixo.
"""

import altair as alt
import pandas as pd

from theme import (
    INK,
    LEVEL_LABEL,
    MARK_COLOR,
    MARK_SIZE,
    MONO,
    MUTED,
    NEEDLE,
    PANEL,
    SANS,
)

# Janela da COVID destacada em todas as pistas. Mar-jun/2020 é o recorte do
# choque (lockdown de NY em 22/mar; reabertura em fases a partir de jun).
COVID_START, COVID_END = "2020-03-01", "2020-07-01"

# Pista fina o bastante para as 11 caberem numa tela sem rolagem. Largura
# fixa porque width="container" não vale dentro de facet no Vega-Lite.
TRACK_W, TRACK_H = 980, 46

DIVIDER = " "  # pista vazia que separa alerta precoce de confirmação
MAE = "MAE (min)"
BIAS = "viés (min)"


def track_order(features: list[str]) -> list[str]:
    return [*features, DIVIDER, MAE, BIAS]


def _tracks(timeline) -> pd.DataFrame:
    """PSI, MAE e viés empilhados: uma linha por (pista, mês)."""
    psi = timeline.drift.rename(columns={"feature": "track", "psi": "value"})
    psi = psi[["track", "date", "month", "value", "level", "ks"]].copy()
    psi["display"] = psi["value"].map("PSI {:.3f}".format)
    psi["reading"] = psi["level"].map(LEVEL_LABEL)

    def performance(column: str, name: str, fmt: str) -> pd.DataFrame:
        frame = timeline.performance[["date", "month", column]].rename(
            columns={column: "value"}).assign(track=name, level="neutro",
                                              ks=None, reading="—")
        frame["display"] = frame["value"].map(fmt.format)
        return frame

    return pd.concat([psi,
                      performance("mae_min", MAE, "MAE {:.2f} min"),
                      performance("bias_min", BIAS, "viés {:+.2f} min")],
                     ignore_index=True)


def _marks(tracks: pd.DataFrame, timeline, selected: str) -> pd.DataFrame:
    """Faixa da COVID, agulha, linhas de referência, rótulos e âncoras."""
    order = track_order(timeline.features)
    alert = timeline.thresholds["alert"]
    holdout = timeline.holdout["mae_min"]
    rows = []

    rows += [{"track": t, "kind": "band", "date": pd.Timestamp(COVID_START),
              "date2": pd.Timestamp(COVID_END)} for t in order]
    rows.append({"track": timeline.features[0], "kind": "bandlabel",
                 "date": pd.Timestamp(COVID_START), "text": "covid-19"})

    needle = pd.Timestamp(f"{selected}-01")
    rows += [{"track": t, "kind": "needle", "date": needle} for t in order]

    # Referências: o que o olho compara em cada metade do sismógrafo.
    rows += [{"track": f, "kind": "ref", "value": alert}
             for f in timeline.features]
    rows.append({"track": timeline.features[0], "kind": "reflabel",
                 "value": alert, "text": f"PSI {alert:g} · alerta"})
    rows += [{"track": MAE, "kind": "ref", "value": holdout},
             {"track": MAE, "kind": "reflabel", "value": holdout,
              "text": f"holdout {holdout:.2f}"},
             {"track": BIAS, "kind": "ref", "value": 0.0},
             {"track": BIAS, "kind": "reflabel", "value": 0.0,
              "text": "sem viés"}]

    # Âncoras invisíveis fixam o domínio y de cada pista (as escalas são
    # independentes): PSI sempre a partir do zero e com espaço para a linha
    # de alerta, mesmo numa feature que nunca saiu de 0.03 — sem isso, ruído
    # de feature estável viraria montanha.
    for feature in timeline.features:
        peak = tracks.loc[tracks["track"] == feature, "value"].max()
        rows += [{"track": feature, "kind": "anchor", "value": 0.0},
                 {"track": feature, "kind": "anchor",
                  "value": max(peak * 1.12, alert * 1.25)},
                 {"track": feature, "kind": "peak",
                  "text": f"pico {peak:.2f}"}]
    for name, pad in ((MAE, 0.25), (BIAS, 0.5)):
        values = tracks.loc[tracks["track"] == name, "value"]
        rows += [{"track": name, "kind": "anchor", "value": values.min() - pad},
                 {"track": name, "kind": "anchor", "value": values.max() + pad}]

    rows.append({"track": DIVIDER, "kind": "divider",
                 "text": "confirmação atrasada — só existe depois que o "
                         "feedback chega"})
    return pd.DataFrame(rows)


def chart(timeline, selected: str) -> alt.Chart:
    """Sismógrafo completo, com a agulha no mês selecionado."""
    tracks = _tracks(timeline)
    data = pd.concat([tracks.assign(kind="trace"),
                      _marks(tracks, timeline, selected)], ignore_index=True)

    # Camadas sem dados próprios: quem carrega a tabela é o facet.
    base = alt.Chart().properties(width=TRACK_W, height=TRACK_H)
    x = alt.X("date:T", title=None,
              axis=alt.Axis(format="%m/%y", labelAngle=0, tickSize=3,
                            tickCount={"interval": "month", "step": 3}))
    y = alt.Y("value:Q", title=None, axis=None)
    hover = alt.selection_point(fields=["date"], on="pointerover",
                                clear="pointerout", empty=False)

    def layer(kind: str) -> alt.Chart:
        return base.transform_filter(alt.datum.kind == kind)

    # x com o MESMO eixo das outras camadas: num gráfico em camadas os eixos
    # de um canal são mesclados, e uma camada com axis=None apagaria o eixo
    # do tempo da página inteira.
    band = layer("band").mark_rect(fill=INK, opacity=0.045).encode(
        x=x, x2="date2:T")
    band_label = layer("bandlabel").mark_text(
        align="left", baseline="top", dx=4, dy=2, color=MUTED,
        font=SANS, fontSize=9).encode(x=x, y=alt.value(0), text="text:N")
    reference = layer("ref").mark_rule(
        color=MUTED, opacity=0.45, strokeDash=[3, 3]).encode(y=y)
    reference_label = layer("reflabel").mark_text(
        align="left", baseline="bottom", dx=2, dy=-3, color=MUTED,
        font=MONO, fontSize=9).encode(x=alt.value(0), y=y, text="text:N")
    anchors = layer("anchor").mark_point(opacity=0).encode(y=y)
    peak = layer("peak").mark_text(
        align="right", baseline="top", dx=-2, dy=1, color=MUTED,
        font=MONO, fontSize=9).encode(
        x=alt.value(TRACK_W), y=alt.value(0), text="text:N")
    divider = layer("divider").mark_text(
        align="center", baseline="middle", color=MUTED, font=SANS,
        fontSize=10).encode(x=alt.value(TRACK_W / 2),
                            y=alt.value(TRACK_H / 2), text="text:N")

    needle = layer("needle").mark_rule(color=NEEDLE, strokeWidth=1.5,
                                       opacity=0.9).encode(x=x)
    crosshair = layer("trace").mark_rule(
        color=INK, opacity=0.28).encode(x=x).transform_filter(hover)
    trace = layer("trace").mark_line(
        color=MUTED, strokeWidth=1.5, opacity=0.55).encode(x=x, y=y)
    # Severidade em DOIS canais (cor e tamanho): sob daltonismo o âmbar e o
    # vermelho continuam separados, e o marcador em alerta salta à vista.
    points = layer("trace").mark_point(
        filled=True, stroke=PANEL, strokeWidth=1.5).encode(
        x=x, y=y,
        color=alt.Color("level:N", legend=None, scale=alt.Scale(
            domain=list(MARK_COLOR), range=list(MARK_COLOR.values()))),
        size=alt.Size("level:N", legend=None, scale=alt.Scale(
            domain=list(MARK_SIZE), range=list(MARK_SIZE.values()))))
    # Alvo de hover largo (~40 px) — ninguém acerta um ponto de 8 px.
    hit = layer("trace").mark_rule(opacity=0, strokeWidth=40).encode(
        x=x, tooltip=[alt.Tooltip("month:N", title="mês"),
                      alt.Tooltip("track:N", title="pista"),
                      alt.Tooltip("display:N", title="valor"),
                      alt.Tooltip("reading:N", title="leitura")],
    ).add_params(hover)

    return alt.layer(
        band, reference, needle, crosshair, trace, points, hit,
        band_label, reference_label, anchors, peak, divider,
    ).facet(
        row=alt.Row("track:N", sort=track_order(timeline.features), title=None,
                    header=alt.Header(orient="left", labelAngle=0,
                                      labelAlign="left", labelPadding=10,
                                      labelLimit=160)),
        data=data,
        spacing=4,
    ).resolve_scale(y="independent")
