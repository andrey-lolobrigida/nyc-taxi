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
    num,
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

# MAE e viés são o desfecho do argumento — o buraco entre o traçado e a
# referência é o que o dashboard existe para mostrar. Numa pista de 46 px por
# 980, +1,5 min de MAE vira uma inclinação que o olho lê como plana.
PERF_H = 96

# Largura fixa da coluna de rótulos: é o que mantém os dois blocos alinhados
# no eixo do tempo depois do vconcat.
HEADER_W = 160

DIVIDER = " "  # pista vazia que separa alerta precoce de confirmação
MAE = "MAE (min)"
BIAS = "viés (min)"


def _mmyy(date) -> str:
    """Mês no mesmo formato do eixo do tempo (01/20), não em ISO cru."""
    return pd.Timestamp(date).strftime("%m/%y")


def track_order(features: list[str]) -> list[str]:
    return [*features, DIVIDER, MAE, BIAS]


def _tracks(timeline) -> pd.DataFrame:
    """PSI, MAE e viés empilhados: uma linha por (pista, mês)."""
    psi = timeline.drift.rename(columns={"feature": "track", "psi": "value"})
    psi = psi[["track", "date", "month", "value", "level", "ks"]].copy()
    psi["display"] = psi["value"].map(
        lambda v: f'PSI {num(v, "{:.3f}")}')
    psi["reading"] = psi["level"].map(LEVEL_LABEL)

    def performance(column: str, name: str, template: str, numfmt: str,
                    baseline: float) -> pd.DataFrame:
        frame = timeline.performance[["date", "month", column]].rename(
            columns={column: "value"}).assign(track=name, level="neutro",
                                              ks=None, reading="—",
                                              baseline=baseline)
        # O número é formatado ANTES de entrar no template: num() devolve
        # texto, e um texto não passa por um format spec numérico.
        frame["display"] = frame["value"].map(
            lambda v: template.format(num(v, numfmt)))
        return frame

    # Só as pistas de desempenho têm um "de onde deveria estar": o MAE do
    # holdout e o viés zero. As de PSI não têm baseline (NaN) e por isso não
    # ganham área preenchida — nelas o que importa é cruzar o limiar, não a
    # distância até ele.
    psi["baseline"] = float("nan")
    return pd.concat([psi,
                      performance("mae_min", MAE, "MAE {} min", "{:.2f}",
                                  timeline.holdout["mae_min"]),
                      performance("bias_min", BIAS, "viés {} min", "{:+.2f}",
                                  0.0)],
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
                 "value": alert,
                 "text": f'PSI {num(alert, "{:g}")} · alerta'})
    rows += [{"track": MAE, "kind": "ref", "value": holdout},
             {"track": MAE, "kind": "reflabel", "value": holdout,
              "text": f"holdout {num(holdout)}"},
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
                  "text": f"pico {num(peak)}"}]
    for name, pad in ((MAE, 0.25), (BIAS, 0.5)):
        values = tracks.loc[tracks["track"] == name, "value"]
        rows += [{"track": name, "kind": "anchor", "value": values.min() - pad},
                 {"track": name, "kind": "anchor", "value": values.max() + pad}]

    # O fecho da história: o MAE dos últimos meses contra o dos primeiros —
    # que são o grupo de controle (jan-fev/2020, ainda mundo pré-pandemia).
    # Sem este rótulo a comparação exige que o olho meça 26 px de diferença
    # ao longo de 980 px de largura, e ninguém mede.
    perf = timeline.performance.sort_values("date")
    if len(perf) >= 4:
        head, tail = perf.head(2), perf.tail(2)
        delta = tail["mae_min"].mean() / head["mae_min"].mean() - 1
        rows.append({
            "track": MAE, "kind": "peak",
            "text": "{} · {} — {:+.0%} vs {} · {}".format(
                num(tail["mae_min"].iloc[0]), num(tail["mae_min"].iloc[1]),
                delta, _mmyy(head["date"].iloc[0]),
                _mmyy(head["date"].iloc[1]))})

    rows.append({"track": DIVIDER, "kind": "divider",
                 "text": "confirmação atrasada — só existe depois que o "
                         "feedback chega"})
    return pd.DataFrame(rows)


def _block(data, tracks: list[str], height: int, x_axis, key: str,
           x_domain):
    """Um bloco de pistas com altura própria, facetado por pista.

    As pistas de PSI e as de desempenho viram DOIS blocos porque um facet do
    Vega-Lite impõe a mesma altura a todas as linhas — e MAE e viés precisam
    de mais altura que `is_weekend`: são o desfecho do argumento, não ruído
    de fundo. O eixo do tempo continua compartilhado porque os dois blocos
    têm a mesma largura, o mesmo domínio de x e a mesma coluna de rótulos
    (labelLimit fixo); só o bloco de baixo desenha o eixo.
    """
    base = alt.Chart().properties(width=TRACK_W, height=height)
    # Domínio de x FIXO e idêntico nos dois blocos. Sem isso cada facet
    # deduz o domínio dos seus próprios dados, e basta um mês sem feedback
    # para o bloco de desempenho encolher e desalinhar do de PSI — perdendo
    # justamente o eixo compartilhado que faz a defasagem ser legível.
    x = alt.X("date:T", title=None, axis=x_axis,
              scale=alt.Scale(domain=x_domain, nice=False))
    y = alt.Y("value:Q", title=None, axis=None)
    # Nome próprio por bloco. Compartilhar a seleção entre os dois blocos
    # daria um crosshair que atravessa do PSI para o MAE, mas não dá: nome
    # repetido vira sinal duplicado e quebra o parser do Vega. O custo é que
    # o crosshair de hover só percorre o bloco sob o cursor — a agulha do mês
    # selecionado, essa sim, continua atravessando os dois.
    hover = alt.selection_point(name=f"hover_{key}", fields=["date"],
                                on="pointerover", clear="pointerout",
                                empty=False)

    def layer(kind: str) -> alt.Chart:
        return base.transform_filter(alt.datum.kind == kind)

    # `name` explícito: ao unificar os parâmetros do vconcat o Altair lê o
    # nome da primeira camada, e sem ele estoura num AttributeError obscuro.
    band = layer("band").mark_rect(fill=INK, opacity=0.045).encode(
        x=x, x2="date2:T").properties(name=f"band_{key}")
    band_label = layer("bandlabel").mark_text(
        align="left", baseline="top", dx=4, dy=2, color=MUTED,
        font=SANS, fontSize=9).encode(x=x, y=alt.value(0), text="text:N")
    # A degradação É a distância entre o traçado e a referência, e comparar
    # duas linhas separadas é a tarefa visual mais cara que existe. Preenchida,
    # a mesma diferença vira área — uma quantidade, não uma inferência.
    # isValid() deixa as pistas de PSI de fora (baseline NaN).
    gap = layer("trace").transform_filter(
        "isValid(datum.baseline)").mark_area(
        color=MUTED, opacity=0.22, line=False).encode(
        x=x, y=y, y2="baseline:Q")
    reference = layer("ref").mark_rule(
        color=MUTED, opacity=0.45, strokeDash=[3, 3]).encode(y=y)
    reference_label = layer("reflabel").mark_text(
        align="left", baseline="bottom", dx=2, dy=-3, color=MUTED,
        font=MONO, fontSize=9).encode(x=alt.value(0), y=y, text="text:N")
    anchors = layer("anchor").mark_point(opacity=0).encode(y=y)
    peak = layer("peak").mark_text(
        align="right", baseline="top", dx=-6, dy=1, color=MUTED,
        font=MONO, fontSize=9).encode(
        x=alt.value(TRACK_W), y=alt.value(0), text="text:N")
    divider = layer("divider").mark_text(
        align="center", baseline="middle", color=MUTED, font=SANS,
        fontSize=10).encode(x=alt.value(TRACK_W / 2),
                            y=alt.value(height / 2), text="text:N")

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
        band, gap, reference, needle, crosshair, trace, points, hit,
        band_label, reference_label, anchors, peak, divider,
    ).facet(
        row=alt.Row("track:N", sort=tracks, title=None,
                    header=alt.Header(orient="left", labelAngle=0,
                                      labelAlign="left", labelPadding=10,
                                      labelLimit=HEADER_W)),
        data=data[data["track"].isin(tracks)],
        spacing=4,
    ).resolve_scale(y="independent")


def chart(timeline, selected: str) -> alt.Chart:
    """Sismógrafo completo, com a agulha no mês selecionado."""
    tracks = _tracks(timeline)
    data = pd.concat([tracks.assign(kind="trace"),
                      _marks(tracks, timeline, selected)], ignore_index=True)

    axis = alt.Axis(format="%m/%y", labelAngle=0, tickSize=3,
                    tickCount={"interval": "month", "step": 3})
    span = pd.to_datetime(data["date"].dropna())
    x_domain = [d.isoformat() for d in (span.min(), span.max())]
    # Só o bloco de baixo desenha o eixo do tempo; o de cima o esconde para
    # os dois ficarem colados. Ambos têm a mesma largura e o mesmo domínio,
    # então as colunas de mês continuam alinhadas de ponta a ponta.
    early = _block(data, [*timeline.features, DIVIDER], TRACK_H,
                   alt.Axis(labels=False, ticks=False, domain=False,
                            title=None), "early", x_domain)
    late = _block(data, [MAE, BIAS], PERF_H, axis, "late", x_domain)
    return alt.vconcat(early, late, spacing=4).resolve_scale(y="independent")
