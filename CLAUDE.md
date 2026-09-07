# Projeto: NYC Taxi MLOps — Detecção de Drift em Produção

## Objetivo do projeto

Projeto de aprendizado prático de MLOps, cobrindo o ciclo completo: treino →
deploy → monitoramento → detecção de drift → (gatilho de) retreino. O foco
principal está na fase 3 (monitoramento), não no modelo em si.

O modelo em si (regressão de duração de corrida de táxi) é só um veículo para
observar, de forma **real e não sintética**, o que acontece com um modelo em
produção quando o mundo muda debaixo dele.

## Por que esse dataset

Dataset: **NYC TLC Trip Record Data — Green Taxi** (dados oficiais de
corridas de táxi verde de Nova York, disponibilizados mensalmente em parquet).

- Fonte: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Por que green e não yellow**: mesmo schema, mesma fonte, mesma história
  de pandemia — mas arquivos ~10x menores (centenas de milhares de corridas
  por mês em vez de ~7 milhões). Isso torna o replay via HTTP viável e o
  ciclo de iteração muito mais rápido, sem perder nada do valor didático.
- Recorte temporal escolhido: **jan/2019 a dez/2021**
  - Treino: dados de 2019 (mundo "normal")
  - Produção simulada: 2020 e 2021, mês a mês
  - **Grupo de controle temporal**: jan e fev de 2020 ainda são "mundo
    normal". Se o PSI já estiver alto nesses meses, o bug está na
    implementação do drift (provavelmente nos bins), não nos dados. Usar
    esses dois meses como sanity check antes de confiar nos números da
    pandemia.
- Motivo da escolha do recorte: atravessa a pandemia de COVID-19, um evento
  real e documentado que causa:
  - **Data drift** óbvio: volume de corridas despenca em mar/abr 2020,
    distribuição de distâncias e horários muda, picos de rush hour somem.
  - **Concept drift** mais sutil: a relação entre features e alvo também
    muda (ruas vazias → viagens mais rápidas mesmo em distâncias parecidas
    às de antes).
  - Isso dá uma história real pro dashboard contar, em vez de um drift
    sintético forçado (tipo adicionar ruído artificial nos dados).

Alvo do modelo: `trip_duration` (em segundos ou minutos) — regressão,
derivado de `lpep_dropoff_datetime - lpep_pickup_datetime`.

## ⚠️ Armadilha conhecida: vazamento de dados (data leakage)

A coluna `trip_distance` do dataset é a distância **efetivamente percorrida**
registrada pelo taxímetro — um valor que só existe quando a corrida termina.
Usá-la como feature é ver o futuro: o modelo fica suspeitamente bom e
inutilizável em produção real.

**Regra**: só podem entrar como features informações disponíveis **no momento
do embarque** (pickup datetime, zona de origem, zona de destino, etc.).

Como substituto da distância real:
1. Desde ~2016 a TLC não publica mais lat/lon, só `PULocationID` e
   `DOLocationID` (IDs de taxi zones).
2. Baixar o **shapefile de taxi zones** da própria TLC (mesma página do
   dataset).
3. Calcular o **centroide** de cada zona.
4. Feature de distância = **haversine entre centroides** de origem e destino.

Isso entra explicitamente na fase 1 (`data_download.py` baixa o shapefile
também; `features.py` implementa o haversine entre centroides).

## Decisões de arquitetura (importantes, não mudar sem pensar de novo)

### 1. Sem notebooks. Python puro com outputs.
A fase de EDA e treino deve ser feita em scripts `.py` normais, que geram
outputs materializados: imagens (`.png`) salvas em disco, relatórios em
markdown gerados programaticamente, modelos serializados (`.pkl`/`.joblib`)
com metadata em `.json`. Nada de células de notebook. Cada script deve ser
re-executável do zero e determinístico (seed fixa).

### 2. Backend de inferência e monitoramento vivem no mesmo FastAPI, em routers separados.
- `POST /predict` — inferência real, tem que ser rápida, loga cada request.
- `POST /feedback` — recebe o ground truth (duração real) quando ele "chega"
  depois, associado a uma predição já feita.
- `/monitoring/*` — endpoints que calculam e expõem métricas de drift e de
  performance (ex: `GET /monitoring/drift?month=2020-04`,
  `GET /monitoring/performance?month=2020-04`).

Por quê: separar quem *calcula* as métricas de quem *desenha* o dashboard.
Isso permite trocar o Streamlit por um front React no futuro sem tocar no
backend — o dashboard vira só um consumidor da API. Também evita que o
`/predict` fique lento por causa de cálculo pesado de monitoramento
acontecendo ao mesmo tempo (analogia: caixa de banco vs. setor de
auditoria — um responde rápido sempre, o outro pode ser lento e não pode
nunca travar o outro).

### 3. `features.py` é importado tanto pelo treino quanto pela API.
Regra de ouro: o código que gera features no treino e em produção **tem que
ser o mesmo código**, nunca uma reimplementação parecida. Isso evita
"training-serving skew" — quando pequenas diferenças de feature engineering
entre treino e produção causam erro silencioso do modelo, difícil de
debugar. Vale a pena propositalmente experimentar esse bug uma vez e depois
corrigir, pra sentir na pele.

### 4. Simulação de produção via replay, não tempo real.
Não faz sentido esperar 2 anos. Um script (`simulate_production.py`) percorre
os dados de 2020-2021 mês a mês:
1. Envia as corridas daquele mês para `/predict` (como se fossem requests
   reais chegando).
2. Depois de um "delay" simulado, envia o ground truth real (já presente no
   dataset) para `/feedback`, imitando o atraso real que existe em produção
   entre "modelo previu" e "sabemos se acertou".

**Amostragem**: mesmo com o green taxi, replay HTTP corrida-a-corrida pode
ficar lento. Usar **amostragem estratificada por dia** (ex: N corridas por
mês, com seed fixa, mantendo a proporção diária) — preserva as distribuições
que o PSI/KS enxergam e mantém o "apertar o play" rápido. O tamanho da
amostra (N) fica parametrizável no script.

Isso permite "apertar o play" e ver a timeline do drift acontecendo mês a
mês no dashboard.

### 5. Dashboard: Streamlit agora, mas desenhado pra permitir troca por React depois.
Como o Streamlit só consome a API de monitoramento (decisão #2), trocar por
um frontend React/HTML mais chique no futuro não deve exigir mudanças no
backend — só uma nova camada de visualização consumindo os mesmos
endpoints.

### 6. Métricas de drift: implementar na unha pelo menos uma vez.
Usar PSI (Population Stability Index) e KS-test (Kolmogorov-Smirnov)
implementados manualmente para pelo menos as features principais, antes de
(opcionalmente) trazer uma lib pronta como `evidently`. Entender a
matemática por trás ajuda a debugar quando o número não bate com o que os
olhos veem no gráfico — algo que acontece direto na prática real.

**Regra dos bins congelados**: os bins do PSI são definidos **uma única vez,
na distribuição de treino (2019), e congelados**. Nunca recalcular bins com
dados de produção — os bins são a régua, e régua que muda de tamanho não
mede nada. Os limites dos bins de cada feature são salvos no
`metadata.json` junto com o modelo, e a API de monitoramento lê de lá.

### 7. SQLite em modo WAL.
O banco é SQLite (suficiente pro volume e escopo local do projeto), com
`PRAGMA journal_mode=WAL` ativado no setup (`db.py`). Motivo: o replay
martela writes no `/predict` enquanto o dashboard lê via `/monitoring/*`;
sem WAL, leituras e escritas concorrentes brigam (`database is locked`).
Com WAL, leitores não bloqueiam o escritor e vice-versa.

### 8. Gatilho de retreino é uma regra explícita, não só um botão.
Medir drift sem definir quando o alarme toca é ter termômetro sem
termostato. Thresholds explícitos (convenção comum de mercado para PSI):

| PSI | Interpretação |
|-----|---------------|
| < 0.1 | Estável |
| 0.1 – 0.2 | Atenção (monitorar de perto) |
| > 0.2 | Drift significativo |

**São dois gatilhos independentes, não um** (v2, calibrado em set/2026
contra os 24 meses replayados — ver histórico no fim desta seção):

| gatilho | condição | disponibilidade |
|---------|----------|-----------------|
| **drift** | K=2 features com PSI > 0.2 por M=2 meses consecutivos | imediata — só olha inputs |
| **performance** | MAE > holdout × 1,25 por M=2 meses consecutivos | atrasada — exige feedback |

O parecer geral é o **pior dos dois**. Os dois existem separados porque
**discordam nas duas direções** na série real, e cada discordância diz uma
coisa diferente:

- jun/2020–mar/2021 — drift vermelho, performance verde: os inputs estão
  longe de 2019, mas o modelo voltou a acertar. Drift não é degradação.
- out–dez/2021 — drift verde, performance vermelha: os inputs voltaram
  para perto de 2019 e o erro **não** voltou (MAE 5,4 contra 3,99 do
  holdout, viés −2,6). Um gatilho só de drift dá alta ao paciente aqui.

A margem de +25% no MAE é calibrada, não escolhida a esmo: com +15% o
alarme fica ligado 17 dos 24 meses (o mesmo defeito que ele deveria
corrigir), e com +35% ele perde quase todo o 2021.

**Reconhecimento (o botão "disparar retreino").** O botão é mock e
continua mock: **retreinar está fora do escopo deste projeto**, que é de
monitoramento. O que ele faz de útil é gravar em `retrain_events` que um
humano viu o alarme e assumiu a ação — e é isso que **silencia** o alarme.
Sem esse estado, o dashboard não distingue "estourou" de "estourou e
ninguém atendeu", e a série real fica 17 meses em vermelho sem que isso
signifique nada.

Regras do reconhecimento:
1. Ele **não muda o status bruto**. Um vermelho reconhecido continua
   vermelho na API; o dashboard é que o desenha apagado. O sinal segue
   auditável depois de alguém dizer "eu vi".
2. Ele vale **por gatilho**, nunca sobre o parecer geral — o geral só
   voltaria a verde quando os dois voltassem, e na série real isso nunca
   acontece, então um clique em 2020 silenciaria 2021 inteiro.
3. Ele **expira sozinho** no primeiro mês em que aquele gatilho volta a
   verde. Episódio resolvido, alarme rearmado.

Endpoints: `GET /monitoring/retrain-check?month=...` (parecer de um mês),
`GET /monitoring/timeline` (a série toda) e
`POST /monitoring/retrain-trigger` (registra o reconhecimento — fica na
API, e não no dashboard, porque o dashboard nunca toca o banco).

**Histórico.** A v1 era só o gatilho de drift (K=2, M=2). Os 24 meses
replayados mostraram que ela fica 🔴 de mai/2020 a set/2021 — 17 meses
seguidos — e depois volta a 🟢 em out/2021 com o MAE ainda em 5,4. K e M
foram **mantidos**: o alarme longo não era erro de threshold, era a regra
certa sem ninguém atendendo o alarme (daí o reconhecimento), e o 🟢
indevido era cegueira a performance (daí o segundo gatilho). `trip_type`
marca PSI 0,24 em jan/2020 — mês do grupo de controle — e portanto
contamina a contagem de K; ficou na conta mesmo assim, porque com o
reconhecimento em vigor ele só muda o status em jul–set/2021, meses em que
o alarme já estaria silenciado.

## Narrativa do dashboard: alerta precoce vs. confirmação atrasada

Uma distinção conceitual que o dashboard deve tornar visível:

- **Data drift (PSI/KS)** é calculado só com os **inputs** — não precisa de
  ground truth. Disponível imediatamente. É o **sistema de alerta precoce**.
- **Concept drift / degradação (MAE/RMSE)** depende do **feedback** chegar.
  É a **confirmação atrasada**.

Plotar as duas curvas alinhadas no tempo permite ver o PSI subindo *antes*
(ou junto) da performance degradar. Essa defasagem é literalmente o
argumento de venda do monitoramento de drift, contado pelos próprios dados
do projeto.

## Design aprovado do dashboard (brainstorm de 2026-07-24)

Decisões fechadas com mockups (3 direções foram propostas; escolhida a
combinação "sismógrafo + certificado"):

- **Dark mode**. Paleta: fundo `#131417`, painéis `#1B1D21`, texto
  `#E6E6E2`, atenção `#D9A013`, alerta `#C4442C`, agulha `#4C8DFF`.
  Sans neutra no chrome; monoespaçada nas tabelas das seções.
  Tema via `.streamlit/config.toml` + CSS injetado + tema Altair custom.
- **Layout** (uma página, top→bottom):
  1. Cabeçalho: título, versão do modelo (`/health`), slider de mês
     ("agulha") que dirige a página inteira.
  2. **Sismógrafo**: 8 pistas finas de PSI (small multiples Altair, eixo x
     compartilhado 2020-01→2021-12), divisor "confirmação atrasada",
     pistas de MAE e viés. Banda translúcida da COVID (mar–jun/2020)
     atravessando todas as pistas; linha tracejada do holdout no MAE;
     agulha no mês selecionado; severidade codificada na cor da
     linha/marcadores (estável cinza, atenção âmbar, alerta vermelho).
     A defasagem PSI→MAE (alerta precoce vs. confirmação atrasada) é o
     argumento visual central.
  3. **Três painéis estilo formulário** (borda fina, cabeçalho caixa alta,
     dados em mono) com o mês selecionado:
     Seção 1 — Exame de distribuição (tabela feature | PSI | leitura + KS);
     Seção 2 — Desempenho aferido (MAE/RMSE/viés vs. holdout, n, cobertura
     de feedback);
     Seção 3 — Parecer de retreino (os dois gatilhos lado a lado, cada um
     com seu status verde/amarelo/vermelho e a tarja de reconhecimento
     quando silenciado; features em alerta, streak, MAE vs. limiar +
     botão "disparar retreino" mock, que registra o reconhecimento).
- **Dados**: o endpoint `GET /monitoring/timeline`
  agrega drift + performance + retrain-status de todos os meses
  disponíveis numa resposta; o Streamlit chama uma vez (`st.cache_data`)
  e slider/seções são filtragem local. Dashboard continua consumindo SÓ a
  API (decisão #2). API fora do ar → mensagem clara com o comando de subir.

## Estrutura de pastas

```
taxi-mlops/
├── CLAUDE.md
├── data/
│   ├── raw/                     # parquet do green taxi + shapefile de zonas, por mês
│   └── processed/                # dados limpos e com features, por mês
├── src/
│   ├── data_download.py          # baixa os parquets da NYC TLC + shapefile de taxi zones
│   ├── features.py               # engenharia de features (compartilhado treino/API)
│   ├── eda.py                    # gera análise exploratória: imagens + relatório .md
│   ├── train.py                  # treina o modelo, salva .pkl + metadata.json
│   ├── evaluate.py                # validação do modelo final, gera relatório .md
│   └── drift.py                  # implementação de PSI, KS-test, etc.
├── reports/
│   ├── eda/                      # imagens + eda_report.md gerados por eda.py
│   └── evaluation/               # métricas de validação, gerado por evaluate.py
├── models/
│   ├── model.pkl                 # modelo serializado
│   └── metadata.json             # versão, data de treino, métricas, schema de features,
│                                 #   e limites dos bins de PSI por feature (congelados)
├── api/
│   ├── main.py                   # FastAPI app, registra os routers
│   ├── routers/
│   │   ├── predict.py             # POST /predict
│   │   ├── feedback.py            # POST /feedback
│   │   └── monitoring.py          # /monitoring/* (drift, performance, timeline,
│   │                              #   retrain-check, retrain-trigger)
│   ├── db.py                      # setup do SQLite (WAL), log de predições e feedback
│   └── schemas.py                 # Pydantic models
├── dashboard/                     # Streamlit, consome só a API (nunca acessa DB direto)
│   ├── app.py                     # a página: layout e ligação das peças
│   ├── config.py                  # ambiente (URL da API)
│   ├── theme.py                   # paleta, tema do Altair, CSS
│   ├── monitoring.py              # cliente da API — única porta de entrada de dados
│   ├── seismograph.py             # o gráfico de pistas
│   └── panels.py                  # os três painéis do mês selecionado
├── .streamlit/config.toml         # tema do Streamlit (decisão de design, ver acima)
├── Makefile                       # comandos do projeto na ordem das fases
├── scripts/
│   ├── simulate_production.py    # replay amostrado de 2020-2021 mês a mês contra a API
│   └── sanity_check_report.py    # grupo de controle (jan-fev/2020) como relatório
└── tests/
    └── ...                        # testes básicos de features.py e drift.py
```

## Fases de implementação

### Fase 1 — ML tradicional (Python puro, sem notebook)
1. `data_download.py`: baixa parquets do **green taxi** de jan/2019 a
   dez/2021 **+ o shapefile de taxi zones** da TLC.
2. `eda.py`: gera análise exploratória de 2019 — distribuições de features,
   correlações, outliers. Salva imagens em `reports/eda/` e um
   `eda_report.md` gerado programaticamente (não escrito à mão).
3. `features.py`: engenharia de features **usando só informação disponível
   no momento do embarque**: haversine entre centroides das zonas de
   origem/destino, hora do dia, dia da semana, feriado, etc.
   **Nunca usar `trip_distance` do taxímetro** (leakage — ver seção de
   armadilhas).
4. `train.py`: treina o modelo (LightGBM ou similar) só com dados de 2019,
   salva `model.pkl` + `metadata.json` (versão, data, hiperparâmetros,
   schema de features esperado, métricas de validação, **e os limites dos
   bins de PSI calculados na distribuição de treino**).
5. `evaluate.py`: validação final, gera relatório de métricas
   (MAE, RMSE, etc.) em `reports/evaluation/`.

### Fase 2 — Backend FastAPI de inferência
1. `POST /predict`: recebe features de uma corrida, retorna duração
   prevista, loga request + predição no SQLite (WAL) com timestamp e um
   `prediction_id`.
2. `POST /feedback`: recebe `prediction_id` + duração real, associa ao log
   existente.
3. Testar manualmente com `curl`/Postman antes de seguir pra fase 3.

### Fase 3 — Monitoramento e MLOps (núcleo do projeto)
1. `drift.py`: implementar PSI (com bins congelados vindos do
   `metadata.json`) e KS-test.
2. `/monitoring/drift`: compara distribuição de features de um mês de
   produção contra a distribuição de treino (2019).
3. `/monitoring/performance`: calcula MAE/RMSE do mês, usando os pares
   predição/feedback já coletados.
4. `/monitoring/retrain-check`: aplica a regra de decisão de retreino
   (K features com PSI > 0.2 por M meses) e retorna o status.
5. `simulate_production.py`: replay **amostrado** (estratificado por dia,
   seed fixa) de 2020-2021 mês a mês contra a API (predict + feedback com
   delay simulado).
6. **Sanity check obrigatório**: rodar jan e fev de 2020 primeiro e
   verificar que o PSI está baixo (< 0.1) antes de interpretar qualquer
   número da pandemia. PSI alto no grupo de controle = bug na
   implementação, não drift.
7. `dashboard/app.py` (Streamlit): painel de diagnóstico com:
   - Seletor de mês (timeline 2020-2021).
   - Gráficos de drift por feature (PSI/KS ao longo do tempo).
   - Gráfico de performance do modelo ao longo do tempo (MAE/RMSE),
     **alinhado no tempo com as curvas de drift** (alerta precoce vs.
     confirmação atrasada).
   - Destaque visual do período de queda abrupta (pandemia).
   - Status do gatilho de retreino (verde/amarelo/vermelho, vindo de
     `/monitoring/retrain-check`) + botão de "disparar retreino" (pode
     começar como um mock/placeholder que só loga a intenção).

## Decisões em aberto (revisar durante a implementação)

- [x] Escolher biblioteca do modelo → **LightGBM** (API nativa, categóricas
      sobre colunas int, NaN tratado nativamente — ver docstring de train.py).
- [x] Tamanho da amostra mensal do replay → **30k corridas/mês** (default
      de simulate_production.py, parametrizável via --sample-size). Com
      esse n o grupo de controle ficou estável (PSI < 0.04 em todas as
      features exceto a deriva secular real de trip_type), e meses de
      pandemia menores que 30k vão inteiros.
- [x] Calibrar K e M da regra de retreino → **K=2 e M=2 mantidos**; o que
      entrou foi um segundo gatilho (performance) e o estado de
      reconhecimento. Ver decisão #8.
- [x] Definir se o "retreino automático" da fase 3 vai de fato re-treinar
      o modelo ou só simular o disparo → **só simular**. Retreinar está
      fora do escopo: este projeto é de monitoramento. O mock virou algo
      útil ao gravar o reconhecimento, que silencia o alarme.
- [ ] Definir se `evidently` entra como comparação com a implementação manual
      de PSI/KS, ou fica de fora do escopo v1.
- [ ] Definir se o front React (mencionado como ambição futura) entra no
      escopo deste projeto ou fica para uma v2.
- [ ] Zonas 264 (Unknown) e 265 (Outside NYC) não têm centroide no shapefile
      → haversine NaN (~0,4% das corridas). Pesquisar depois se dá para
      atribuir centroides a elas (outras fontes ou heurística) em vez de
      aceitar o NaN.
- [x] Corridas despachadas vs. street-hail → **resolvido em evaluate.py**:
      modelo único com trip_type (NaN → categoria 0) fica; viés extra só na
      despachada explícita (1,4% do tráfego), que vira item de monitoramento.
      Racional original:
      Corridas despachadas vs. street-hail (o bloco MNAR do dicionário de
      dados; ver seção 3 da EDA — madrugada com ~25% de despachadas e
      medianas ~30 min). `trip_type` é conhecido no embarque (sem leakage),
      então o caminho natural é:
      1. incluir `trip_type` como feature candidata no `train.py` — árvores
         aprendem relações diferentes por grupo via split, sem precisar de
         dois modelos;
      2. no `evaluate.py`, quebrar os resíduos por grupo — esse é o critério
         real de decisão, não as distribuições marginais;
      3. só considerar dois modelos separados se o modelo único *com* a
         feature ainda errar sistematicamente num dos grupos.
      Uma EDA comparativa entre os grupos entra como contexto, mas com
      tamanhos de efeito, não p-valores (com n na casa das centenas de
      milhares, qualquer teste rejeita H₀ para diferenças minúsculas).
      Bônus de monitoramento: a proporção despachada/street-hail é um input
      que o PSI pode vigiar — mudança na mistura durante a pandemia é mais
      um sinal de drift de graça no dashboard.

## Fora de escopo (por enquanto)

- Orquestração com Airflow/Prefect/Dagster — o replay manual via script
  resolve para fins de aprendizado.
- Deploy em nuvem / containers em produção real — tudo roda local
  (inclusive o banco: SQLite, não Postgres — ver decisão #7).
- CI/CD.
