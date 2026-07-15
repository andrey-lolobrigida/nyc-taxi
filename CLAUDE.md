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

Lógica de decisão (v1, ajustável): o gatilho de retreino dispara se
**K ou mais features** apresentarem PSI > 0.2 por **M meses consecutivos**
(valores iniciais sugeridos: K=2, M=2 — evita alarme por outlier de um mês
só). A regra fica implementada num endpoint tipo
`GET /monitoring/retrain-check?month=...`, que o dashboard exibe como
status (verde/amarelo/vermelho) ao lado do botão de retreino.

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
│   │   └── monitoring.py          # GET /monitoring/* (drift, performance, retrain-check)
│   ├── db.py                      # setup do SQLite (WAL), log de predições e feedback
│   └── schemas.py                 # Pydantic models
├── dashboard/
│   └── app.py                     # Streamlit, consome só a API (nunca acessa DB direto)
├── scripts/
│   └── simulate_production.py    # replay amostrado de 2020-2021 mês a mês contra a API
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
- [ ] Definir tamanho da amostra mensal do replay (ponto de partida:
      30-50k corridas/mês ou o mês inteiro se for menor que isso).
- [ ] Calibrar K e M da regra de retreino depois de ver os números reais.
- [ ] Definir se o "retreino automático" da fase 3 vai de fato re-treinar
      o modelo ou só simular o disparo do pipeline.
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
