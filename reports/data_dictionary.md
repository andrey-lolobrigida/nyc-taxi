# Dicionário de dados — NYC Green Taxi

Gerado por `src/data_description.py` a partir de `green_tripdata_2019-01` a `green_tripdata_2019-12` (6,300,985 corridas, 20 colunas).

Interpretações conforme o data dictionary oficial da TLC ([fonte](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)).

**Disponível no embarque?** é a coluna mais importante da tabela: só variáveis com ✅ podem virar features do modelo. As demais só existem depois que a corrida termina — usá-las é vazamento de dados (ver CLAUDE.md).

## Visão geral

| coluna | dtype | NaN | % NaN | únicos | no embarque? |
|---|---|---|---|---|---|
| `VendorID` | int64 | 0 | 0.00% | 3 | ✅ |
| `lpep_pickup_datetime` | datetime64[us] | 0 | 0.00% | 5,228,457 | ✅ |
| `lpep_dropoff_datetime` | datetime64[us] | 0 | 0.00% | 5,222,847 | ❌ |
| `store_and_fwd_flag` | str | 673,138 | 10.68% | 2 | ❌ |
| `RatecodeID` | float64 | 673,138 | 10.68% | 7 | ❌ |
| `PULocationID` | int64 | 0 | 0.00% | 262 | ✅ |
| `DOLocationID` | int64 | 0 | 0.00% | 261 | ✅ |
| `passenger_count` | float64 | 673,138 | 10.68% | 10 | ✅ |
| `trip_distance` | float64 | 0 | 0.00% | 6,796 | ❌ |
| `fare_amount` | float64 | 0 | 0.00% | 9,193 | ❌ |
| `extra` | float64 | 0 | 0.00% | 31 | ❌ |
| `mta_tax` | float64 | 0 | 0.00% | 22 | ❌ |
| `tip_amount` | float64 | 0 | 0.00% | 2,886 | ❌ |
| `tolls_amount` | float64 | 0 | 0.00% | 531 | ❌ |
| `ehail_fee` | object | 6,298,768 | 99.96% | 2 | ❌ |
| `improvement_surcharge` | float64 | 2 | 0.00% | 11 | ❌ |
| `total_amount` | float64 | 0 | 0.00% | 12,480 | ❌ |
| `payment_type` | float64 | 673,138 | 10.68% | 5 | ❌ |
| `trip_type` | float64 | 675,360 | 10.72% | 2 | ✅ |
| `congestion_surcharge` | float64 | 1,219,139 | 19.35% | 9 | ❌ |

## Detalhes por coluna

### `VendorID`

Código do fornecedor do sistema LPEP que gerou o registro. 1 = Creative Mobile Technologies; 2 = VeriFone Inc. Metadado do equipamento, não da corrida — pouco útil como feature.

dtype `int64` · 0 NaN (0.00%) · 3 valores únicos · min = 1 · max = 5 · mean = 1.86

| valor | corridas | % |
|---|---|---|
| 1 | 894,041 | 14.19% |
| 2 | 5,406,935 | 85.81% |
| 5 | 9 | 0.00% |

### `lpep_pickup_datetime`

Data/hora em que o taxímetro foi acionado (início da corrida). Base das features temporais: hora do dia, dia da semana, feriado. Junto com o dropoff, define o alvo `trip_duration`.

dtype `datetime64[us]` · 0 NaN (0.00%) · 5,228,457 valores únicos · min = 2008-10-21 15:52:05 · max = 2062-08-15 00:00:00

### `lpep_dropoff_datetime`

Data/hora em que o taxímetro foi desligado (fim da corrida). Só existe quando a corrida termina — usar como feature é leakage. Uso legítimo: derivar o alvo `trip_duration` no treino.

dtype `datetime64[us]` · 0 NaN (0.00%) · 5,222,847 valores únicos · min = 2008-10-21 15:54:26 · max = 2062-08-15 16:34:10

### `store_and_fwd_flag`

Y/N — indica se o registro ficou retido na memória do veículo ('store and forward') por falta de conexão com o servidor antes do envio. Detalhe de telemetria, irrelevante para o modelo.

dtype `str` · 673,138 NaN (10.68%) · 2 valores únicos

| valor | corridas | % |
|---|---|---|
| N | 5,613,388 | 89.09% |
| Y | 14,459 | 0.23% |
| NaN | 673,138 | 10.68% |

### `RatecodeID`

Código de tarifa **final** da corrida: 1 = Standard; 2 = JFK; 3 = Newark; 4 = Nassau/Westchester; 5 = tarifa negociada; 6 = corrida em grupo. Por ser o código final (pode mudar durante a corrida), tratamos como pós-corrida por segurança.

dtype `float64` · 673,138 NaN (10.68%) · 7 valores únicos · min = 1.00 · max = 99.00 · mean = 1.19

| valor | corridas | % |
|---|---|---|
| 1.0 | 5,355,386 | 84.99% |
| 2.0 | 11,108 | 0.18% |
| 3.0 | 2,627 | 0.04% |
| 4.0 | 4,164 | 0.07% |
| 5.0 | 254,455 | 4.04% |
| 6.0 | 57 | 0.00% |
| 99.0 | 50 | 0.00% |
| NaN | 673,138 | 10.68% |

### `PULocationID`

Taxi zone de embarque (1–263, mapeável pelo shapefile/lookup). Conhecida no embarque. Origem do centroide usado na feature de distância haversine.

dtype `int64` · 0 NaN (0.00%) · 262 valores únicos · min = 1 · max = 265 · mean = 109.00

### `DOLocationID`

Taxi zone de desembarque informada como destino. No green taxi é registrada pelo sistema ao fim da corrida, mas em produção real o destino declarado existe no embarque — tratamos como disponível. Destino do centroide da feature haversine.

dtype `int64` · 0 NaN (0.00%) · 261 valores únicos · min = 1 · max = 265 · mean = 129.20

### `passenger_count`

Número de passageiros, digitado manualmente pelo motorista. Conhecido no embarque, mas ruidoso (erros de digitação, zeros).

dtype `float64` · 673,138 NaN (10.68%) · 10 valores únicos · min = 0.00 · max = 9.00 · mean = 1.31

| valor | corridas | % |
|---|---|---|
| 0.0 | 11,689 | 0.19% |
| 1.0 | 4,840,305 | 76.82% |
| 2.0 | 416,864 | 6.62% |
| 3.0 | 77,078 | 1.22% |
| 4.0 | 28,859 | 0.46% |
| 5.0 | 164,320 | 2.61% |
| 6.0 | 88,486 | 1.40% |
| 7.0 | 87 | 0.00% |
| 8.0 | 124 | 0.00% |
| 9.0 | 35 | 0.00% |
| NaN | 673,138 | 10.68% |

### `trip_distance`

⚠️ **Distância efetivamente percorrida** (milhas) registrada pelo taxímetro — só existe ao fim da corrida. É a armadilha de leakage central do projeto: **nunca usar como feature**. Substituto: haversine entre centroides das zonas (features.py).

dtype `float64` · 0 NaN (0.00%) · 6,796 valores únicos · min = -20,329.08 · max = 77,843.76 · mean = 3.53

### `fare_amount`

Tarifa de tempo-e-distância calculada pelo taxímetro. Resultado da corrida — leakage.

dtype `float64` · 0 NaN (0.00%) · 9,193 valores únicos · min = -890.00 · max = 4,011.50 · mean = 15.44

### `extra`

Extras e sobretaxas diversas (inclui $0.50 e $1 de rush hour e noturno). Componente do valor final — leakage.

dtype `float64` · 0 NaN (0.00%) · 31 valores únicos · min = -4.50 · max = 13.75 · mean = 0.62

### `mta_tax`

Taxa MTA de $0.50 acionada pela tarifa em uso. Leakage.

dtype `float64` · 0 NaN (0.00%) · 22 valores únicos · min = -0.50 · max = 17.33 · mean = 0.48

### `tip_amount`

Gorjeta — só registra gorjetas em cartão; gorjetas em dinheiro não entram. Pós-corrida, leakage.

dtype `float64` · 0 NaN (0.00%) · 2,886 valores únicos · min = -90.50 · max = 441.00 · mean = 0.94

### `tolls_amount`

Total de pedágios pagos na corrida. Pós-corrida, leakage.

dtype `float64` · 0 NaN (0.00%) · 531 valores únicos · min = -21.00 · max = 935.50 · mean = 0.29

### `ehail_fee`

Taxa de e-hail (chamada por app). Praticamente 100% nula no período — candidata a descarte imediato.

dtype `object` · 6,298,768 NaN (99.96%) · 2 valores únicos

| valor | corridas | % |
|---|---|---|
| 0.0 | 2,189 | 0.03% |
| 1.95 | 28 | 0.00% |
| NaN | 4,468,525 | 70.92% |
| NaN | 1,830,243 | 29.05% |

### `improvement_surcharge`

Sobretaxa de melhoria de $0.30 cobrada na bandeirada (desde 2015). Componente do valor final — leakage.

dtype `float64` · 2 NaN (0.00%) · 11 valores únicos · min = -0.30 · max = 0.47 · mean = 0.26

| valor | corridas | % |
|---|---|---|
| -0.3 | 17,791 | 0.28% |
| 0.0 | 839,091 | 13.32% |
| 0.12 | 1 | 0.00% |
| 0.13 | 1 | 0.00% |
| 0.17 | 1 | 0.00% |
| 0.3 | 5,444,091 | 86.40% |
| 0.32 | 1 | 0.00% |
| 0.43 | 3 | 0.00% |
| 0.44 | 1 | 0.00% |
| 0.46 | 1 | 0.00% |
| 0.47 | 1 | 0.00% |
| NaN | 2 | 0.00% |

### `total_amount`

Valor total cobrado (não inclui gorjetas em dinheiro). Resultado da corrida — leakage.

dtype `float64` · 0 NaN (0.00%) · 12,480 valores únicos · min = -890.30 · max = 4,012.30 · mean = 18.33

### `payment_type`

Forma de pagamento: 1 = cartão de crédito; 2 = dinheiro; 3 = sem cobrança; 4 = disputa; 5 = desconhecido; 6 = anulada. Só se sabe ao pagar — pós-corrida.

dtype `float64` · 673,138 NaN (10.68%) · 5 valores únicos · min = 1.00 · max = 5.00 · mean = 1.44

| valor | corridas | % |
|---|---|---|
| 1.0 | 3,207,070 | 50.90% |
| 2.0 | 2,378,568 | 37.75% |
| 3.0 | 29,513 | 0.47% |
| 4.0 | 12,498 | 0.20% |
| 5.0 | 198 | 0.00% |
| NaN | 673,138 | 10.68% |

### `trip_type`

1 = street-hail (parada na rua); 2 = dispatch (despachada). Definido no início da corrida; atribuído automaticamente, mas editável pelo motorista.

dtype `float64` · 675,360 NaN (10.72%) · 2 valores únicos · min = 1.00 · max = 2.00 · mean = 1.04

| valor | corridas | % |
|---|---|---|
| 1.0 | 5,375,316 | 85.31% |
| 2.0 | 250,309 | 3.97% |
| NaN | 675,360 | 10.72% |

### `congestion_surcharge`

Sobretaxa de congestionamento (introduzida em 2019) aplicada conforme a corrida. Componente do valor final — leakage.

dtype `float64` · 1,219,139 NaN (19.35%) · 9 valores únicos · min = -2.75 · max = 2.75 · mean = 0.40

| valor | corridas | % |
|---|---|---|
| -2.75 | 72 | 0.00% |
| -2.5 | 1 | 0.00% |
| -0.75 | 1 | 0.00% |
| 0.0 | 4,346,769 | 68.99% |
| 0.3 | 1 | 0.00% |
| 0.75 | 73 | 0.00% |
| 2.0 | 2 | 0.00% |
| 2.5 | 1,692 | 0.03% |
| 2.75 | 733,235 | 11.64% |
| NaN | 1,219,139 | 19.35% |

## Notas de qualidade (investigação sobre 2019)

**O bloco de NaNs idênticos é uma subpopulação, não ruído.** As mesmas
673.138 linhas (10,7% de 2019) têm NaN simultâneo em `store_and_fwd_flag`,
`RatecodeID`, `passenger_count`, `payment_type` e `trip_type`. O perfil delas
é distinto do resto:

| característica | bloco NaN | demais corridas |
|---|---|---|
| VendorID | 100% VeriFone (2) | mistura de 1 e 2 |
| distância mediana | 6,2 mi | 1,8 mi |
| tarifa mediana | $29,45 | $9,50 |
| duração mediana | 28 min | 11 min |
| pico de horário | 9h–13h | 15h–19h |
| congestion_surcharge | 100% NaN | ~10% NaN |
| fração do ano | cresce de 6% (jan) para 21% (dez) | — |

O padrão (sem dados de taxímetro/pagamento, corridas longas, diurnas, caras,
via um único vendor) indica corridas **despachadas/pré-arranjadas ingeridas
por um canal de reporte reduzido**, e não falha aleatória de coleta — é
missingness informativo (MNAR).

Implicações para o projeto:
- **Não descartar essas linhas às cegas**: `lpep_pickup_datetime`,
  `PULocationID` e `DOLocationID` (nossas features centrais) estão íntegros
  nelas, e a fração cresce ao longo do tempo — remover mudaria a distribuição
  de forma correlacionada com o tempo, contaminando o baseline do PSI.
- `passenger_count` NaN nessas linhas: tratar na engenharia de features
  (imputação ou categoria própria), ou descartar a feature.
- Outros achados menores: 273 corridas com pickup fora de 2019 (2008–2062,
  todas fora do bloco) → filtro de sanidade temporal na limpeza;
  `RatecodeID = 99` em 50 linhas (código não documentado); `VendorID = 5`
  em 9 linhas; ~0,3% de valores monetários negativos (estornos).
