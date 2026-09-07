# Comandos do projeto, na ordem das fases do CLAUDE.md.
# `make` sozinho lista os alvos.
#
# Tudo escuta só em loopback (API_HOST): o projeto roda local, ninguém da
# rede precisa alcançar a API nem o dashboard.

PY        := uv run python
API_HOST  ?= 127.0.0.1
API_PORT  ?= 8000
API_URL   := http://$(API_HOST):$(API_PORT)

.DEFAULT_GOAL := help
.PHONY: help install data centroids describe eda train evaluate api api-reload \
        dashboard control sanity replay seismograph test clean

help:  ## lista os alvos
	@grep -E '^[a-z][a-z-]*:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## instala as dependências (uv sync)
	uv sync

# --- fase 1: ML tradicional -------------------------------------------------

data:  ## baixa os parquets do green taxi (2019-2021) + shapefile de zonas
	$(PY) src/data_download.py

centroids:  ## calcula os centroides das taxi zones (data/processed/)
	$(PY) src/features.py

describe:  ## gera o dicionário de dados (reports/data_dictionary.md)
	$(PY) src/data_description.py

eda:  ## análise exploratória de 2019 (reports/eda/)
	$(PY) src/eda.py

train:  ## treina o LightGBM em 2019 (models/model.pkl + metadata.json)
	$(PY) src/train.py

evaluate:  ## valida no holdout de nov-dez/2019 (reports/evaluation/)
	$(PY) src/evaluate.py

# --- fase 2: API ------------------------------------------------------------

api:  ## sobe a API (predict + feedback + monitoring)
	uv run uvicorn api.main:app --host $(API_HOST) --port $(API_PORT)

api-reload:  ## sobe a API com --reload (desenvolvimento)
	uv run uvicorn api.main:app --host $(API_HOST) --port $(API_PORT) --reload

# --- fase 3: monitoramento --------------------------------------------------
# O replay precisa da API já no ar noutro terminal (make api).

control:  ## replay do grupo de controle (jan-fev/2020) — rodar ANTES do resto
	$(PY) scripts/simulate_production.py --months 2020-01 2020-02 \
	  --api-url $(API_URL)

sanity:  ## gera o relatório do controle (reports/monitoring/)
	$(PY) scripts/sanity_check_report.py --api-url $(API_URL)

replay:  ## replay da pandemia em diante (mar/2020 a dez/2021)
	$(PY) scripts/simulate_production.py --start 2020-03 --end 2021-12 \
	  --api-url $(API_URL)

seismograph:  ## renderiza o sismógrafo em PNG (reports/monitoring/)
	uv run --with vl-convert-python python scripts/render_seismograph.py \
	  --api-url $(API_URL)

dashboard:  ## sobe o dashboard Streamlit (só localhost)
	TAXI_DASH_API_URL=$(API_URL) uv run streamlit run dashboard/app.py

# --- utilidades -------------------------------------------------------------

test:  ## roda os testes
	uv run pytest

clean:  ## apaga só caches (não toca em data/, models/ nem reports/)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
