.PHONY: install lock lint format typecheck test coverage check audit mutation train run up down logs retrain drift drift-data labels-dataset performance

# Datos etiquetados para (re)entrenar: `make train DATA=data/churn-q3.csv` o
# `make retrain DATA=...`. Sin DATA, `train` usa el generador sintetico (primer
# entrenamiento o demos con DRIFT_SHIFT) y `retrain` las etiquetas recibidas por la API.
DATA ?=
DRIFT_SHIFT ?= 0
# Salida de `make labels-dataset` (dataset + sidecar <fichero>.meta.json con su huella)
LABELS_OUT ?= data/labels.parquet

install:            ## Dependencias de desarrollo (versiones fijadas en el lockfile)
	pip install -r requirements-dev.lock

lock:               ## Regenera los lockfiles a partir de requirements*.txt (necesita uv)
	uv pip compile requirements.txt -o requirements.lock --universal --python-version 3.11
	uv pip compile requirements-dev.txt -o requirements-dev.lock --universal --python-version 3.11

lint:               ## ruff: reglas + formato
	ruff check .
	ruff format --check .

format:             ## Aplica el formato de ruff
	ruff format .

typecheck:
	mypy src/churn

test:
	pytest -v

coverage:           ## Tests con informe de cobertura (mismo umbral que CI)
	pytest --cov=churn --cov-report=term-missing --cov-fail-under=90

check: lint typecheck coverage   ## Todo lo que ejecuta el CI (salvo Docker)

audit:              ## Vulnerabilidades conocidas en las dependencias de produccion
	pip-audit -r requirements.lock

mutation:           ## Mutation testing de los modulos puros (lento; ver [tool.mutmut])
	mutmut run

train:              ## Entrena en local (sin Docker) y deja artefactos en ./models (DATA= fichero etiquetado)
ifneq ($(DATA),)
	PYTHONPATH=src python -m churn.training.train --data "$(DATA)" --model-dir models
else
	PYTHONPATH=src python -m churn.training.train --rows 20000 --drift-shift $(DRIFT_SHIFT) --model-dir models
endif

run: train          ## API local sin Docker
	CHURN_MODEL_DIR=models PYTHONPATH=src uvicorn churn.serving.main:app --reload --port 8010 --no-access-log

up:                 ## Stack completo local (MLflow + trainer + API)
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f --tail 100

retrain:            ## Reentrena dentro de Docker (DATA= fichero etiquetado; sin DATA, las etiquetas de POST /labels) y recarga la API si se promueve
ifneq ($(DATA),)
	docker compose run --rm -v "$(abspath $(DATA)):/data/$(notdir $(DATA)):ro" trainer \
	  python -m churn.training.train --data "/data/$(notdir $(DATA))" --model-dir /models
else
	@echo ">> Sin DATA= se reentrena con las etiquetas recibidas por la API (dataset dentro del volumen)."
	docker compose run --rm --no-deps -T trainer python -m churn.data.labels --model-dir /models --out /models/datasets/labels.parquet
	docker compose run --rm trainer python -m churn.training.train --data /models/datasets/labels.parquet --model-dir /models
endif
	curl -fsS -X POST http://localhost:8010/model/reload -H 'X-Admin-Token: token-local'

labels-dataset:     ## Sin Docker: construye LABELS_OUT con las etiquetas recibidas por la API local (`make run`), listo para `make train DATA=`
	PYTHONPATH=src python -m churn.data.labels --model-dir models --out "$(LABELS_OUT)"

drift-data:         ## Genera data/churn-drift.csv: un "mundo desplazado" etiquetado para ensayar el reentreno
	PYTHONPATH=src python scripts/generate_data.py --rows 20000 --seed 43 --drift-shift 1.0 --out data/churn-drift.csv

drift:              ## Informe de drift del trafico reciente
	curl -fsS http://localhost:8010/monitoring/drift | python -m json.tool

performance:        ## Acierto real del modelo sobre las predicciones etiquetadas (AUC, precision, recall...)
	curl -fsS http://localhost:8010/monitoring/performance | python -m json.tool
