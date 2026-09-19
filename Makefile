.PHONY: install lock lint test coverage train run up down logs retrain drift

install:            ## Dependencias de desarrollo (versiones fijadas en el lockfile)
	pip install -r requirements-dev.lock

lock:               ## Regenera los lockfiles a partir de requirements*.txt (necesita uv)
	uv pip compile requirements.txt -o requirements.lock --universal --python-version 3.11
	uv pip compile requirements-dev.txt -o requirements-dev.lock --universal --python-version 3.11

lint:
	ruff check .

test:
	pytest -v

coverage:           ## Tests con informe de cobertura (mismo umbral que CI)
	pytest --cov=churn --cov-report=term-missing --cov-fail-under=90

train:              ## Entrena en local (sin Docker) y deja artefactos en ./models
	PYTHONPATH=src python -m churn.training.train --rows 20000 --model-dir models

run: train          ## API local sin Docker
	CHURN_MODEL_DIR=models PYTHONPATH=src uvicorn churn.serving.main:app --reload --port 8010

up:                 ## Stack completo local (MLflow + trainer + API)
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f --tail 100

retrain:            ## Reentrena dentro de Docker y recarga la API (falla si el gate de calidad no pasa)
	docker compose run --rm trainer
	curl -fsS -X POST http://localhost:8010/model/reload -H 'X-Admin-Token: token-local'

drift:              ## Informe de drift del trafico reciente
	curl -fsS http://localhost:8010/monitoring/drift | python -m json.tool
