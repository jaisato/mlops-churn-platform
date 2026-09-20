# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.1.0/); versiones [SemVer](https://semver.org/lang/es/).

## [1.2.0] - 2026-09-20

### Anadido
- Almacen de modelos versionado (`models/versions/<version>` + puntero atomico `current`) con retencion (`CHURN_MODEL_KEEP_VERSIONS`), `GET /model/versions` y `POST /model/rollback`; el layout plano de la 1.0 sigue cargandose.
- Predicciones recientes persistidas en SQLite dentro del volumen (`CHURN_DRIFT_STORE`, `CHURN_DRIFT_DB_PATH`), compartidas entre workers y reinicios.
- Comprobacion de compatibilidad antes de servir: versiones de Python/scikit-learn en el metadata, rechazo de features distintas y de otra version menor de scikit-learn (`CHURN_STRICT_ARTIFACT_COMPAT`).
- Alias `champion` en el Model Registry de MLflow para cada version que supera el gate (`CHURN_MLFLOW_CHAMPION_ALIAS`); `mlflow_model_version` en el metadata y en `/model/info`.
- `/health/live` (liveness), `/metrics` (Prometheus, registro por aplicacion), `X-Request-ID` y log de acceso JSON; los loggers de uvicorn usan el mismo formato.
- API key opcional (`CHURN_API_KEY`) para scoring, informacion del modelo y drift.
- Scripts `deploy/scripts/backup.sh` (copias de volumenes con retencion) y `deploy/scripts/retrain-if-drift.sh` (reentrenamiento por drift con aviso por webhook).
- Lockfiles `requirements.lock` / `requirements-dev.lock` (uv), `make lock`, `make check`, `make mutation`.
- CI: `ruff format`, `mypy`, `shellcheck`, `actionlint`, `hadolint`, matriz Python 3.11-3.13, `pip-audit`, Dependabot; huella SSH del VPS fijada en el despliegue (`DEPLOY_HOST_KEY`).
- Tests de propiedades con Hypothesis para el PSI; suite de 227 tests con cobertura del 100 %.
- `docker-compose.prod.yml` ensayable en local (`IMAGE_REGISTRY`, `CADDY_HTTP_PORT`, `CADDY_HTTPS_PORT`).

### Cambiado
- Python minimo 3.11 (3.10 llega a fin de vida en octubre de 2026); MLflow cliente y servidor alineados en la familia 2.22.
- `setup-vps.sh`: actualizaciones de seguridad automaticas y SSH solo con claves cuando ya hay claves autorizadas.
- Dockerfile: UID numerico, healthcheck en forma exec, `--no-access-log` en uvicorn.

## [1.1.0] - 2026-09-14

### Anadido
- Gate de calidad en el entrenamiento (`CHURN_MIN_ROC_AUC`, `--min-auc`): un modelo por debajo del AUC minimo no sobreescribe artefactos.
- Drift de predicciones (PSI de las probabilidades frente a la referencia) y `model_version` en el informe de drift.
- Metadata de trazabilidad: semilla, resultado del gate, `mlflow_run_id`; modelos de respuesta tipados para reload y drift.
- Guardia de token en produccion, validacion de umbrales y buffer de drift configurable.
- `deploy.sh` con espera real a `/health`, entrenamiento inicial automatico y rollback; healthcheck de MLflow en compose.

### Corregido
- PSI numerico ciego en variables binarias/enteras y referencias constantes; PSI categorico con tipos mezclados.
- Reload con artefactos corruptos (500 con traza) y arranque con artefactos corruptos; el modelo anterior se conserva.
- MLflow como punto unico de fallo del trainer; colision de `model_version` en el mismo segundo; errores duplicados en la validacion con nulos.
- Release: `latest` no se publicaba en tags, input `v1.0.0` frente a etiqueta `1.0.0`, `.env` pisaba el `TAG` del CI, rollback documentado que redesplegaba la version actual.
- Caddy publicable en todas las interfaces sin `VPN_BIND_IP`; redireccion `/mlflow -> /mlflow/`.

## [1.0.0] - 2026-07-11

Version inicial: generacion y validacion de datos, entrenamiento con MLflow opcional, API FastAPI, drift PSI/KS, compose local y de produccion, CI/CD con despliegue por VPN.
