# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.1.0/); versiones [SemVer](https://semver.org/lang/es/).

## [Sin publicar]

### Anadido
- **Bucle de etiquetas cerrado**: cada prediccion devuelve un `prediction_id`; `POST /labels` (`X-Admin-Token`, lotes de 1-1000 `{prediction_id, churn, observed_at?}`) guarda el churn real junto a la prediccion (copia de features, probabilidad y version en una tabla `labels` de `drift.sqlite` que sobrevive al desalojo de la ventana). Idempotente por `prediction_id` (la ultima etiqueta vale; los ids desconocidos vuelven en `unknown`). Metricas `churn_labels_total{result}` y `churn_labels_stored`.
- `GET /monitoring/performance`: AUC, accuracy, precision, recall, F1 y Brier sobre las predicciones etiquetadas recientes (`CHURN_DRIFT_BUFFER_SIZE`, minimo `CHURN_DRIFT_MIN_ROWS`), en conjunto y por version del modelo, con el AUC de holdout de la version en servicio para comparar; gauges `churn_performance_*{model_version}`. El entrenamiento calcula sus metricas de holdout con la misma funcion y umbral (`churn.monitoring.performance`), y el metadata gana `precision` y `recall`.
- `python -m churn.data.labels` (`make labels-dataset` en local, `make retrain` sin `DATA=` en Docker): construye el dataset de reentreno (`.csv`/`.parquet` con las columnas del contrato, en orden de prediccion) desde las etiquetas del almacen, con un *sidecar* `<fichero>.meta.json` (filas, tasa de churn, ventana de prediccion y de observacion, versiones, huella SHA-256). Si la huella coincide al cargar el fichero con `--data`, el metadata del modelo registra `data_source.kind = "labels"` y esa ventana (`/model/info`). Codigos 0/2/3 y resumen JSON por stdout.
- `CHURN_PREDICTION_KEEP_ROWS` (100 000): retencion del almacen de predicciones, ahora independiente de la ventana de drift, para que las predicciones sigan siendo etiquetables semanas despues. `CHURN_LABELS_MIN_ROWS` (500): minimo de etiquetas para reentrenar. `make performance`.
- Tests: `test_labels.py`, `test_performance.py`, `test_label_loop.py` (bucle completo sin dobles: predicciones -> etiquetas -> dataset -> `--data` -> reload, con el modelo nuevo acertando mas y el drift desapareciendo), etiquetas en el almacen (incluida la migracion de un `drift.sqlite` anterior) y en la API, y `retrain-if-drift.sh` construyendo el dataset.
- **Reentrenamiento real**: el trainer acepta `--data <fichero.csv|.parquet>` con filas etiquetadas (`churn.data.sources`), ademas del generador sintetico; el metadata guarda `data_source` (origen, filas y huella SHA-256 del dataset) y la referencia de drift sale de los datos con los que se entreno esa version.
- **Promocion campeon/retador** (`CHURN_PROMOTION_MARGIN`, `--promotion-margin`): el modelo en servicio se evalua sobre el holdout de los datos nuevos y la version nueva solo pasa a `current` (y recibe el alias `champion` en MLflow) si su AUC es al menos igual menos el margen. Si pierde, se guarda sin promover (`promotion.decision = rejected`, codigo de salida 3), `/model/versions` lo muestra y `/model/info` expone `promotion` y `data_source`. Reentrenar con los mismos datos y semilla que el campeon se detecta (`identical_training_data`) y se avisa.
- Controles de drift deterministas en el generador (`DriftSpec`, `--drift-shift` en el trainer y en `scripts/generate_data.py`): desplazan features y relacion con la etiqueta para producir un mundo cambiado, reproducible, en tests y demos. Sin especificacion el dataset es bit a bit el de siempre.
- `make train DATA=...`, `make retrain DATA=...` (monta el fichero en el trainer) y `make drift-data`.
- Tests: `test_retraining.py` (reentrenar con el mismo generador no cambia el modelo ni el drift; con datos del mundo nuevo el modelo cambia, gana y el drift desaparece; retador rechazado, margen, campeon ilegible o incompatible, API tras reload), `test_sources.py`, drift del generador y `retrain-if-drift.sh` con dobles de `curl`/`docker`.

### Cambiado
- `deploy/scripts/retrain-if-drift.sh` reentrena solo con datos etiquetados reales: construye `/models/datasets/labels.parquet` con las etiquetas recibidas por la API y entrena con `--data` sobre el volumen, o monta `CHURN_TRAIN_DATA` (ruta a un `.csv`/`.parquet` etiquetado propio) si esta definida; recarga la API solo si el modelo nuevo se promueve (codigo 3 del trainer = no promovido, sin reload). Con menos de `CHURN_LABELS_MIN_ROWS` etiquetas, o si el fichero no existe, avisa y termina con codigo 3 en lugar de reentrenar en silencio con el generador sintetico; si el dataset de etiquetas no se puede construir, codigo 2.
- `make retrain` sin `DATA=` reentrena con las etiquetas de la API en lugar de con el generador sintetico (que producia el mismo modelo).
- El almacen SQLite anade la columna `prediction_id` y la tabla `labels` a los ficheros existentes al arrancar; las predicciones anteriores no son etiquetables y se desalojan con la ventana.
- El alias `champion` de MLflow solo se asigna a las versiones promovidas; los runs de las rechazadas se registran con `promotion_decision=rejected` y sin alias.

### Corregido
- El reentrenamiento programado no podia corregir el drift: `train.py` siempre entrenaba con `generate_dataset(seed=42)`, asi que cada ejecucion del cron producia el mismo dataset, el mismo modelo y la misma referencia, y el informe de drift no cambiaba. El ciclo drift -> reentreno -> recuperacion que describia el README no ocurria. Ahora requiere datos nuevos etiquetados y un test deja constancia del comportamiento anterior.

### Seguridad
- `deploy.yml`: el tag del despliegue manual y los secretos llegan a los scripts por `env`, y el tag se valida (`latest`, `X.Y.Z[-pre]` o `sha-<hash>`) antes de viajar en el comando remoto de ssh. Antes se interpolaban con `${{ }}` dentro de `run:`, lo que permitia inyectar comandos en el runner y en el VPS a quien pudiera lanzar el workflow.

### Corregido
- `requirements.txt` vuelve a `pandas>=2.2,<3.0` y `mlflow>=2.22,<3.0`, las familias que fijan los lockfiles y el servidor MLflow de los compose. Dependabot los habia subido a 3.x sin regenerar los locks, y como la CI instala desde el lock, su verde no validaba los rangos declarados.
- Dependabot ignora las versiones mayores de las dependencias Python; se actualizan a mano con `make lock`.

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
