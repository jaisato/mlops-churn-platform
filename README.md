# 📈 MLOps Churn Platform

[![CI](https://github.com/jaisato/mlops-churn-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/jaisato/mlops-churn-platform/actions/workflows/ci.yml)

Plataforma **MLOps end-to-end** para predicción de bajas de clientes (*churn*): generación/validación de datos, entrenamiento reproducible con **gate de calidad** que protege el modelo en servicio, **almacén de modelos versionado con rollback**, registro en **MLflow** con alias de campeón, serving en tiempo real con **FastAPI**, **monitorización de drift** persistente (PSI + Kolmogorov-Smirnov sobre las features y PSI sobre las predicciones) y **métricas Prometheus**.

> **Por qué este proyecto**: *ML Engineer* es el perfil IA con más ofertas activas en España y *MLOps* tiene más vacantes que candidatos (salarios senior 78-100 k€). Lo que piden esas ofertas es exactamente esto: llevar un modelo del prototipo a producción con ciclo de vida completo (entrenar → registrar → servir → monitorizar → reentrenar → volver atrás si hace falta).

---

## ✨ Qué hace

1. **Genera datos sintéticos realistas** de un negocio de suscripciones (reproducibles por semilla) — sin depender de datos privados.
2. **Valida los datos antes de entrenar** (esquema, nulos, rangos, categorías, balance de clases): *fail-fast*. Los rangos y categorías son la **fuente única de verdad** que también usa el contrato de la API.
3. **Entrena** un `HistGradientBoostingClassifier` (pipeline sklearn con preprocesado incluido), evalúa (AUC, F1, Brier) y aplica un **gate de calidad** (`CHURN_MIN_ROC_AUC`): si el modelo no lo supera, el entrenamiento termina con código 2 y **no toca** el modelo en servicio.
4. **Publica cada modelo como una versión inmutable** en `models/versions/<versión>/` (`model.joblib` + `metadata.json` + `reference.csv`) y mueve el puntero `models/current` de forma atómica. Las versiones antiguas se podan (`CHURN_MODEL_KEEP_VERSIONS`), nunca la que está en servicio.
5. **Registra** parámetros, métricas y modelo en **MLflow Model Registry** (opcional) y asigna el alias **`champion`** a cada versión que supera el gate. Si MLflow no responde, el entrenamiento avisa y termina igualmente: el tracking es trazabilidad, no un punto único de fallo. El `run_id` y la versión del registry quedan en `metadata.json` y en `/model/info`.
6. **Comprueba la compatibilidad antes de servir**: el metadata guarda las versiones de Python y scikit-learn con las que se serializó el pipeline; la API rechaza modelos con otras features o con otra versión menor de scikit-learn (los pickles no son portables) en lugar de servirlos a ciegas.
7. **Sirve** predicciones vía API REST (individual y batch) con niveles de riesgo de negocio (`bajo/medio/alto`), API key opcional, `X-Request-ID` y logs JSON.
8. **Monitoriza drift** con las predicciones guardadas en **SQLite dentro del volumen** (sobreviven a reinicios y se comparten entre workers): PSI y KS por variable contra la referencia de entrenamiento y PSI de las **predicciones** contra las puntuaciones de la referencia, expuesto también como métricas Prometheus.
9. **Reentrena, recarga y vuelve atrás sin parar el servicio**: `docker compose run trainer` + `POST /model/reload`; `POST /model/rollback` devuelve la versión anterior (o una concreta). Un reload o rollback fallido **nunca degrada el servicio**: se conserva el último modelo bueno.

## 🏗️ Arquitectura

```
        (cron / manual / CI)                          volumen compartido /models
 ┌──────────────┐  valida+entrena  ┌────────────────────┐  current ─▶ versions/<v>/model.joblib
 │   trainer     │──gate calidad──▶│  LocalModelStore   │                        metadata.json
 │ (mismo image) │──log runs──┐    │ (versionado+atómico)                        reference.csv
 └──────────────┘             ▼    └────────────────────┘  drift.sqlite (predicciones recientes)
                        ┌──────────┐        ▲ carga al arrancar / reload / rollback
                        │  MLflow  │  ┌─────┴─────┐   /predict /predict/batch  /model/info
  clientes ────────────▶│ registry │  │ API FastAPI│──▶ /model/versions /model/rollback
                        │ @champion│  └───────────┘   /monitoring/drift  /metrics  /health/live
                        └──────────┘        │ features + probabilidad por predicción
                                            ▼
                              drift PSI + KS (features) y PSI (predicciones)
```

## 🚀 Arranque local con Docker

```bash
docker compose up --build -d
```

MLflow arranca con *healthcheck*; el servicio `trainer` entrena cuando MLflow está listo (20 000 filas) y la API arranca **solo cuando el entrenamiento termina con éxito** (`service_completed_successfully`).

| Servicio | URL |
|---|---|
| API de scoring | http://localhost:8010 (docs en `/docs`, métricas en `/metrics`) |
| MLflow UI | http://localhost:5000 |

Prueba de scoring:

```bash
curl -X POST http://localhost:8010/predict -H 'Content-Type: application/json' -d '{
  "tenure_months": 3, "monthly_charges": 95.5, "total_charges": 280,
  "num_products": 1, "support_tickets_90d": 6, "is_fiber": 0,
  "contract_type": "mensual", "payment_method": "transferencia"
}'
# → {"churn_probability": 0.87, "risk_level": "alto", "model_version": "20260920-101500-3f9a1c"}
```

Reentrenar, recargar en caliente y volver atrás:

```bash
make retrain      # docker compose run trainer + POST /model/reload (falla si el gate no pasa)
curl -X POST localhost:8010/model/rollback -H 'X-Admin-Token: token-local'   # versión anterior
curl localhost:8010/model/versions                                            # qué hay publicado
```

Informe de drift del tráfico acumulado:

```bash
make drift        # → PSI y KS por feature, PSI de predicciones y lista de features con drift
```

Sin Docker: `make install && make run` (entrena y sirve en local; requiere Python 3.11+).

## ⚙️ Configuración

| Variable | Por defecto | Descripción |
|---|---|---|
| `CHURN_ENVIRONMENT` | `dev` | En `production` la API se niega a arrancar con un token de administración inseguro |
| `CHURN_MODEL_DIR` | `models` | Directorio de artefactos (volumen compartido) |
| `CHURN_MODEL_KEEP_VERSIONS` | `5` | Versiones publicadas que se conservan (la actual nunca se borra) |
| `CHURN_STRICT_ARTIFACT_COMPAT` | `true` | Rechazar modelos serializados con otra versión menor de scikit-learn |
| `CHURN_MLFLOW_TRACKING_URI` | *(vacío = desactivado)* | URL de MLflow para tracking/registry |
| `CHURN_MLFLOW_CHAMPION_ALIAS` | `champion` | Alias asignado en el registry a cada versión que supera el gate (`""` = ninguno) |
| `CHURN_MIN_ROC_AUC` | `0.75` | Gate de calidad: AUC mínimo para publicar artefactos (también `--min-auc` en el trainer) |
| `CHURN_RISK_MEDIUM` / `CHURN_RISK_HIGH` | `0.35` / `0.65` | Umbrales de riesgo de negocio (inclusivos; `medium < high` obligatorio) |
| `CHURN_DRIFT_MIN_ROWS` | `200` | Mínimo de predicciones para calcular drift |
| `CHURN_DRIFT_BUFFER_SIZE` | `5000` | Ventana rodante de predicciones recientes |
| `CHURN_DRIFT_STORE` / `CHURN_DRIFT_DB_PATH` | `sqlite` / `<model_dir>/drift.sqlite` | Dónde viven las predicciones (`memory` = por proceso, solo demos) |
| `CHURN_PSI_ALERT_THRESHOLD` | `0.2` | Umbral de alerta PSI |
| `CHURN_ADMIN_TOKEN` | — | Token de `/model/reload` y `/model/rollback` (>= 16 caracteres en producción; `openssl rand -hex 32`) |
| `CHURN_API_KEY` | *(vacío = abierto)* | Si se define, `/predict*`, `/model/info`, `/model/versions` y `/monitoring/drift` exigen `X-API-Key` |
| `CHURN_METRICS_ENABLED` / `CHURN_ACCESS_LOG` | `true` / `true` | `/metrics` Prometheus y una línea JSON por petición |

## 📡 API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health/live` | Liveness: el proceso responde (siempre 200) |
| `GET` | `/health` | Readiness: 200 con la versión del modelo cargado; 503 si no hay modelo |
| `GET` | `/metrics` | Métricas Prometheus: peticiones y latencia por ruta, predicciones por riesgo y versión, histograma de probabilidades, PSI del último informe, versión en servicio |
| `POST` | `/predict` | Probabilidad de churn + nivel de riesgo para un cliente |
| `POST` | `/predict/batch` | Lo mismo para 1-1000 clientes |
| `GET` | `/model/info` | Versión, métricas, gate, semilla, runtime, `run_id` y versión de MLflow |
| `GET` | `/model/versions` | Versiones publicadas en el almacén, cuál apunta `current` y cuál sirve este proceso |
| `POST` | `/model/reload` | Carga la versión `current` (cabecera `X-Admin-Token`); conserva la anterior si falla |
| `POST` | `/model/rollback` | Vuelve a la versión anterior o a `{"version": "..."}`; mueve `current` solo si carga bien |
| `GET` | `/monitoring/drift` | Informe de drift; 409 mientras no haya `CHURN_DRIFT_MIN_ROWS` predicciones |

Tras un reload o rollback, el drift de predicciones solo compara puntuaciones producidas por la **versión en servicio**; las features del tráfico anterior siguen contando. Cada respuesta lleva `X-Request-ID` (propagado si el cliente lo envía) y el mismo identificador aparece en la línea JSON del log de acceso.

## ✅ Tests automatizados

```bash
make install && make test       # suite completa (~25 s, sin servicios externos)
make check                      # lo mismo que el CI: ruff, mypy y cobertura (umbral 90 %)
```

Qué cubren (227 tests en 13 módulos, cobertura de líneas y ramas del 100 %):

- **Generador**: esquema, reproducibilidad, señal predictiva, dtypes y contrato con los rangos del validador.
- **Validación**: 15 casos de fallo sin duplicar errores, acumulación de errores y sincronía con el generador.
- **Entrenamiento**: artefactos versionados, gate de calidad que no sobreescribe, metadata de trazabilidad con runtime, versiones únicas, MLflow desactivado / no instalado / caído / correcto / alias fallido (con dobles, sin servidor) y CLI con códigos de salida.
- **Almacén de modelos**: publicación atómica, retención, layout plano heredado, rollback en ambas direcciones, versiones incompletas o corruptas, compatibilidad de features y runtime.
- **Almacén de predicciones**: memoria y SQLite con el mismo contrato, ventana rodante, persistencia entre instancias, escrituras concurrentes.
- **Drift**: PSI en binarias desbalanceadas, referencias constantes, baja cardinalidad, nulos, categóricas con tipos mezclados, *prediction drift*, estructura del informe y **propiedades con Hypothesis** (no negatividad, simetría, cobertura de los bins).
- **Configuración**: prefijo `CHURN_`, umbrales coherentes, guardia de token en producción.
- **API**: coherencia de riesgo, validación 422, batch consistente con individual, 503 sin modelo, drift 409→200 y detectado, reload y rollback con token, versión nueva, artefactos ausentes, corruptos o incompatibles, retención, API key, métricas, `X-Request-ID` y log de acceso.
- **Contratos**: el esquema Pydantic de la API coincide con los rangos y categorías del validador.

`make mutation` ejecuta mutation testing (mutmut) sobre los módulos puros de datos y drift para comprobar que los tests detectan cambios de lógica, no solo que ejecutan líneas.

## 🔁 CI/CD (GitHub Actions)

- **`ci.yml`** en cada push/PR: `ruff` (reglas y formato) + `mypy` + `shellcheck`/`actionlint`/`hadolint` → `pytest` en Python 3.11, 3.12 y 3.13 con cobertura mínima del 90 % → `pip-audit` sobre el lockfile de producción → validación de los ficheros compose y build de la imagen.
- **`deploy.yml`** (tag `vX.Y.Z`): build → push a **GHCR** (`X.Y.Z`, `sha-…` y `latest`) → túnel **WireGuard** → SSH al VPS con huella fijada → `deploy/scripts/deploy.sh` (pull, up, espera a `/health`, entrenamiento inicial si el volumen está vacío, rollback automático si la API no responde).
- **Dependencias fijadas** en `requirements.lock` / `requirements-dev.lock` (generados con `uv`, `make lock`) y vigiladas por Dependabot.

## 🏭 Despliegue a producción (VPS netcup + VPN)

Guía completa: [`deploy/DESPLIEGUE_NETCUP.md`](deploy/DESPLIEGUE_NETCUP.md). Resumen:

1. `bash deploy/scripts/setup-vps.sh` en el VPS (Docker, ufw, fail2ban, WireGuard, actualizaciones automáticas, SSH solo con claves).
2. Alta de peers VPN (portátil + runner CI) y login GHCR en el VPS.
3. Clonar en `/opt/mlops-churn-platform`, `cp .env.example .env` (token seguro y `VPN_BIND_IP` obligatorios), `TAG=latest ./deploy/scripts/deploy.sh`.
4. Secretos en GitHub (`WG_CONFIG`, `DEPLOY_HOST`, `DEPLOY_HOST_KEY`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`) y desplegar con `git tag v1.2.0 && git push --tags`.
5. Operación: `deploy/scripts/retrain-if-drift.sh` (cron) reentrena solo si hay drift y `deploy/scripts/backup.sh` (cron) archiva los volúmenes. Rollback de **imagen**: `TAG=$(cat .previous_tag) ./deploy/scripts/deploy.sh`; rollback de **modelo**: `POST /model/rollback`.

Todo el stack de producción, incluido `deploy.sh`, se puede ensayar en local con un registro Docker local (`IMAGE_REGISTRY`, `CADDY_*_PORT` en `.env`); la guía explica cómo.

La API y MLflow solo son accesibles **dentro de la VPN** (Caddy con TLS; MLflow bajo `/mlflow/`).

## 🧠 Decisiones de diseño (nivel senior)

- **Una sola imagen** para entrenar y servir (`trainer` y `api` = misma imagen, distinto comando): elimina divergencias de dependencias entre training y serving, y el lockfile hace el build reproducible.
- **Almacén de modelos versionado con puntero atómico**: publicar es renombrar un directorio y sustituir un fichero de texto; volver atrás es mover el puntero. La API no depende de MLflow para arrancar → MLflow es *plus* de trazabilidad, no punto único de fallo, pero el alias `champion` lo convierte en la fuente de "qué modelo está en producción".
- **Gate de calidad en el pipeline, no solo en los tests**: un reentreno que degrada el AUC no llega nunca al volumen de modelos; el CI además falla si un cambio de código degrada el modelo — el modelo se trata como código.
- **Un modelo solo se sirve si este runtime puede cargarlo**: versiones de scikit-learn y features comprobadas antes de servir; el fallo silencioso tras reconstruir la imagen era el riesgo más caro que quedaba.
- **Un reload o rollback fallido nunca degrada el servicio**: se conserva el último modelo bueno conocido y el error se devuelve al operador.
- **Drift con PSI/KS implementado y auditable** en lugar de una dependencia pesada, con *binning* robusto para variables binarias/enteras (donde los cuantiles clásicos enmascaran el cambio), *prediction drift* y propiedades verificadas con Hypothesis; migrar a Evidently es trivial si el equipo lo prefiere.
- **Predicciones en SQLite dentro del volumen**: cero infraestructura adicional, compartido entre workers y persistente; el contrato del endpoint no cambia si mañana el sink es un warehouse.
- **Fuente única de verdad del dominio de las features**: los rangos y categorías viven en el validador y el contrato Pydantic se deriva de ellos (un test de contrato lo garantiza).
- **Observabilidad de serie**: métricas por plantilla de ruta (cardinalidad acotada), `X-Request-ID` de extremo a extremo y un único formato JSON para la aplicación y uvicorn.

## 🗺️ Posibles extensiones

Reentrenamiento programado por drift ya cubierto por `retrain-if-drift.sh`; quedan calibración de probabilidades y ajuste de umbrales con curvas precision-recall, hash del dataset en el metadata para datos reales, feature store, A/B de modelos (shadow deployment), explicabilidad SHAP por predicción y export a ONNX para latencias < 5 ms.

---
*Proyecto de portfolio orientado a roles **ML Engineer / MLOps Engineer**. Licencia MIT.*
