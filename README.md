# 📈 MLOps Churn Platform

Plataforma **MLOps end-to-end** para predicción de bajas de clientes (*churn*): generación/validación de datos, entrenamiento reproducible con **gate de calidad** que protege el modelo en servicio, registro en **MLflow**, serving en tiempo real con **FastAPI** y **monitorización de drift** (PSI + Kolmogorov-Smirnov sobre las features y PSI sobre las predicciones) del tráfico de producción.

> **Por qué este proyecto**: *ML Engineer* es el perfil IA con más ofertas activas en España y *MLOps* tiene más vacantes que candidatos (salarios senior 78-100 k€). Lo que piden esas ofertas es exactamente esto: llevar un modelo del prototipo a producción con ciclo de vida completo (entrenar → registrar → servir → monitorizar → reentrenar).

---

## ✨ Qué hace

1. **Genera datos sintéticos realistas** de un negocio de suscripciones (reproducibles por semilla) — sin depender de datos privados.
2. **Valida los datos antes de entrenar** (esquema, nulos, rangos, categorías, balance de clases): *fail-fast*. Los rangos y categorías son la **fuente única de verdad** que también usa el contrato de la API.
3. **Entrena** un `HistGradientBoostingClassifier` (pipeline sklearn con preprocesado incluido), evalúa (AUC, F1, Brier) y aplica un **gate de calidad** (`CHURN_MIN_ROC_AUC`): si el modelo no lo supera, el entrenamiento falla y **no toca** los artefactos del modelo en servicio.
4. **Persiste artefactos versionados de forma atómica**: `model.joblib` + `metadata.json` + `reference.csv` (muestra para drift). Un reload nunca ve un fichero a medio escribir.
5. **Registra** parámetros, métricas y modelo en **MLflow Model Registry** (opcional, activado por variable de entorno). Si MLflow no responde, el entrenamiento avisa y termina igualmente: el tracking es trazabilidad, no un punto único de fallo. El `run_id` queda enlazado en `metadata.json` y en `/model/info`.
6. **Sirve** predicciones vía API REST (individual y batch) con niveles de riesgo de negocio (`bajo/medio/alto`).
7. **Monitoriza drift**: la API acumula las features y las probabilidades del tráfico real; `/monitoring/drift` calcula PSI y KS por variable contra la referencia de entrenamiento y PSI de las **predicciones** contra las puntuaciones de la referencia (*prediction drift*).
8. **Reentrena y recarga sin parar el servicio**: `docker compose run trainer` + `POST /model/reload` (autenticado). Si el reload falla (artefactos ausentes o corruptos), la API **sigue sirviendo el modelo anterior** y responde 404/500 con el motivo.

## 🏗️ Arquitectura

```
        (cron / manual / CI)                        volumen compartido /models
 ┌──────────────┐  valida+entrena  ┌──────────────────┐   model.joblib
 │   trainer     │──gate calidad──▶│  LocalModelStore │   metadata.json
 │ (mismo image) │──log runs──┐    │ (escritura atómica)   reference.csv
 └──────────────┘             ▼    └──────────────────┘
                        ┌──────────┐        ▲ carga al arrancar / reload
                        │  MLflow  │  ┌─────┴─────┐   /predict /predict/batch
  clientes ────────────▶│ registry │  │ API FastAPI│──▶ /model/info /monitoring/drift
                        └──────────┘  └───────────┘
                                            │ buffer de features + probabilidades
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
| API de scoring | http://localhost:8010 (docs en `/docs`) |
| MLflow UI | http://localhost:5000 |

Prueba de scoring:

```bash
curl -X POST http://localhost:8010/predict -H 'Content-Type: application/json' -d '{
  "tenure_months": 3, "monthly_charges": 95.5, "total_charges": 280,
  "num_products": 1, "support_tickets_90d": 6, "is_fiber": 0,
  "contract_type": "mensual", "payment_method": "transferencia"
}'
# → {"churn_probability": 0.87, "risk_level": "alto", "model_version": "20260711-101500-3f9a1c"}
```

Reentrenar y recargar en caliente:

```bash
make retrain      # docker compose run trainer + POST /model/reload (falla si el gate no pasa)
```

Informe de drift del tráfico acumulado:

```bash
make drift        # → PSI y KS por feature, PSI de predicciones y lista de features con drift
```

Sin Docker: `make install && make run` (entrena y sirve en local).

## ⚙️ Configuración

| Variable | Por defecto | Descripción |
|---|---|---|
| `CHURN_ENVIRONMENT` | `dev` | En `production` la API se niega a arrancar con un token de administración inseguro |
| `CHURN_MODEL_DIR` | `models` | Directorio de artefactos (volumen compartido) |
| `CHURN_MLFLOW_TRACKING_URI` | *(vacío = desactivado)* | URL de MLflow para tracking/registry |
| `CHURN_MIN_ROC_AUC` | `0.75` | Gate de calidad: AUC mínimo para publicar artefactos (también `--min-auc` en el trainer) |
| `CHURN_RISK_MEDIUM` / `CHURN_RISK_HIGH` | `0.35` / `0.65` | Umbrales de riesgo de negocio (inclusivos; `medium < high` obligatorio) |
| `CHURN_DRIFT_MIN_ROWS` | `200` | Mínimo de predicciones para calcular drift |
| `CHURN_DRIFT_BUFFER_SIZE` | `5000` | Tamaño del buffer rodante de tráfico reciente |
| `CHURN_PSI_ALERT_THRESHOLD` | `0.2` | Umbral de alerta PSI |
| `CHURN_ADMIN_TOKEN` | — | Token del endpoint `/model/reload` (>= 16 caracteres en producción; `openssl rand -hex 32`) |

## 📡 API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | 200 con la versión del modelo cargado; 503 si no hay modelo (*readiness*) |
| `POST` | `/predict` | Probabilidad de churn + nivel de riesgo para un cliente |
| `POST` | `/predict/batch` | Lo mismo para 1-1000 clientes |
| `GET` | `/model/info` | Versión, métricas, gate de calidad, semilla, `run_id` de MLflow |
| `POST` | `/model/reload` | Recarga los artefactos (cabecera `X-Admin-Token`); conserva el modelo anterior si falla |
| `GET` | `/monitoring/drift` | Informe de drift; 409 mientras no haya `CHURN_DRIFT_MIN_ROWS` predicciones |

Tras un reload, el drift de predicciones solo compara puntuaciones producidas por la **versión en servicio** (las del modelo anterior no son comparables); las features del tráfico anterior siguen contando.

## ✅ Tests automatizados

```bash
make install && make test       # suite completa (~20 s, sin servicios externos)
make coverage                   # con informe de cobertura (umbral 90%, el mismo que CI)
```

Qué cubren (más de 140 tests, cobertura de líneas y ramas > 99 %):

- **Generador**: esquema, reproducibilidad, señal predictiva, dtypes y contrato con los rangos del validador.
- **Validación**: 15 casos de fallo (columnas, nulos sin duplicar errores, rangos, categorías, tamaño, tasa de churn, target inválido, columnas duplicadas...).
- **Entrenamiento**: artefactos, **gate de calidad AUC** (no sobreescribe el modelo anterior), metadata de trazabilidad, versiones únicas, MLflow desactivado / no instalado / caído / correcto (con dobles, sin servidor) y CLI con códigos de salida.
- **Registro de artefactos**: juego incompleto, metadata/modelo/referencia corruptos, escritura atómica sin residuos, actualización de metadata.
- **Drift**: PSI en binarias desbalanceadas, referencias constantes, baja cardinalidad, nulos, categóricas con tipos mezclados, *prediction drift* y estructura del informe.
- **Configuración**: prefijo `CHURN_`, umbrales coherentes, guardia de token en producción.
- **API**: coherencia de riesgo, validación 422, batch consistente con individual y sus límites, 503 sin modelo, flujo de drift 409→200, drift detectado a través de la API, buffer acotado, reload autenticado con token incorrecto / nueva versión / artefactos ausentes / corruptos, arranque con artefactos corruptos y recuperación, OpenAPI.
- **Contratos**: el esquema Pydantic de la API coincide con los rangos y categorías del validador.

El entrenamiento de test usa un modelo pequeño compartido por sesión → suite rápida y sin servicios externos.

## 🔁 CI/CD (GitHub Actions)

- **`ci.yml`**: `ruff` + sintaxis de scripts de despliegue + `pytest` con cobertura mínima del 90 % + validación de los ficheros compose + build de la imagen en cada push/PR.
- **`deploy.yml`** (tag `vX.Y.Z`): build → push a **GHCR** (`X.Y.Z`, `sha-…` y `latest`) → túnel **WireGuard** → SSH al VPS netcup → `deploy/scripts/deploy.sh` (pull, up, espera a `/health`, entrenamiento inicial si el volumen está vacío, rollback automático si la API no responde).

## 🏭 Despliegue a producción (VPS netcup + VPN)

Guía completa: [`deploy/DESPLIEGUE_NETCUP.md`](deploy/DESPLIEGUE_NETCUP.md). Resumen:

1. `bash deploy/scripts/setup-vps.sh` en el VPS (Docker, ufw, fail2ban, WireGuard).
2. Alta de peers VPN (portátil + runner CI) y login GHCR en el VPS.
3. Clonar en `/opt/mlops-churn-platform`, `cp .env.example .env` (token seguro y `VPN_BIND_IP` obligatorios), `TAG=latest ./deploy/scripts/deploy.sh`.
4. Secretos en GitHub (`WG_CONFIG`, `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`) y desplegar con `git tag v1.0.0 && git push --tags`.
5. Reentrenos en producción: `docker compose -f docker-compose.prod.yml --profile train run --rm trainer && curl -fsS -X POST localhost:8010/model/reload -H "X-Admin-Token: $TOKEN"` (o prográmalo con cron). Rollback: `TAG=$(cat .previous_tag) ./deploy/scripts/deploy.sh`.

La API y MLflow solo son accesibles **dentro de la VPN** (Caddy con TLS; MLflow bajo `/mlflow/`).

## 🧠 Decisiones de diseño (nivel senior)

- **Una sola imagen** para entrenar y servir (`trainer` y `api` = misma imagen, distinto comando): elimina divergencias de dependencias entre training y serving.
- **Contrato de artefactos explícito y atómico** (`model.joblib` + `metadata.json` + `reference.csv`, escritos vía *staging* + `os.replace`): la API no depende de MLflow para arrancar → MLflow es *plus* de trazabilidad, no punto único de fallo.
- **Gate de calidad en el pipeline, no solo en los tests**: un reentreno que degrada el AUC no llega nunca al volumen de modelos; el CI además falla si un cambio de código degrada el modelo — el modelo se trata como código.
- **Un reload fallido nunca degrada el servicio**: se conserva el último modelo bueno conocido y el error se devuelve al operador.
- **Drift con PSI/KS implementado y auditable** en lugar de una dependencia pesada, con *binning* robusto para variables binarias/enteras (donde los cuantiles clásicos enmascaran el cambio) y *prediction drift*; migrar a Evidently es trivial si el equipo lo prefiere.
- **Fuente única de verdad del dominio de las features**: los rangos y categorías viven en el validador y el contrato Pydantic se deriva de ellos (un test de contrato lo garantiza).
- **Buffer de inferencia en memoria** para drift: cero fricción para la demo; en producción real se sustituiría por un sink a base de datos/warehouse (el contrato del endpoint no cambia).

## 🗺️ Posibles extensiones

Reentrenamiento programado por drift (webhook → Actions), almacén de modelos versionado con rollback por API, feature store, A/B de modelos (shadow deployment), explicabilidad SHAP por predicción, métricas Prometheus y export a ONNX para latencias < 5 ms.

---
*Proyecto de portfolio orientado a roles **ML Engineer / MLOps Engineer**. Licencia MIT.*
