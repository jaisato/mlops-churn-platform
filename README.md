# 📈 MLOps Churn Platform

[![CI](https://github.com/jaisato/mlops-churn-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/jaisato/mlops-churn-platform/actions/workflows/ci.yml)

Plataforma **MLOps end-to-end** para predicción de bajas de clientes (*churn*): generación/validación de datos, entrenamiento reproducible desde datos etiquetados (CSV/Parquet) o sintéticos con **gate de calidad** y **promoción campeón/retador** que protegen el modelo en servicio, **almacén de modelos versionado con rollback**, registro en **MLflow** con alias de campeón, serving en tiempo real con **FastAPI**, **monitorización de drift** persistente (PSI + Kolmogorov-Smirnov sobre las features y PSI sobre las predicciones), **bucle de etiquetas cerrado** (el churn real vuelve a la plataforma, mide el acierto del modelo en producción y alimenta el reentreno automático) y **métricas Prometheus**.

> **Por qué este proyecto**: *ML Engineer* es el perfil IA con más ofertas activas en España y *MLOps* tiene más vacantes que candidatos (salarios senior 78-100 k€). Lo que piden esas ofertas es exactamente esto: llevar un modelo del prototipo a producción con ciclo de vida completo (entrenar → registrar → servir → monitorizar → reentrenar → volver atrás si hace falta).

---

## ✨ Qué hace

1. **Entrena desde datos etiquetados** (`--data clientes.csv|.parquet`, mismas columnas que el contrato de la API) o **genera datos sintéticos realistas** de un negocio de suscripciones (reproducibles por semilla). El generador tiene **controles de drift explícitos** (`--drift-shift`): desplaza las features y la relación con la etiqueta para producir, de forma determinista, "el mundo ha cambiado" en tests y demos.
2. **Valida los datos antes de entrenar** (esquema, nulos, rangos, categorías, balance de clases): *fail-fast*. Los rangos y categorías son la **fuente única de verdad** que también usa el contrato de la API.
3. **Entrena** un `HistGradientBoostingClassifier` (pipeline sklearn con preprocesado incluido), evalúa (AUC, F1, Brier) y aplica un **gate de calidad** (`CHURN_MIN_ROC_AUC`): si el modelo no lo supera, el entrenamiento termina con código 2 y **no toca** el modelo en servicio.
4. **Compara el retador con el campeón** sobre el *holdout* de los datos nuevos: el modelo en servicio se evalúa en las mismas filas que el recién entrenado y **solo se promueve** (puntero `current` + alias MLflow) si su AUC es al menos igual, con margen configurable (`CHURN_PROMOTION_MARGIN`). El holdout es **estable y por grupos**: el 20 % de los sujetos (`subject_ref`) o, sin él, de las filas, elegido por un hash con sal fija, así que al acumular etiquetas ninguna fila cambia de lado entre reentrenos y un cliente puntuado varias veces nunca queda a ambos lados. Si el campeón se entrenó con etiquetas, **solo se le mide con las predichas después** de las suyas, y sin al menos 80 grupos con ambas clases no se le sustituye. Si pierde, la versión se guarda **sin promover** (auditable en `/model/versions`, recuperable con rollback explícito) y el trainer termina con código 3. La decisión, la evidencia usada, los AUC de ambos y la **huella SHA-256 del dataset** quedan en `metadata.json`; reentrenar con los mismos datos que el campeón (misma huella) no promueve nada.
5. **Publica cada modelo como una versión inmutable** en `models/versions/<versión>/` (`model.joblib` + `metadata.json` + `reference.csv` — la referencia de drift sale de los datos con los que se entrenó **esa** versión) y mueve el puntero `models/current` de forma atómica. Las versiones antiguas se podan (`CHURN_MODEL_KEEP_VERSIONS`), nunca la que está en servicio.
6. **Registra** parámetros, métricas, decisión de promoción y modelo en **MLflow Model Registry** (opcional) y asigna el alias **`champion`** a cada versión promovida. Si MLflow no responde, el entrenamiento avisa y termina igualmente: el tracking es trazabilidad, no un punto único de fallo. El `run_id` y la versión del registry quedan en `metadata.json` y en `/model/info`.
7. **Comprueba la compatibilidad antes de servir**: el metadata guarda las versiones de Python y scikit-learn con las que se serializó el pipeline; la API rechaza modelos con otras features o con otra versión menor de scikit-learn (los pickles no son portables) en lugar de servirlos a ciegas.
8. **Sirve** predicciones vía API REST (individual y batch) con niveles de riesgo de negocio (`bajo/medio/alto`) y un **`prediction_id`** por predicción (más una clave opaca opcional del cliente, `subject_ref`), API key opcional, `X-Request-ID` y logs JSON.
9. **Monitoriza drift** con las predicciones guardadas en **SQLite dentro del volumen** (sobreviven a reinicios y se comparten entre workers): PSI y KS por variable contra la referencia de entrenamiento y PSI de las **predicciones** contra las puntuaciones de la referencia, expuesto también como métricas Prometheus.
10. **Cierra el bucle de etiquetas**: cuando el negocio conoce el churn real de un cliente puntuado lo devuelve con `POST /labels` (`prediction_id` + `churn` + `observed_at`, idempotente, con un **token propio** `CHURN_LABELS_TOKEN` para que el job del CRM no necesite el de administración). La etiqueta se guarda **junto a la predicción** (features, probabilidad, versión que la produjo y sujeto) y se puede corregir aunque la predicción ya haya salido de la ventana; una `observed_at` anterior a la predicción se rechaza. `GET /monitoring/performance` mide el **acierto real** del modelo (AUC, precision, recall, F1, Brier) en conjunto y por versión, y `python -m churn.data.labels` convierte el historial etiquetado en un dataset de reentreno (**una etiqueta por sujeto y horizonte**) con **huella SHA-256 y ventana temporal** que acaban en el metadata del modelo entrenado con él.
11. **Reentrena, recarga y vuelve atrás sin parar el servicio**: `retrain-if-drift.sh` (una ejecución a la vez, con `flock`) consulta el drift y, si lo hay, construye el dataset con las etiquetas recibidas (o usa `CHURN_TRAIN_DATA` si el operador aporta un fichero), no hace nada si no han llegado etiquetas nuevas desde el modelo en servicio (misma huella), reentrena, promueve solo si el retador gana y entonces hace `POST /model/reload`; `POST /model/rollback` devuelve la versión anterior (o una concreta). Un reload o rollback fallido **nunca degrada el servicio**: se conserva el último modelo bueno.

## 🏗️ Arquitectura

```
  datos etiquetados (--data)      valida+entrena        volumen compartido /models
  o sintéticos (--drift-shift)     gate calidad       current ─▶ versions/<v>/model.joblib
 ┌──────────────┐  campeón vs retador ┌────────────────────┐                     metadata.json
 │   trainer     │──promueve si gana─▶│  LocalModelStore   │                     reference.csv
 │ (mismo image) │──log runs──┐       │ (versionado+atómico)  drift.sqlite (predicciones + etiquetas)
 └──────────────┘             ▼       └────────────────────┘  datasets/labels.parquet (+ .meta.json)
        ▲ --data        ┌──────────┐        ▲ carga al arrancar / reload / rollback
        │               │  MLflow  │  ┌─────┴─────┐   /predict /predict/batch ◀── clientes
  python -m             │ registry │  │ API FastAPI│──▶ /model/info /model/versions /model/rollback
  churn.data.labels     │ @champion│  └─────┬─────┘   /monitoring/drift  /monitoring/performance
  (etiquetas ─▶ dataset)└──────────┘        │          /metrics  /health/live
        ▲                                   ▼ features + probabilidad + prediction_id
  drift.sqlite ◀──────── POST /labels: churn real por prediction_id (negocio / CRM)
                            drift PSI + KS (features) y PSI (predicciones)
                            rendimiento real: AUC / precision / recall por versión
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
  "contract_type": "mensual", "payment_method": "transferencia",
  "subject_ref": "c-7f3a9c41"
}'
# → {"prediction_id": "3f2a…", "churn_probability": 0.87, "risk_level": "alto", "model_version": "20260920-101500-3f9a1c"}
```

`subject_ref` es opcional: una clave **opaca** y estable del cliente (p. ej. un hash de su id interno, nunca datos personales) que no llega al modelo, pero permite saber que varias predicciones son del mismo cliente. Semanas después, cuando se sabe si ese cliente se dio de baja, el negocio cierra el bucle (en local, sin `CHURN_LABELS_TOKEN`, vale el token de administración):

```bash
curl -X POST http://localhost:8010/labels -H 'X-Admin-Token: token-local' -H 'Content-Type: application/json' \
  -d '{"labels": [{"prediction_id": "3f2a…", "churn": 1, "observed_at": "2026-10-20T09:00:00Z"}]}'
# → {"received": 1, "created": 1, "updated": 0, "unknown": [], "rejected": [], "labelled_total": 1}
make performance   # AUC / precision / recall reales sobre las predicciones etiquetadas, por versión
make retrain       # construye /models/datasets/labels.parquet con esas etiquetas, reentrena y recarga si gana
```

Ciclo completo drift → reentreno → recuperación, ensayable sin datos reales:

```bash
make drift-data                          # data/churn-drift.csv: mundo desplazado, etiquetado (--drift-shift 1.0)
python scripts/generate_data.py --rows 400 --seed 44 --drift-shift 1.0 --out data/trafico.csv
# ...envía data/trafico.csv (sin la columna churn) a /predict/batch: el tráfico ya no se parece a la referencia
make drift                               # → drift_detected: true (tenure_months, monthly_charges, ...)
make retrain DATA=data/churn-drift.csv   # entrena con --data, gana al campeón sobre el holdout nuevo, promueve y recarga
make drift                               # → la nueva referencia sale de los datos nuevos: sin drift
curl -X POST localhost:8010/model/rollback -H 'X-Admin-Token: token-local'   # versión anterior
curl localhost:8010/model/versions       # qué hay publicado y qué decisión de promoción tuvo cada versión
```

`make retrain` sin `DATA=` reentrena con las **etiquetas recibidas por la API**: construye el dataset dentro del volumen (`python -m churn.data.labels`, código 3 y sin reentreno si hay menos de `CHURN_LABELS_MIN_ROWS`) y entrena con `--data`. Si el retador no gana, el trainer termina con código 3, la API no se recarga y la versión queda en `/model/versions` como `rejected`.

Sin Docker: `make install && make run` (entrena y sirve en local; requiere Python 3.11+); `make train DATA=fichero.parquet` entrena desde un fichero y `make labels-dataset` escribe `data/labels.parquet` (más `data/labels.parquet.meta.json`, el *sidecar* con filas, ventana temporal y huella) con las etiquetas recibidas por la API local, listo para `make train DATA=data/labels.parquet`.

## ⚙️ Configuración

| Variable | Por defecto | Descripción |
|---|---|---|
| `CHURN_ENVIRONMENT` | `dev` | En `production` la API se niega a arrancar con un token de administración inseguro |
| `CHURN_MODEL_DIR` | `models` | Directorio de artefactos (volumen compartido) |
| `CHURN_MODEL_KEEP_VERSIONS` | `5` | Versiones publicadas que se conservan (la actual nunca se borra; los retadores rechazados no desalojan a los campeones anteriores) |
| `CHURN_STRICT_ARTIFACT_COMPAT` | `true` | Rechazar modelos serializados con otra versión menor de scikit-learn |
| `CHURN_MLFLOW_TRACKING_URI` | *(vacío = desactivado)* | URL de MLflow para tracking/registry |
| `CHURN_MLFLOW_CHAMPION_ALIAS` | `champion` | Alias asignado en el registry a cada versión que supera el gate (`""` = ninguno) |
| `CHURN_MIN_ROC_AUC` | `0.75` | Gate de calidad: AUC mínimo para publicar artefactos (también `--min-auc` en el trainer) |
| `CHURN_PROMOTION_MARGIN` | `0` | Campeón/retador: el modelo nuevo pasa a `current` si `AUC_nuevo >= AUC_actual - margen` sobre el holdout de los datos nuevos (`1` = promover siempre que pase el gate, salvo con los mismos datos que el campeón; también `--promotion-margin`) |
| `CHURN_TRAIN_DATA` | *(vacío)* | Solo `retrain-if-drift.sh`: ruta en el host a un `.csv`/`.parquet` **etiquetado** propio (export del CRM). Sin ella el script reentrena con las etiquetas recibidas por `POST /labels` |
| `CHURN_LABELS_MIN_ROWS` | `500` | Mínimo de ejemplos etiquetados (tras quedarse con una etiqueta por sujeto y horizonte) para construir el dataset de reentreno (`python -m churn.data.labels --min-rows`); no puede bajar de 500, el mínimo del validador de entrenamiento. Por debajo, el script avisa y no reentrena |
| `CHURN_PREDICTION_KEEP_ROWS` | `100000` | Predicciones que conserva el almacén para poder etiquetarlas (ver [retención](#retención-y-dimensionado-del-almacén)). Una predicción desalojada sin etiquetar ya no se puede etiquetar; las ya etiquetadas se conservan aparte, sin límite, y se pueden corregir |
| `CHURN_PREDICTION_KEEP_DAYS` | `0` | Antigüedad máxima de las predicciones en días, además del límite de filas (`0` = sin límite) |
| `CHURN_RISK_MEDIUM` / `CHURN_RISK_HIGH` | `0.35` / `0.65` | Umbrales de riesgo de negocio (inclusivos; `medium < high` obligatorio) |
| `CHURN_DRIFT_MIN_ROWS` | `200` | Mínimo de predicciones (drift) o de predicciones etiquetadas (rendimiento) para calcular el informe |
| `CHURN_DRIFT_BUFFER_SIZE` | `5000` | Ventana reciente que miran `/monitoring/drift` y `/monitoring/performance` |
| `CHURN_DRIFT_STORE` / `CHURN_DRIFT_DB_PATH` | `sqlite` / `<model_dir>/drift.sqlite` | Dónde viven las predicciones y sus etiquetas (`memory` = por proceso, solo demos) |
| `CHURN_PSI_ALERT_THRESHOLD` | `0.2` | Umbral de alerta PSI |
| `CHURN_ADMIN_TOKEN` | — | Token de `/model/reload` y `/model/rollback` (y de `/labels` si no hay `CHURN_LABELS_TOKEN`); >= 16 caracteres en producción (`openssl rand -hex 32`) |
| `CHURN_LABELS_TOKEN` | *(vacío)* | Token propio de `POST /labels` (cabecera `X-Labels-Token`): el job del CRM que envía etiquetas no puede recargar ni hacer rollback. Si se define, `/labels` solo acepta este (no el de administración); vacío = `/labels` exige `X-Admin-Token` y la API lo avisa al arrancar en producción. Distinto del de administración y >= 16 caracteres en producción |
| `CHURN_API_KEY` | *(vacío = abierto)* | Si se define, `/predict*`, `/model/info`, `/model/versions` y `/monitoring/*` exigen `X-API-Key` |
| `CHURN_METRICS_ENABLED` / `CHURN_ACCESS_LOG` | `true` / `true` | `/metrics` Prometheus y una línea JSON por petición |

## 📡 API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health/live` | Liveness: el proceso responde (siempre 200) |
| `GET` | `/health` | Readiness: 200 con la versión del modelo cargado; 503 si no hay modelo |
| `GET` | `/metrics` | Métricas Prometheus: peticiones y latencia por ruta, predicciones por riesgo y versión, histograma de probabilidades, PSI del último informe, versión en servicio, etiquetas recibidas por resultado (`churn_labels_total`), etiquetadas disponibles (`churn_labels_stored`) y rendimiento real por versión (`churn_performance_*`) |
| `POST` | `/predict` | Probabilidad de churn + nivel de riesgo para un cliente, con el `prediction_id` que identifica la predicción para etiquetarla después. `subject_ref` opcional (clave opaca del cliente, 1-128 caracteres; no es una feature) |
| `POST` | `/predict/batch` | Lo mismo para 1-1000 clientes (`subject_ref` opcional por cliente) |
| `POST` | `/labels` | Churn real de 1-1000 predicciones ya servidas: `[{"prediction_id", "churn": 0\|1, "observed_at"?, "subject_ref"?}]` (cabecera `X-Labels-Token` si `CHURN_LABELS_TOKEN` está definido; si no, `X-Admin-Token`; sin credenciales, 401 antes de validar el cuerpo). **Idempotente por `prediction_id`**: la última etiqueta recibida es la que vale, reenviar un lote no duplica nada y una corrección funciona aunque la predicción ya haya salido de la ventana. Por elemento, sin invalidar el resto: `unknown` (el id no está en el almacén) y `rejected` (`observed_at` anterior a la predicción). `observed_at` ausente = instante de recepción; futura (más de 5 minutos) o ids repetidos = 422 |
| `GET` | `/monitoring/performance` | Acierto real sobre las predicciones etiquetadas más recientes (`CHURN_DRIFT_BUFFER_SIZE`): AUC, accuracy, precision, recall, F1 y Brier con umbral 0.5 (el mismo que el holdout del entrenamiento), en conjunto y **por versión del modelo**, más el AUC de holdout de la versión en servicio para comparar; 409 mientras no haya `CHURN_DRIFT_MIN_ROWS` etiquetadas. No exige modelo cargado |
| `GET` | `/model/info` | Versión, métricas, gate, **decisión de promoción** (AUC del campeón y del retador, filas y grupos comparados y corte temporal del campeón), **origen y huella de los datos** (si salen de las etiquetas, `data_source.labels` con su ventana temporal, etiquetas, sujetos y versiones que puntuaron; `kind` sigue siendo `file`), **split** (grupos de entrenamiento y holdout), semilla, runtime, `run_id` y versión de MLflow |
| `GET` | `/model/versions` | Versiones publicadas en el almacén (con su decisión `promoted`/`rejected`/`no_champion`), cuál apunta `current` y cuál sirve este proceso |
| `POST` | `/model/reload` | Carga la versión `current` (cabecera `X-Admin-Token`); conserva la anterior si falla |
| `POST` | `/model/rollback` | Vuelve a la versión anterior que llegó a servirse (se salta los retadores rechazados) o a `{"version": "..."}`; mueve `current` solo si carga bien |
| `GET` | `/monitoring/drift` | Informe de drift; 409 mientras no haya `CHURN_DRIFT_MIN_ROWS` predicciones |

Tras un reload o rollback, el drift de predicciones solo compara puntuaciones producidas por la **versión en servicio**; las features del tráfico anterior siguen contando. El rendimiento real, en cambio, se desglosa por versión: la etiqueta llega semanas después de puntuar, así que una ventana suele mezclar la versión en servicio con la anterior. Cada respuesta lleva `X-Request-ID` (propagado si el cliente lo envía) y el mismo identificador aparece en la línea JSON del log de acceso.

### Dataset de reentreno a partir de las etiquetas

```bash
python -m churn.data.labels --model-dir /models                      # → /models/datasets/labels.parquet (+ .meta.json)
python -m churn.data.labels --db models/drift.sqlite --out data/labels.csv --min-rows 800
```

Une cada etiqueta con las features y la probabilidad de su predicción y escribe un `.csv`/`.parquet` con **las columnas del contrato de entrenamiento** más cuatro de trazabilidad que no son features (`prediction_id`, `subject_ref`, `predicted_at`, `observed_at`; el trainer las usa para el split por sujeto y para comparar con el campeón solo con etiquetas posteriores a las suyas), en orden de predicción y validado con las mismas reglas que el trainer. **Una etiqueta por sujeto y horizonte**: si el mismo cliente se puntuó varias veces y esas predicciones se etiquetaron con la misma observación (`observed_at`), solo cuenta la más reciente; sin `subject_ref`, las filas idénticas cuentan una vez. Junto al fichero deja el *sidecar* `<fichero>.meta.json` (filas, etiquetas antes de deduplicar, sujetos, tasa de churn, rango de fechas de predicción y de observación, versiones que puntuaron, huella SHA-256); ambos se escriben en un temporal y se renombran (un lector nunca ve un fichero a medias). Cuando el trainer carga ese fichero con `--data` y la huella coincide, el metadata del modelo registra esa ventana en `data_source.labels`. Imprime un resumen JSON por stdout (los logs van a stderr) y termina con `0` (escrito), `3` (menos de `--min-rows`/`CHURN_LABELS_MIN_ROWS` ejemplos) o `2` (almacén inexistente o dataset inválido).

### Madurez de la etiqueta y `observed_at`

`churn` responde a la pregunta de la predicción: ¿se dio de baja el cliente **dentro del horizonte** (p. ej. 90 días desde `predicted_at`)? Un `1` se puede enviar en cuanto se produce la baja; un `0` solo es firme cuando el horizonte ha vencido sin baja (**etiqueta madura**): enviarlo antes es un falso negativo que sesga el reentreno y el acierto real. `observed_at` es cuándo se observó el resultado (la fecha de la baja, o el cierre del horizonte para un `0`), no cuándo se envía: la API rechaza (`rejected`) las anteriores a la predicción y devuelve 422 si es futura (holgura de 5 minutos para desfases de reloj; una hora local sin zona suele delatarse así). Con `subject_ref`, todas las predicciones de un cliente etiquetadas con la misma observación cuentan como un solo ejemplo (la más reciente).

### Retención y dimensionado del almacén

`drift.sqlite` guarda las predicciones hasta que se etiquetan: `CHURN_PREDICTION_KEEP_ROWS` debe cubrir **predicciones por día × días hasta que llega la etiqueta** (el horizonte más el retraso del CRM), con margen; p. ej. 2 000 predicciones/día × 120 días × 1.5 = 360 000. Cada predicción ocupa unos 0.36 KB (100 000 ≈ 36 MB) y el desalojo cuesta lo mismo con 5 000 que con un millón de filas (por id, no recorre la ventana). `CHURN_PREDICTION_KEEP_DAYS` acota además la antigüedad (minimización de datos). Una predicción desalojada sin etiquetar ya no se puede etiquetar (`unknown`); las etiquetadas se conservan aparte, sin límite, y siguen siendo corregibles.

## ✅ Tests automatizados

```bash
make install && make test       # suite completa (~25 s, sin servicios externos)
make check                      # lo mismo que el CI: ruff, mypy y cobertura (umbral 90 %)
```

Qué cubren (431 tests en 18 módulos, cobertura de líneas y ramas del 100 %):

- **Generador**: esquema, reproducibilidad, señal predictiva, dtypes, contrato con los rangos del validador y **drift determinista** (`DriftSpec`: sin especificación el dataset es bit a bit el de siempre; con ella las features superan el umbral PSI, la relación con la etiqueta cambia y todo sigue dentro del contrato).
- **Fuentes de datos**: CSV y Parquet, extensión desconocida, fichero ausente, vacío o corrupto, Parquet sin motor, huella SHA-256 estable y sensible al contenido.
- **Validación**: 15 casos de fallo sin duplicar errores, acumulación de errores y sincronía con el generador.
- **Entrenamiento**: artefactos versionados, gate de calidad que no sobreescribe, metadata de trazabilidad con runtime, origen de datos, split y promoción, versiones únicas, MLflow desactivado / no instalado / caído / correcto / alias fallido / **retador rechazado sin alias** (con dobles, sin servidor) y CLI (`--data`, `--drift-shift`) con códigos de salida 0/2/3.
- **Reentrenamiento real** (`test_retraining.py`): reentrenar con el mismo generador y semilla produce un modelo idéntico, **el drift no cambia** y ya no se promueve; reentrenar con datos del mundo nuevo produce otro modelo, gana al campeón y **el drift desaparece**; retador peor guardado sin promover, margen, campeón ilegible o con otras features, y la API sigue sirviendo al campeón tras un reload. **Holdout estable por grupos**: el split aleatorio anterior metía en el holdout del 2º reentreno filas de entrenamiento del campeón (aquí, ~23 %) y el nuevo nunca; **dos reentrenos sucesivos** por el camino real (almacén → etiquetas → dataset acumulado) en los que el retador mejor sí se promueve y el campeón solo se mide con etiquetas posteriores a las suyas; evidencia nueva insuficiente; datos sin `predicted_at`; holdout sin ambas clases; y un **cliente puntuado varias veces** (300 y 1000 clientes × 4) que ya no promueve un modelo peor.
- **Almacén de modelos**: publicación atómica, retención, layout plano heredado, rollback en ambas direcciones, versiones incompletas o corruptas, compatibilidad de features y runtime.
- **Almacén de predicciones y etiquetas**: memoria y SQLite con el mismo contrato, `prediction_id` único, `subject_ref`, ventana rodante (también predicción a predicción) y retención por antigüedad, persistencia entre instancias, escrituras y etiquetado concurrentes, etiquetas que crean/corrigen/reportan desconocidos o rechazan las observadas antes de predecir, normalización de `observed_at` a UTC, etiquetas que sobreviven al desalojo y **se pueden corregir después**, orden de predicción y límite, migración de un `drift.sqlite` anterior al bucle (y de uno sin `subject_ref`), **migración concurrente** (otro worker con el cerrojo: antes "duplicate column name") y migración fallida sin esquema a medias.
- **Dataset de etiquetas** (`test_labels.py`): columnas del contrato y de trazabilidad en orden de predicción, **una etiqueta por sujeto y horizonte** y filas idénticas sin sujeto, ventana temporal y versiones, CSV/Parquet con *sidecar* cuya huella es la que calcula el trainer, **escritura atómica** (un fallo a mitad conserva el dataset anterior), mínimos (insuficiente, por debajo del validador), dataset inválido, formato no soportado y CLI (defectos de `settings`, `--db/--out/--min-rows`, almacén inexistente, códigos 0/2/3 y resumen JSON solo en stdout).
- **Rendimiento real** (`test_performance.py`): métricas con ambas clases y con una sola (AUC indefinido), umbral, informe global y por versión.
- **Bucle completo** (`test_label_loop.py`): la API puntúa un mundo desplazado, el negocio devuelve el churn real, el acierto real cae por debajo del holdout, la CLI construye el dataset desde el mismo SQLite, el trainer entrena con `--data`, gana al campeón, el metadata registra la procedencia de las etiquetas y, tras el reload, el modelo nuevo acierta más y el drift desaparece.
- **Drift**: PSI en binarias desbalanceadas, referencias constantes, baja cardinalidad, nulos, categóricas con tipos mezclados, *prediction drift*, estructura del informe y **propiedades con Hypothesis** (no negatividad, simetría, cobertura de los bins).
- **Configuración**: prefijo `CHURN_`, umbrales coherentes, guardia de token en producción (también del de etiquetas, que debe ser distinto del de administración), `CHURN_LABELS_MIN_ROWS >= 500`, retención por días.
- **API**: coherencia de riesgo, validación 422, batch consistente con individual, 503 sin modelo, drift 409→200 y detectado, reload y rollback con token, versión nueva, artefactos ausentes, corruptos o incompatibles, retención, API key, métricas, `X-Request-ID` y log de acceso; `prediction_id` único, `subject_ref` (opcional, no llega al modelo), `POST /labels` (token de administración o **token propio**, 401 antes de validar el cuerpo, 422, creación/corrección/desconocidos/rechazadas, `observed_at` futura, corrección tras el desalojo, idempotencia, `observed_at` por defecto, sin modelo, persistencia entre reinicios, aviso de arranque en producción sin token propio) y `/monitoring/performance` (409→200, por versión tras un reload, sin modelo cargado, ventana acotada, una sola clase, métricas Prometheus).
- **Contratos**: el esquema Pydantic de la API coincide con los rangos y categorías del validador.
- **Scripts**: `generate_data.py` desde cualquier directorio y con `--drift-shift`; `retrain-if-drift.sh` con dobles de `curl` y `docker`: sin drift, 409, API caída, **drift sin `CHURN_TRAIN_DATA` y con etiquetas suficientes (construye el dataset en el volumen y entrena con él) o insuficientes (avisa, no reentrena)**, sin etiquetas nuevas desde el modelo en servicio (misma huella: no reentrena), otra ejecución en curso (`flock`), dataset de etiquetas inválido o imposible de construir, fichero inexistente, montaje del fichero y `--data`, retador no promovido (sin reload), gate rechazado y avisos por webhook; `backup.sh` con un doble de `docker`: instantánea de `drift.sqlite` antes de archivar y sin los ficheros vivos del WAL, sin `drift.sqlite`, instantánea fallida (aborta), volumen inexistente y retención.

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
5. Operación: `deploy/scripts/retrain-if-drift.sh` (cron) reentrena solo si hay drift y hay etiquetas suficientes, y `deploy/scripts/backup.sh` (cron) archiva los volúmenes. Rollback de **imagen**: `TAG=$(cat .previous_tag) ./deploy/scripts/deploy.sh`; rollback de **modelo**: `POST /model/rollback`.

Todo el stack de producción, incluido `deploy.sh`, se puede ensayar en local con un registro Docker local (`IMAGE_REGISTRY`, `CADDY_*_PORT` en `.env`); la guía explica cómo.

La API y MLflow solo son accesibles **dentro de la VPN** (Caddy con TLS; MLflow bajo `/mlflow/`).

## 🧠 Decisiones de diseño (nivel senior)

- **Una sola imagen** para entrenar y servir (`trainer` y `api` = misma imagen, distinto comando): elimina divergencias de dependencias entre training y serving, y el lockfile hace el build reproducible.
- **Almacén de modelos versionado con puntero atómico**: publicar es renombrar un directorio y sustituir un fichero de texto; volver atrás es mover el puntero. La API no depende de MLflow para arrancar → MLflow es *plus* de trazabilidad, no punto único de fallo, pero el alias `champion` lo convierte en la fuente de "qué modelo está en producción".
- **Gate de calidad en el pipeline, no solo en los tests**: un reentreno que degrada el AUC no llega nunca al volumen de modelos; el CI además falla si un cambio de código degrada el modelo — el modelo se trata como código.
- **Campeón/retador sobre el holdout de los datos nuevos**: el gate absoluto no basta (un modelo puede superar 0.75 y aun así ser peor que el que está en servicio). Evaluar ambos en el mismo 20 % de los datos nuevos responde a la pregunta que importa: *¿cuál de los dos funciona mejor en el mundo de hoy?* Si el mundo ha cambiado, el campeón pierde ahí y el retador se promueve; si los datos nuevos no aportan nada, el campeón se queda. El retador rechazado se conserva sin promover: un operador puede ponerlo en servicio con `POST /model/rollback {"version": ...}` si discrepa.
- **Reentrenar exige datos etiquetados, y la plataforma lo dice**: la etiqueta real llega semanas después, desde el negocio. Antes, `retrain-if-drift.sh` "reentrenaba" con el generador sintético de siempre — mismo dataset, mismo modelo, misma referencia — y el drift seguía exactamente igual; un test (`test_reentrenar_con_el_mismo_generador_no_cambia_nada`) lo deja escrito. Ahora el script reentrena solo con etiquetas reales (las de `POST /labels` o un fichero del operador), avisa con claridad si no hay suficientes, y la huella SHA-256 del dataset (en el metadata y en el *sidecar*) evita reentrenar —y promover— con los mismos datos que el modelo en servicio.
- **El bucle de etiquetas se cierra con el `prediction_id`; el cliente, si se quiere, con una clave opaca**: la plataforma no conoce a los clientes, conoce predicciones; el CRM guarda el `prediction_id` junto a su propio identificador y lo devuelve con el churn observado. Así una etiqueta se une sin ambigüedad a las features y a la probabilidad exactas que se sirvieron (y a la versión que las produjo). Lo que el `prediction_id` no dice es que dos predicciones son del **mismo cliente**: puntuado cada mes, sus casi duplicados caían a ambos lados del holdout y un retador que había memorizado clientes (AUC de holdout 0.997, real 0.89) desbancaba a un campeón mejor. El `subject_ref` opcional (un hash del id interno, nunca datos personales) lo resuelve sin que la plataforma conozca a nadie: una etiqueta por sujeto y horizonte en el dataset y holdout separado por sujeto. Al etiquetar se guarda una **copia del ejemplo** en una tabla aparte: sobrevive al desalojo de la ventana de predicciones, se puede corregir después y es, literalmente, el dataset de reentreno. La ingesta es **idempotente por `prediction_id`** (reenviar un lote no duplica, corregir sobrescribe) y los ids desconocidos o rechazados no invalidan el lote, para que un *job* del CRM pueda reintentarse sin cuidado; ese *job* usa su propio token (`CHURN_LABELS_TOKEN`), no el que recarga modelos.
- **Un holdout que no cambia entre reentrenos**: el dataset de etiquetas es acumulado, así que un split aleatorio (que depende del tamaño) metía en el holdout del siguiente reentreno filas con las que el campeón ya había entrenado; el campeón jugaba con ventaja y se rechazaban retadores mejores (en 24 escenarios de dos reentrenos, 1 decisión correcta antes, 23 ahora). El holdout se elige por un hash con sal fija del grupo (sujeto o fila): nadie cambia de lado nunca. Como defensa adicional, el campeón entrenado con etiquetas solo se mide con las predichas después de las suyas (`predicted_to`), y sin al menos 80 grupos con ambas clases no se le sustituye: la evidencia se cuenta en clientes, no en filas. Con pocas bajas nuevas la comparación sigue siendo ruidosa (el AUC se apoya en los pares baja/no baja): el margen y el volumen de etiquetas son los mandos.
- **El acierto real se mide con las mismas métricas y umbral que el holdout**: `/monitoring/performance` y el entrenamiento comparten `classification_metrics`, así que la cifra de producción y la de `metadata.json` son comparables; el desglose por versión evita atribuir al modelo en servicio los aciertos (o fallos) de la versión anterior.
- **Procedencia de los datos en el metadata, sin cambiar la interfaz del trainer ni romper el rollback**: el dataset de etiquetas viaja con un *sidecar* `.meta.json` (filas, ventana, versiones, huella); `--data` sigue siendo un fichero, pero si la huella del sidecar coincide con el fichero cargado, el modelo registra su ventana en el bloque `data_source.labels`. `kind` sigue siendo `file` a propósito: la imagen anterior valida `kind` con un `Literal` cerrado, así que un valor nuevo rompería su `/model/info` tras un rollback de imagen, mientras que un campo nuevo simplemente se ignora. Si alguien edita el fichero después, la huella no coincide y la procedencia se descarta con un aviso; si la huella es la del modelo en servicio, no hay nada que reentrenar.
- **Copias de seguridad consistentes de un SQLite vivo**: `drift.sqlite` está en modo WAL y se escribe en cada predicción; un `tar` del volumen copia la base y su `-wal` en instantes distintos (en una prueba con escrituras continuas, 15 de 15 copias corruptas). `backup.sh` usa la API de backup de SQLite para una instantánea consistente sin parar la API (5 de 5 íntegras) y la guarda aparte del `.tgz`.
- **Un modelo solo se sirve si este runtime puede cargarlo**: versiones de scikit-learn y features comprobadas antes de servir; el fallo silencioso tras reconstruir la imagen era el riesgo más caro que quedaba.
- **Un reload o rollback fallido nunca degrada el servicio**: se conserva el último modelo bueno conocido y el error se devuelve al operador.
- **Drift con PSI/KS implementado y auditable** en lugar de una dependencia pesada, con *binning* robusto para variables binarias/enteras (donde los cuantiles clásicos enmascaran el cambio), *prediction drift* y propiedades verificadas con Hypothesis; migrar a Evidently es trivial si el equipo lo prefiere.
- **Predicciones en SQLite dentro del volumen**: cero infraestructura adicional, compartido entre workers y persistente; el contrato del endpoint no cambia si mañana el sink es un warehouse.
- **Fuente única de verdad del dominio de las features**: los rangos y categorías viven en el validador y el contrato Pydantic se deriva de ellos (un test de contrato lo garantiza).
- **Observabilidad de serie**: métricas por plantilla de ruta (cardinalidad acotada), `X-Request-ID` de extremo a extremo y un único formato JSON para la aplicación y uvicorn.

## 🗺️ Posibles extensiones

Reentrenamiento por drift, promoción campeón/retador y bucle de etiquetas ya cubiertos; quedan la calibración de probabilidades y el ajuste de umbrales con curvas precision-recall (ahora con etiquetas reales sobre las que calcularlas), alertas automáticas por caída del acierto real (hoy `/monitoring/performance` y `churn_performance_*` lo exponen, pero no disparan nada), feature store, A/B de modelos (shadow deployment), explicabilidad SHAP por predicción y export a ONNX para latencias < 5 ms.

---
*Proyecto de portfolio orientado a roles **ML Engineer / MLOps Engineer**. Licencia MIT.*
