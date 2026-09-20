# Validación de correcciones

## Ronda 3 (19/20-09-2026): verificación real, mejoras de prioridad alta, almacén versionado, observabilidad

Resultado: `make check` limpio (ruff + formato, mypy, **227 tests en 13 módulos, 100 % de
cobertura de líneas y ramas**); `shellcheck`, `actionlint` y `hadolint` sin hallazgos; CI en
GitHub en verde en la matriz 3.11/3.12/3.13. Repositorio: https://github.com/jaisato/mlops-churn-platform.

Verificado con el stack levantado (no solo tests):

| Escenario | Cómo | Resultado |
|---|---|---|
| Stack local completo | `docker compose up --build`: MLflow con healthcheck → trainer → API | `/health` 200 en el primer intento; run en MLflow y alias `champion` → v1; `/predict`, drift 409 → 251 predicciones → 200 con PSI de predicciones; `/metrics`; `X-Request-ID`; log de acceso JSON |
| Reentreno, reload y rollback | `docker compose run trainer` + `POST /model/reload` + `POST /model/rollback` | Segunda versión publicada (alias `champion` → v2), reload a la nueva, rollback a la anterior; el volumen muestra `current`, `versions/<v>/` y `drift.sqlite` |
| Stack de producción en local | Registro Docker local (`IMAGE_REGISTRY=localhost:5001`), `.env` con `TAG=latest` y `TAG=1.0.0 ./deploy/scripts/deploy.sh` | Volumen vacío → 503 → entrenamiento inicial → reload → 200 en 15 s; `.current_tag=1.0.0` (el `TAG` pedido prevalece sobre el de `.env`) |
| Caddy con TLS interno | `curl -k https://127.0.0.1:8443/...` | **Bug heredado**: con el sitio `:443` los clientes por IP (sin SNI) recibían `tlsv1 alert internal error`. Corregido con el sitio en la IP de la VPN y `default_sni`. Tras la corrección: API 200 vía proxy, certificado con SAN de la IP, `/mlflow` → 301 → `/mlflow/`, UI de MLflow (index, assets JS, `ajax-api`) 200 tras `strip_prefix`, HTTP → HTTPS 301 |
| Rollback automático de imagen | Imagen rota etiquetada `1.0.1` + `TAG=1.0.1 ./deploy/scripts/deploy.sh` | La API entra en bucle de reinicio, `deploy.sh` agota la espera, vuelve a `1.0.0` y termina con código 1; `/health` 200 con la imagen anterior |
| Backup de volúmenes | `deploy/scripts/backup.sh` | **Bug propio**: no leía `.env` y buscaba los volúmenes con el nombre de proyecto equivocado. Corregido: dos `.tgz` (models, mlflow-data) con retención |
| Reentrenamiento por drift | `deploy/scripts/retrain-if-drift.sh` sin datos, con `FORCE=1` y con `CHURN_MIN_ROC_AUC=0.999` | **Bug propio**: `FORCE=1` se ignoraba si no había datos de drift. Corregido: 409 → salida 0; forzado → nueva versión recargada y alias `champion` actualizado; gate rechazado → salida 2 y modelo en servicio intacto. Requirió pasar `CHURN_MIN_ROC_AUC` al trainer en el compose de producción |
| Healthcheck de MLflow | `docker run` de la imagen oficial 2.22 con el comando del compose | `healthy` |

Mejoras de esta ronda (detalle en `CHANGELOG.md`): almacén de modelos versionado con
rollback y retención, predicciones en SQLite, compatibilidad de artefactos (runtime y
features), alias `champion` en MLflow, `/health/live`, `/metrics`, `X-Request-ID` y logs
JSON unificados, API key opcional, lockfiles con uv, Python 3.11+, scripts de backup y
reentreno, endurecimiento del VPS, huella SSH fijada, CI ampliado y tests de propiedades.

Pendiente de verificar en el VPS real: el túnel WireGuard, el workflow `deploy.yml` con
los secretos, y `https://10.8.0.1/mlflow/` desde un navegador dentro de la VPN.

## Ronda 2 (14-09-2026): bugs, robustez operativa y cobertura

Resultado: `ruff check .` limpio; `pytest --cov=churn` → **148 tests, 100 % de cobertura
de líneas y ramas** en ~18 s sin servicios externos. `docker compose config` valida los
dos ficheros compose (el de producción falla, como debe, si falta `VPN_BIND_IP`);
`caddy validate` acepta el Caddyfile.

Bugs corregidos (todos reproducidos antes de tocar código):

| Área | Bug | Corrección |
|---|---|---|
| Drift | `psi_numeric` devolvía **0.0** para binarias desbalanceadas (p. ej. `is_fiber` 70/30 → 30/70) y para referencias constantes; en enteros de baja cardinalidad 3 y 4 compartían bin | Bins por valor cuando hay pocos valores distintos, bin estrecho para constantes, cuantiles solo para continuas |
| Drift | `psi_categorical` lanzaba `TypeError` con tipos mezclados | Normalización a `str` + `value_counts` |
| Validación | Un NaN generaba tres errores (nulos, fuera de rango, categoría `nan`) | Rangos y categorías se evalúan sin nulos |
| Entrenamiento | `model_version` colisionaba en el mismo segundo | Sufijo aleatorio de 6 hex |
| Entrenamiento | Un fallo de MLflow tumbaba el trainer tras guardar artefactos (y en `docker compose` el trainer arrancaba antes que MLflow) | Tracking tolerante a fallos + healthcheck de MLflow y `service_healthy` |
| API | `/model/reload` con artefactos corruptos devolvía un 500 con traza; arranque con artefactos corruptos hacía crash | 404/500 controlados conservando el modelo anterior; arranque en 503 recuperable con reload |
| API | Comparación del token vulnerable a *timing* | `secrets.compare_digest` |
| Config | Token por defecto aceptado en producción; umbrales de riesgo sin validar | Guardia en `environment=production`; `risk_medium < risk_high`; `drift_min_rows <= buffer` |
| Registro | `exists()` solo miraba `model.joblib`; escritura no atómica | Juego completo de artefactos + staging y `os.replace` |
| Release | `latest` nunca se publicaba en tags (`is_default_branch` es falso en un tag); input `v1.0.0` no coincidía con la etiqueta `1.0.0` | `enable` explícito para tags; se elimina la `v` |
| Deploy | `.env` (`TAG=latest`) pisaba el `TAG` pasado por CI; el "rollback" documentado redesplegaba la versión actual; la verificación era `grep Up` sobre cualquier servicio | Prioridad al `TAG` del invocador; `.current_tag`/`.previous_tag`; espera real a `/health` con rollback y entrenamiento inicial automático |
| Prod | Caddy podía quedar publicado en `0.0.0.0` (Docker se salta ufw); `/mlflow` sin barra final caía en la API | `VPN_BIND_IP` obligatoria; redirección `/mlflow → /mlflow/` |
| Scripts | `generate_data.py` solo funcionaba desde la raíz del repo | Ruta resuelta respecto al fichero |

Mejoras funcionales: gate de calidad `CHURN_MIN_ROC_AUC` en el propio entrenamiento (un
reentreno malo no sobreescribe el modelo en servicio), *prediction drift* (PSI de las
probabilidades frente a la referencia, solo con puntuaciones de la versión en servicio),
`run_id` de MLflow y gate en `metadata.json` y `/model/info`, buffer de drift configurable,
rangos de la API derivados del validador (fuente única de verdad) y modelos de respuesta
tipados para reload y drift en OpenAPI.

Pendiente de verificar en un entorno real: la interfaz de MLflow tras Caddy con
`strip_prefix` (analizada sobre el código fuente de mlflow 2.x, no probada extremo a
extremo) y el build de la imagen Docker (lo hace el CI).

## Ronda 1 (08-09-2026)

El validador informa errores ante columnas duplicadas, valores numéricos almacenados como
texto, categorías desconocidas de tipos mezclados y etiquetas distintas de 0/1. Validación
sin entrenar: 19 pruebas con
`PYTHONPATH=src python -m pytest --noconftest tests/test_validation.py tests/test_drift.py tests/test_generator.py`.
