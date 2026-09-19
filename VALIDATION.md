# Validación de correcciones

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
