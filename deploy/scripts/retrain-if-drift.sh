#!/usr/bin/env bash
# =============================================================================
# Reentrenamiento disparado por drift. Consulta /monitoring/drift y, si el informe
# detecta cambio, reentrena con el perfil "train" sobre DATOS ETIQUETADOS y recarga
# la API si el modelo nuevo se ha promovido.
#
# De donde salen los datos etiquetados (en este orden):
#   1. CHURN_TRAIN_DATA: ruta en el host a un .csv/.parquet con las features y la
#      columna `churn` (export del CRM/warehouse). Se monta en el trainer tal cual.
#   2. Sin CHURN_TRAIN_DATA: las etiquetas recibidas por la API (POST /labels). El
#      trainer construye /models/datasets/labels.parquet dentro del volumen con
#      `python -m churn.data.labels` (las predicciones etiquetadas, una por sujeto y
#      horizonte, con su huella y ventana temporal en un sidecar que acaba en el metadata
#      del modelo). Si hay menos de CHURN_LABELS_MIN_ROWS ejemplos, avisa y no reentrena:
#      reentrenar con el generador sintetico de siempre produciria exactamente el mismo
#      modelo. Si la huella del dataset es la del modelo en servicio (no han llegado
#      etiquetas nuevas desde su entrenamiento), tampoco reentrena.
#
# Que hace el trainer con esos datos: valida, entrena, aplica el gate de calidad
# (CHURN_MIN_ROC_AUC) y compara con el modelo en servicio sobre el holdout de los datos
# nuevos, solo con las etiquetas posteriores a las suyas (CHURN_PROMOTION_MARGIN). Solo si
# gana pasa a `current` (codigo 0); si pierde queda guardado sin promover (codigo 3) y la
# API no se recarga.
#
# Una sola ejecucion a la vez: un cerrojo (flock sobre .retrain.lock en la raiz del repo)
# evita que el cron y un FORCE=1 manual construyan el dataset o entrenen a la vez; la
# segunda ejecucion avisa y termina sin hacer nada.
#
# Pensado para cron en el VPS, p. ej. cada lunes a las 04:00:
#   0 4 * * 1 /opt/mlops-churn-platform/deploy/scripts/retrain-if-drift.sh >> /var/log/churn-retrain.log 2>&1
# Variables (ademas de las de .env): CHURN_TRAIN_DATA (fichero etiquetado; sin ella se
# usan las etiquetas de la API), CHURN_LABELS_MIN_ROWS (minimo de etiquetas, defecto 500;
# llega al trainer por el compose), API_URL (http://127.0.0.1:8010), CHURN_API_KEY (si la
# API la exige), FORCE=1 (reentrena sin consultar el drift), WEBHOOK_URL (aviso opcional
# en JSON {"text": ...}, p. ej. Slack/Mattermost).
# Codigos de salida: 0 ok, nada que hacer (sin drift, sin etiquetas nuevas, otra
# ejecucion en curso) o retador no promovido | 1 no se pudo consultar el drift | 2
# reentreno rechazado (datos invalidos o gate) o dataset de etiquetas no construible | 3
# sin datos etiquetados (CHURN_TRAIN_DATA no existe o etiquetas insuficientes)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi

API_URL="${API_URL:-http://127.0.0.1:8010}"
COMPOSE="docker compose -f docker-compose.prod.yml"
LABELS_DATASET="/models/datasets/labels.parquet"   # dentro del volumen de modelos
LOCK_FILE="${RETRAIN_LOCK_FILE:-.retrain.lock}"
auth=()
if [ -n "${CHURN_API_KEY:-}" ]; then auth=(-H "X-API-Key: ${CHURN_API_KEY}"); fi

notify() {
  echo ">> $1"
  if [ -n "${WEBHOOK_URL:-}" ]; then
    curl -fsS -X POST -H 'Content-Type: application/json' \
      -d "{\"text\": \"[churn] $1\"}" "$WEBHOOK_URL" >/dev/null || echo "!! Aviso no enviado"
  fi
}

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    notify "Ya hay un reentrenamiento en curso (${LOCK_FILE}); se omite esta ejecucion"
    exit 0
  fi
else
  echo "!! flock no disponible: no se protege contra ejecuciones simultaneas"
fi

if [ "${FORCE:-0}" = "1" ]; then
  reason="reentreno forzado (FORCE=1)"
else
  report="$(mktemp)"
  trap 'rm -f "$report"' EXIT
  code="$(curl -s -o "$report" -w '%{http_code}' --max-time 30 "${auth[@]}" \
    "${API_URL}/monitoring/drift" || true)"
  case "$code" in
    200) ;;
    409) echo ">> Datos insuficientes para evaluar drift todavia"; exit 0 ;;
    *) notify "No se pudo consultar el drift (HTTP ${code:-sin respuesta})"; exit 1 ;;
  esac
  summary="$(python3 - "$report" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
flag = "1" if d["drift_detected"] else "0"
features = ",".join(d["drifted_features"]) or "predicciones"
print(f"{flag} {features}")
PY
)"
  if [ "${summary%% *}" != "1" ]; then
    echo ">> Sin drift; no se reentrena"
    exit 0
  fi
  reason="drift detectado (${summary#* })"
fi

# Datos etiquetados: fichero del operador (montado) o etiquetas de la API (en el volumen).
mounts=()
if [ -n "${CHURN_TRAIN_DATA:-}" ]; then
  if [ ! -f "$CHURN_TRAIN_DATA" ]; then
    notify "${reason}, pero CHURN_TRAIN_DATA=${CHURN_TRAIN_DATA} no existe; no se hace nada"
    exit 3
  fi
  data_file="$(basename "$CHURN_TRAIN_DATA")"
  data_host="$(cd "$(dirname "$CHURN_TRAIN_DATA")" && pwd)/${data_file}"
  data_container="/data/${data_file}"
  mounts=(-v "${data_host}:${data_container}:ro")
else
  echo ">> CHURN_TRAIN_DATA no esta definida: se construye el dataset con las etiquetas recibidas por la API"
  set +e
  # El resumen JSON va por stdout (los logs del trainer, por stderr); -T evita que un TTY lo altere.
  build="$($COMPOSE --profile train run --rm --no-deps -T trainer \
    python -m churn.data.labels --model-dir /models --out "$LABELS_DATASET" | tail -n 1)"
  status=$?
  set -e
  detail="$(python3 - "$build" <<'PY' || true
import json, sys
try:
    d = json.loads(sys.argv[1])
except ValueError:
    sys.exit(1)
if d.get("status") == "ok":
    print(f"{d['rows']} filas etiquetadas (predicciones del {d['predicted_from']} al {d['predicted_to']})")
elif d.get("status") == "insufficient":
    total = d.get("labels_total", d["rows"])
    print(f"solo hay {d['rows']} ejemplos etiquetados ({total} etiquetas) de un minimo de {d['min_rows']}")
else:
    print(d.get("error", "sin detalle"))
PY
)"
  case "$status" in
    0) echo ">> Dataset de etiquetas construido en ${LABELS_DATASET}: ${detail}" ;;
    3) notify "${reason}, pero ${detail:-no hay etiquetas suficientes}: define CHURN_TRAIN_DATA o espera a recibir mas etiquetas (POST /labels); no se hace nada"; exit 3 ;;
    *) notify "${reason}, pero no se pudo construir el dataset de etiquetas (codigo ${status}): ${detail:-sin detalle}; no se hace nada"; exit 2 ;;
  esac
  # Sin etiquetas nuevas desde el modelo en servicio, el dataset es el mismo con el que se
  # entreno (misma huella): reentrenar no puede cambiar nada.
  fingerprint="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("fingerprint", ""))' "$build" 2>/dev/null || true)"
  serving="$(curl -s --max-time 30 "${auth[@]}" "${API_URL}/model/info" | python3 -c 'import json,sys; print((json.load(sys.stdin).get("data_source") or {}).get("fingerprint", ""))' 2>/dev/null || true)"
  if [ -n "$fingerprint" ] && [ "$fingerprint" = "$serving" ]; then
    notify "${reason}, pero no hay etiquetas nuevas desde el entrenamiento del modelo en servicio (huella ${fingerprint:0:12}): no se reentrena"
    exit 0
  fi
  data_file="etiquetas de la API (${detail%% (*})"
  data_container="$LABELS_DATASET"
fi

notify "Reentrenando con ${data_file}: ${reason}"
set +e
$COMPOSE --profile train run --rm "${mounts[@]}" trainer \
  python -m churn.training.train --data "$data_container" --model-dir /models
status=$?
set -e
case "$status" in
  0) ;;
  3) notify "Modelo nuevo entrenado pero NO promovido: no supera al modelo en servicio sobre los datos nuevos; se mantiene el actual"; exit 0 ;;
  *) notify "Reentrenamiento rechazado (datos invalidos o gate de calidad) o fallido (codigo ${status}); se mantiene el modelo actual"; exit 2 ;;
esac

curl -fsS -X POST "${API_URL}/model/reload" \
  -H "X-Admin-Token: ${CHURN_ADMIN_TOKEN:?definir CHURN_ADMIN_TOKEN en .env}" >/dev/null
version="$(curl -s "${API_URL}/health" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("model_version"))')"
notify "Modelo recargado: version ${version} (${reason}, datos ${data_file})"
