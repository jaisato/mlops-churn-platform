#!/usr/bin/env bash
# =============================================================================
# Reentrenamiento disparado por drift. Consulta /monitoring/drift y, si el informe
# detecta cambio, reentrena con el perfil "train" sobre DATOS ETIQUETADOS NUEVOS y
# recarga la API si el modelo nuevo se ha promovido.
#
# Por que hacen falta datos etiquetados: la API guarda las features y la probabilidad
# de cada prediccion (drift.sqlite), pero NO la etiqueta real (si el cliente se dio de
# baja o no), que llega semanas despues desde el negocio. Sin etiquetas no se puede
# reentrenar, y reentrenar con el generador sintetico de siempre produce exactamente el
# mismo modelo: no corrige nada. Por eso este script exige CHURN_TRAIN_DATA (ruta en el
# host a un .csv/.parquet con las features y la columna `churn`) y falla con claridad si
# no esta definida, en lugar de reentrenar en silencio con datos que no han cambiado.
#
# Que hace el trainer con esos datos: valida, entrena, aplica el gate de calidad
# (CHURN_MIN_ROC_AUC) y compara con el modelo en servicio sobre el holdout de los datos
# nuevos (CHURN_PROMOTION_MARGIN). Solo si gana pasa a `current` (codigo 0); si pierde
# queda guardado sin promover (codigo 3) y la API no se recarga.
#
# Pensado para cron en el VPS, p. ej. cada lunes a las 04:00:
#   0 4 * * 1 /opt/mlops-churn-platform/deploy/scripts/retrain-if-drift.sh >> /var/log/churn-retrain.log 2>&1
# Variables (ademas de las de .env): CHURN_TRAIN_DATA (obligatoria para reentrenar),
# API_URL (http://127.0.0.1:8010), CHURN_API_KEY (si la API la exige), FORCE=1
# (reentrena sin consultar el drift), WEBHOOK_URL (aviso opcional en JSON {"text": ...},
# p. ej. Slack/Mattermost).
# Codigos de salida: 0 ok, nada que hacer o retador no promovido | 1 no se pudo
# consultar el drift | 2 reentreno rechazado (datos invalidos o gate) | 3 falta
# CHURN_TRAIN_DATA o el fichero no existe
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi

API_URL="${API_URL:-http://127.0.0.1:8010}"
COMPOSE="docker compose -f docker-compose.prod.yml"
auth=()
if [ -n "${CHURN_API_KEY:-}" ]; then auth=(-H "X-API-Key: ${CHURN_API_KEY}"); fi

notify() {
  echo ">> $1"
  if [ -n "${WEBHOOK_URL:-}" ]; then
    # JSON construido con json.dumps: el mensaje lleva nombres de fichero y de features, y
    # una comilla o una barra invertida rompian el JSON concatenado a mano.
    local payload
    payload="$(python3 -c 'import json, sys; print(json.dumps({"text": "[churn] " + sys.argv[1]}))' "$1")" \
      && curl -fsS -X POST -H 'Content-Type: application/json' \
        -d "$payload" "$WEBHOOK_URL" >/dev/null \
      || echo "!! Aviso no enviado"
  fi
}

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

# Sin datos etiquetados nuevos no hay reentrenamiento posible: avisar y salir.
if [ -z "${CHURN_TRAIN_DATA:-}" ]; then
  notify "${reason}, pero CHURN_TRAIN_DATA no esta definida: hace falta un fichero etiquetado (.csv/.parquet) para reentrenar; no se hace nada"
  exit 3
fi
if [ ! -f "$CHURN_TRAIN_DATA" ]; then
  notify "${reason}, pero CHURN_TRAIN_DATA=${CHURN_TRAIN_DATA} no existe; no se hace nada"
  exit 3
fi
data_file="$(basename "$CHURN_TRAIN_DATA")"
data_host="$(cd "$(dirname "$CHURN_TRAIN_DATA")" && pwd)/${data_file}"

notify "Reentrenando con ${data_file}: ${reason}"
set +e
$COMPOSE --profile train run --rm -v "${data_host}:/data/${data_file}:ro" trainer \
  python -m churn.training.train --data "/data/${data_file}" --model-dir /models
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
