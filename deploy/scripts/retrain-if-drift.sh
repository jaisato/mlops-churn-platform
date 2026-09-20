#!/usr/bin/env bash
# =============================================================================
# Reentrenamiento disparado por drift. Consulta /monitoring/drift y, si el informe
# detecta cambio, reentrena con el perfil "train" y recarga la API. Si el modelo
# nuevo no supera el gate de calidad, el trainer falla y el modelo en servicio no
# se toca. Pensado para cron en el VPS, p. ej. cada lunes a las 04:00:
#   0 4 * * 1 /opt/mlops-churn-platform/deploy/scripts/retrain-if-drift.sh >> /var/log/churn-retrain.log 2>&1
# Variables (ademas de las de .env): API_URL (http://127.0.0.1:8010), CHURN_API_KEY
# (si la API la exige), FORCE=1 (reentrena aunque no haya drift), WEBHOOK_URL
# (aviso opcional en JSON {"text": ...}, p. ej. Slack/Mattermost).
# Codigos de salida: 0 ok o nada que hacer | 1 no se pudo consultar | 2 reentreno rechazado
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
    curl -fsS -X POST -H 'Content-Type: application/json' \
      -d "{\"text\": \"[churn] $1\"}" "$WEBHOOK_URL" >/dev/null || echo "!! Aviso no enviado"
  fi
}

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
flag="${summary%% *}"
features="${summary#* }"

if [ "$flag" != "1" ] && [ "${FORCE:-0}" != "1" ]; then
  echo ">> Sin drift; no se reentrena"
  exit 0
fi

notify "Drift detectado (${features}); reentrenando"
if ! $COMPOSE --profile train run --rm trainer; then
  notify "Reentrenamiento rechazado (gate de calidad) o fallido; se mantiene el modelo actual"
  exit 2
fi
curl -fsS -X POST "${API_URL}/model/reload" \
  -H "X-Admin-Token: ${CHURN_ADMIN_TOKEN:?definir CHURN_ADMIN_TOKEN en .env}" >/dev/null
version="$(curl -s "${API_URL}/health" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("model_version"))')"
notify "Modelo recargado tras el drift: version ${version}"
