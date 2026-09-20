#!/usr/bin/env bash
# =============================================================================
# Script de despliegue ejecutado EN el VPS por GitHub Actions (o a mano).
# Uso:  TAG=v1.0.0 ./deploy/scripts/deploy.sh
# Estrategia: pull -> up -d -> espera a que /health responda 200
#             -> si la API arranca sin modelo (503), entrenamiento inicial + reload
#             -> verificacion -> limpieza.
# Si la API no llega a responder 200, rollback automatico al tag en servicio.
# Ficheros de estado:  .current_tag  (tag en servicio)  .previous_tag (para rollback)
# Rollback manual:     TAG=$(cat .previous_tag) ./deploy/scripts/deploy.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

REQUESTED_TAG="${TAG:-}"            # lo que pide el invocador (CI o a mano)...
if [ -f .env ]; then set -a; . ./.env; set +a; fi
TAG="${REQUESTED_TAG:-${TAG:-latest}}"   # ...tiene prioridad sobre el TAG de .env
TAG="${TAG#v}"                      # las imagenes GHCR se etiquetan sin la "v" (semver)
export TAG

COMPOSE="docker compose -f docker-compose.prod.yml"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8010/health}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-120}"   # segundos de espera maxima a la API

CURRENT="$(cat .current_tag 2>/dev/null || cat .last_tag 2>/dev/null || echo '')"
echo ">> Desplegando tag: ${TAG} (en servicio: ${CURRENT:-ninguno})"

# 0 = /health responde 200 | 2 = la API responde pero sin modelo (503) | 1 = sin respuesta
wait_for_api() {
  local deadline code
  deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$HEALTH_URL" || true)"
    case "$code" in
      200) return 0 ;;
      503) return 2 ;;
    esac
    sleep 3
  done
  return 1
}

rollback() {
  if [ -n "$CURRENT" ] && [ "$CURRENT" != "$TAG" ]; then
    echo "!! Rollback a ${CURRENT}"
    TAG="$CURRENT" $COMPOSE up -d --remove-orphans
  fi
  exit 1
}

$COMPOSE pull
$COMPOSE up -d --remove-orphans

echo ">> Esperando a la API en ${HEALTH_URL} (max ${HEALTH_TIMEOUT}s)..."
set +e; wait_for_api; status=$?; set -e

if [ "$status" -eq 2 ]; then
  echo ">> La API arranca pero no hay modelo: entrenamiento inicial (perfil train)"
  $COMPOSE --profile train run --rm trainer
  curl -fsS -X POST "${HEALTH_URL%/health}/model/reload" \
    -H "X-Admin-Token: ${CHURN_ADMIN_TOKEN:?definir CHURN_ADMIN_TOKEN en .env}" > /dev/null
  set +e; wait_for_api; status=$?; set -e
fi

if [ "$status" -ne 0 ]; then
  echo "!! La API no responde 200 en ${HEALTH_URL} (estado: ${status})"
  $COMPOSE ps || true
  $COMPOSE logs --tail 50 api || true
  rollback
fi

if [ -n "$CURRENT" ] && [ "$CURRENT" != "$TAG" ]; then echo "$CURRENT" > .previous_tag; fi
echo "$TAG" > .current_tag
rm -f .last_tag
docker image prune -f > /dev/null
echo ">> Despliegue de ${TAG} completado. Rollback: TAG=\$(cat .previous_tag) $0"
