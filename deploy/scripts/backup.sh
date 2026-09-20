#!/usr/bin/env bash
# =============================================================================
# Copia de seguridad de los volumenes de produccion (modelos + MLflow) con retencion.
# Pensado para cron en el VPS, p. ej.:
#   0 3 * * * /opt/mlops-churn-platform/deploy/scripts/backup.sh >> /var/log/churn-backup.log 2>&1
# Uso:  BACKUP_DIR=/backup/mlops-churn-platform KEEP_DAYS=14 ./deploy/scripts/backup.sh
# Restaurar un volumen (con el stack parado):
#   docker run --rm -v mlops-churn-platform_models:/v -v /backup/mlops-churn-platform:/b \
#     alpine:3.20 sh -c 'rm -rf /v/* && tar xzf /b/models-<STAMP>.tgz -C /v'
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi   # COMPOSE_PROJECT_NAME, si esta definido

BACKUP_DIR="${BACKUP_DIR:-/backup/mlops-churn-platform}"
KEEP_DAYS="${KEEP_DAYS:-14}"
PROJECT="${COMPOSE_PROJECT_NAME:-$(basename "$PWD")}"   # prefijo de los volumenes de compose
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$BACKUP_DIR"
for volume in models mlflow-data; do
  name="${PROJECT}_${volume}"
  if ! docker volume inspect "$name" >/dev/null 2>&1; then
    echo "!! El volumen ${name} no existe; se omite"
    continue
  fi
  file="${volume}-${STAMP}.tgz"
  docker run --rm -v "${name}:/v:ro" -v "${BACKUP_DIR}:/b" alpine:3.20 \
    tar czf "/b/${file}" -C /v .
  echo ">> ${BACKUP_DIR}/${file} ($(du -h "${BACKUP_DIR}/${file}" | cut -f1))"
done

find "$BACKUP_DIR" -name '*.tgz' -mtime +"$KEEP_DAYS" -delete
echo ">> Copias anteriores a ${KEEP_DAYS} dias eliminadas"
