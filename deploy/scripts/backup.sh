#!/usr/bin/env bash
# =============================================================================
# Copia de seguridad de los volumenes de produccion (modelos + MLflow) con retencion.
# Pensado para cron en el VPS, p. ej.:
#   0 3 * * * /opt/mlops-churn-platform/deploy/scripts/backup.sh >> /var/log/churn-backup.log 2>&1
# Uso:  BACKUP_DIR=/backup/mlops-churn-platform KEEP_DAYS=14 ./deploy/scripts/backup.sh
#
# drift.sqlite (predicciones y etiquetas) esta en modo WAL y se escribe en cada /predict:
# un `tar` del fichero vivo puede copiar la base y su -wal en instantes distintos (copia
# inconsistente o corrupta, justo lo que no se puede perder: las etiquetas). Por eso se
# copia aparte con la API de backup de SQLite (instantanea consistente sin parar la API,
# en un contenedor de la propia imagen) como drift-<STAMP>.sqlite, y el .tgz del volumen
# excluye los ficheros vivos.
#
# Restaurar el volumen de modelos (con el stack parado):
#   docker run --rm -v mlops-churn-platform_models:/v -v /backup/mlops-churn-platform:/b \
#     alpine:3.20 sh -c 'rm -rf /v/* && tar xzf /b/models-<STAMP>.tgz -C /v \
#       && cp /b/drift-<STAMP>.sqlite /v/drift.sqlite && chown 10001:10001 /v/drift.sqlite'
# (sin drift-<STAMP>.sqlite, la API arranca con un almacen vacio). MLflow: igual con
# mlflow-data-<STAMP>.tgz y sin el paso de drift.sqlite.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi   # COMPOSE_PROJECT_NAME, si esta definido

BACKUP_DIR="${BACKUP_DIR:-/backup/mlops-churn-platform}"
KEEP_DAYS="${KEEP_DAYS:-14}"
PROJECT="${COMPOSE_PROJECT_NAME:-$(basename "$PWD")}"   # prefijo de los volumenes de compose
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
COMPOSE="docker compose -f docker-compose.prod.yml"
SNAPSHOT=".backup-drift.sqlite"   # temporal, dentro del volumen de modelos
# Instantanea con la API de backup de SQLite: lee una vista consistente (incluido lo que aun
# esta en el -wal) sin bloquear a la API, la deja en modo rollback (un unico fichero, sin
# -wal; la API vuelve a activar WAL al abrirla) y la publica con un rename atomico.
SNAPSHOT_PY='
import os, sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
if not os.path.exists(src):
    print("sin " + src + ": nada que copiar")
    sys.exit(0)
tmp = dst + ".tmp"
source, target = sqlite3.connect(src, timeout=60), sqlite3.connect(tmp)
source.backup(target)
target.execute("PRAGMA journal_mode=DELETE")
target.close()
source.close()
os.replace(tmp, dst)
'

mkdir -p "$BACKUP_DIR"
for volume in models mlflow-data; do
  name="${PROJECT}_${volume}"
  if ! docker volume inspect "$name" >/dev/null 2>&1; then
    echo "!! El volumen ${name} no existe; se omite"
    continue
  fi
  file="${volume}-${STAMP}.tgz"
  excludes=()
  if [ "$volume" = models ]; then
    $COMPOSE --profile train run --rm --no-deps -T trainer \
      python -c "$SNAPSHOT_PY" /models/drift.sqlite "/models/${SNAPSHOT}"
    excludes=(--exclude ./drift.sqlite --exclude ./drift.sqlite-wal --exclude ./drift.sqlite-shm
              --exclude "./${SNAPSHOT}" --exclude "./${SNAPSHOT}.tmp")
  fi
  docker run --rm -v "${name}:/v" -v "${BACKUP_DIR}:/b" alpine:3.20 \
    tar czf "/b/${file}" -C /v "${excludes[@]}" .
  echo ">> ${BACKUP_DIR}/${file} ($(du -h "${BACKUP_DIR}/${file}" | cut -f1))"
  if [ "$volume" = models ]; then
    db="drift-${STAMP}.sqlite"
    docker run --rm -v "${name}:/v" -v "${BACKUP_DIR}:/b" alpine:3.20 \
      sh -c 'if [ -f "/v/$1" ]; then mv "/v/$1" "/b/$2"; fi' sh "$SNAPSHOT" "$db"
    if [ -f "${BACKUP_DIR}/${db}" ]; then
      echo ">> ${BACKUP_DIR}/${db} ($(du -h "${BACKUP_DIR}/${db}" | cut -f1), instantanea consistente)"
    fi
  fi
done

find "$BACKUP_DIR" \( -name '*.tgz' -o -name 'drift-*.sqlite' \) -mtime +"$KEEP_DAYS" -delete
echo ">> Copias anteriores a ${KEEP_DAYS} dias eliminadas"
