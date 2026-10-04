import fcntl
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from churn.data.generator import FEATURE_COLUMNS, TARGET, DriftSpec, generate_dataset

REPO = Path(__file__).resolve().parents[1]
RETRAIN_SCRIPT = REPO / "deploy" / "scripts" / "retrain-if-drift.sh"
BACKUP_SCRIPT = REPO / "deploy" / "scripts" / "backup.sh"


def test_generate_data_funciona_desde_cualquier_cwd(tmp_path):
    out = tmp_path / "salida" / "churn.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "generate_data.py"),
            "--rows",
            "600",
            "--seed",
            "3",
            "--out",
            str(out),
        ],
        cwd=tmp_path,  # distinto del raiz del repo: el script no debe depender del cwd
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "600 filas escritas" in result.stdout
    df = pd.read_csv(out)
    assert len(df) == 600
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET]


def test_generate_data_con_drift_shift(tmp_path):
    out = tmp_path / "drift.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "generate_data.py"),
            "--rows",
            "600",
            "--seed",
            "3",
            "--drift-shift",
            "1.0",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "drift shift=1.0" in result.stdout
    esperado = generate_dataset(600, seed=3, drift=DriftSpec.from_shift(1.0))
    assert pd.read_csv(out)["tenure_months"].tolist() == esperado["tenure_months"].tolist()


# --------------------------------------------------------------------------- retrain-if-drift.sh

DRIFT_YES = '{"drift_detected": true, "drifted_features": ["tenure_months", "monthly_charges"]}'
DRIFT_NO = '{"drift_detected": false, "drifted_features": []}'
DRIFT_PREDICTIONS = '{"drift_detected": true, "drifted_features": []}'

CURL_STUB = """#!/usr/bin/env bash
# Doble de curl: registra la llamada y responde segun la URL (el argumento que empieza por http).
echo "$*" >> "$STUB_LOG_DIR/curl.log"
url=""; out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift ;;
    http*) url="$1" ;;
  esac
  shift
done
case "$url" in
  */monitoring/drift) printf '%s' "$DRIFT_BODY" > "$out"; printf '%s' "${DRIFT_HTTP:-200}" ;;
  */model/info) printf '%s' "${MODEL_INFO_BODY:-}" ;;
  */model/reload) exit 0 ;;
  */health) printf '{"model_version": "v-nueva"}' ;;
  *) exit 22 ;;
esac
"""

DOCKER_STUB = """#!/usr/bin/env bash
# Doble de docker: registra la llamada; al construir el dataset de etiquetas imprime el
# resumen JSON (LABELS_BODY) por stdout como hace `python -m churn.data.labels`.
echo "$*" >> "$STUB_LOG_DIR/docker.log"
case "$*" in
  *churn.data.labels*) printf '%s\\n' "$LABELS_BODY"; exit "${LABELS_EXIT:-0}" ;;
esac
exit "${DOCKER_EXIT:-0}"
"""

LABELS_OK = (
    '{"status": "ok", "path": "/models/datasets/labels.parquet", "rows": 640, '
    '"predicted_from": "2026-09-01T08:00:00+00:00", "predicted_to": "2026-09-28T17:30:00+00:00", '
    '"fingerprint": "abc"}'
)
LABELS_INSUFFICIENT = '{"status": "insufficient", "rows": 120, "min_rows": 500}'
LABELS_ERROR = '{"status": "error", "error": "Tasa de churn sospechosa: 0.950"}'
LABELS_BUILD = (
    "compose -f docker-compose.prod.yml --profile train run --rm --no-deps -T trainer "
    "python -m churn.data.labels --model-dir /models --out /models/datasets/labels.parquet"
)
LABELS_TRAIN = (
    "compose -f docker-compose.prod.yml --profile train run --rm trainer "
    "python -m churn.training.train --data /models/datasets/labels.parquet --model-dir /models"
)


@pytest.fixture()
def retrain(tmp_path):
    """Copia el script a un arbol temporal (sin .env) y lo ejecuta con curl/docker de mentira."""
    root = tmp_path / "repo"
    (root / "deploy" / "scripts").mkdir(parents=True)
    script = root / "deploy" / "scripts" / "retrain-if-drift.sh"
    shutil.copy(RETRAIN_SCRIPT, script)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, body in (("curl", CURL_STUB), ("docker", DOCKER_STUB)):
        stub = stubs / name
        stub.write_text(body)
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    logs = tmp_path / "logs"
    logs.mkdir()

    def _run(
        *,
        drift=DRIFT_YES,
        http="200",
        docker_exit="0",
        labels=LABELS_INSUFFICIENT,
        labels_exit="3",
        **env,
    ) -> subprocess.CompletedProcess:
        full_env = {
            **os.environ,
            "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
            "STUB_LOG_DIR": str(logs),
            "DRIFT_BODY": drift,
            "DRIFT_HTTP": http,
            "DOCKER_EXIT": docker_exit,
            "LABELS_BODY": labels,
            "LABELS_EXIT": labels_exit,
            "CHURN_ADMIN_TOKEN": "token-de-prueba-0123456789",
            **env,
        }
        if "CHURN_TRAIN_DATA" not in env:
            full_env.pop("CHURN_TRAIN_DATA", None)
        return subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            timeout=60,
            env=full_env,
            check=False,
        )

    _run.logs = logs  # type: ignore[attr-defined]
    _run.root = root  # type: ignore[attr-defined]
    return _run


def _log(logs: Path, name: str) -> str:
    path = logs / f"{name}.log"
    return path.read_text() if path.exists() else ""


def test_retrain_sin_drift_no_hace_nada(retrain):
    result = retrain(drift=DRIFT_NO)
    assert result.returncode == 0, result.stderr
    assert "Sin drift" in result.stdout
    assert _log(retrain.logs, "docker") == ""


def test_retrain_sin_datos_suficientes_sale_limpio(retrain):
    result = retrain(http="409")
    assert result.returncode == 0
    assert "insuficientes" in result.stdout


def test_retrain_api_caida_devuelve_1(retrain):
    result = retrain(http="500")
    assert result.returncode == 1
    assert "No se pudo consultar el drift (HTTP 500)" in result.stdout
    assert _log(retrain.logs, "docker") == ""


def test_retrain_con_drift_pero_sin_etiquetas_suficientes_falla_con_claridad(retrain):
    result = retrain()  # sin CHURN_TRAIN_DATA y con 120 etiquetas de un minimo de 500
    assert result.returncode == 3
    assert "CHURN_TRAIN_DATA no esta definida" in result.stdout
    assert "solo hay 120 ejemplos etiquetados (120 etiquetas) de un minimo de 500" in result.stdout
    assert "POST /labels" in result.stdout
    assert (
        "tenure_months,monthly_charges" in result.stdout
    )  # el motivo del reentreno queda en el aviso
    docker = _log(retrain.logs, "docker")
    assert docker.count("\n") == 1 and LABELS_BUILD in docker  # solo el intento de construirlo
    assert "churn.training.train" not in docker  # no se reentrena con el generador sintetico
    assert "/model/reload" not in _log(retrain.logs, "curl")


def test_retrain_forzado_tambien_exige_datos(retrain):
    result = retrain(FORCE="1")
    assert result.returncode == 3
    assert "forzado" in result.stdout
    assert "/monitoring/drift" not in _log(retrain.logs, "curl")


def test_retrain_sin_train_data_construye_el_dataset_con_las_etiquetas_de_la_api(retrain):
    result = retrain(labels=LABELS_OK, labels_exit="0", drift=DRIFT_PREDICTIONS)
    assert result.returncode == 0, result.stdout + result.stderr
    docker = _log(retrain.logs, "docker").splitlines()
    assert docker == [LABELS_BUILD, LABELS_TRAIN]  # sin montar nada: el dataset vive en el volumen
    assert "640 filas etiquetadas (predicciones del 2026-09-01T08:00:00+00:00 al" in result.stdout
    assert "Reentrenando con etiquetas de la API (640 filas etiquetadas): drift" in result.stdout
    curl = _log(retrain.logs, "curl")
    assert "/model/reload" in curl
    assert (
        "Modelo recargado: version v-nueva (drift detectado (predicciones), datos etiquetas de "
        "la API (640 filas etiquetadas))" in result.stdout
    )


def test_retrain_con_train_data_no_construye_el_dataset_de_etiquetas(retrain, tmp_path):
    data = tmp_path / "clientes.csv"
    data.write_text("x")
    result = retrain(CHURN_TRAIN_DATA=str(data), labels=LABELS_OK, labels_exit="0")
    assert result.returncode == 0
    docker = _log(retrain.logs, "docker")
    assert "churn.data.labels" not in docker and "--data /data/clientes.csv" in docker


def test_retrain_dataset_de_etiquetas_invalido_sale_2(retrain):
    result = retrain(labels=LABELS_ERROR, labels_exit="2")
    assert result.returncode == 2
    assert "no se pudo construir el dataset de etiquetas (codigo 2): Tasa de churn" in result.stdout
    assert "churn.training.train" not in _log(retrain.logs, "docker")
    assert "/model/reload" not in _log(retrain.logs, "curl")


def test_retrain_fallo_al_construir_el_dataset_sale_2(retrain):
    result = retrain(labels="", labels_exit="1")  # docker/trainer caido: sin resumen JSON
    assert result.returncode == 2
    assert "(codigo 1): sin detalle" in result.stdout
    assert "churn.training.train" not in _log(retrain.logs, "docker")


def test_retrain_no_recarga_si_el_retador_de_las_etiquetas_no_se_promueve(retrain):
    result = retrain(labels=LABELS_OK, labels_exit="0", docker_exit="3")
    assert result.returncode == 0
    assert "NO promovido" in result.stdout
    assert "/model/reload" not in _log(retrain.logs, "curl")


def test_retrain_con_fichero_inexistente_falla_con_claridad(retrain, tmp_path):
    result = retrain(CHURN_TRAIN_DATA=str(tmp_path / "no-existe.csv"))
    assert result.returncode == 3
    assert "no existe" in result.stdout
    assert _log(retrain.logs, "docker") == ""


def test_retrain_monta_los_datos_y_recarga_si_se_promueve(retrain, tmp_path):
    data = tmp_path / "datos" / "clientes_q3.parquet"
    data.parent.mkdir()
    data.write_bytes(b"PAR1")
    result = retrain(CHURN_TRAIN_DATA=str(data), drift=DRIFT_PREDICTIONS)
    assert result.returncode == 0, result.stdout + result.stderr
    docker = _log(retrain.logs, "docker")
    assert docker.count("\n") == 1
    assert "compose -f docker-compose.prod.yml --profile train run --rm" in docker
    assert f"-v {data.resolve()}:/data/clientes_q3.parquet:ro trainer" in docker
    assert (
        "python -m churn.training.train --data /data/clientes_q3.parquet --model-dir /models"
        in docker
    )
    curl = _log(retrain.logs, "curl")
    assert "/model/reload" in curl and "X-Admin-Token: token-de-prueba-0123456789" in curl
    assert "Modelo recargado: version v-nueva" in result.stdout
    assert "(drift detectado (predicciones), datos clientes_q3.parquet)" in result.stdout


def test_retrain_no_recarga_si_el_retador_no_se_promueve(retrain, tmp_path):
    data = tmp_path / "clientes.csv"
    data.write_text("x")
    result = retrain(CHURN_TRAIN_DATA=str(data), docker_exit="3")
    assert result.returncode == 0
    assert "NO promovido" in result.stdout
    assert "/model/reload" not in _log(retrain.logs, "curl")


def test_retrain_rechazado_por_el_gate_devuelve_2(retrain, tmp_path):
    data = tmp_path / "clientes.csv"
    data.write_text("x")
    result = retrain(CHURN_TRAIN_DATA=str(data), docker_exit="2")
    assert result.returncode == 2
    assert "rechazado" in result.stdout and "codigo 2" in result.stdout
    assert "/model/reload" not in _log(retrain.logs, "curl")


def test_retrain_envia_avisos_al_webhook(retrain, tmp_path):
    data = tmp_path / "clientes.csv"
    data.write_text("x")
    result = retrain(CHURN_TRAIN_DATA=str(data), WEBHOOK_URL="http://hook.test/x")
    assert result.returncode == 0
    curl = _log(retrain.logs, "curl")
    assert curl.count("http://hook.test/x") == 2  # "Reentrenando" + "Modelo recargado"
    assert "[churn] Reentrenando con clientes.csv" in curl


def test_retrain_no_reentrena_si_no_hay_etiquetas_nuevas_desde_el_modelo_en_servicio(retrain):
    servido = '{"model_version": "v1", "data_source": {"kind": "file", "fingerprint": "abc"}}'
    result = retrain(labels=LABELS_OK, labels_exit="0", MODEL_INFO_BODY=servido)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no hay etiquetas nuevas desde el entrenamiento del modelo en servicio" in result.stdout
    assert "(huella abc)" in result.stdout
    assert _log(retrain.logs, "docker").splitlines() == [LABELS_BUILD]  # sin entrenar
    curl = _log(retrain.logs, "curl")
    assert "/model/info" in curl and "/model/reload" not in curl


def test_retrain_con_etiquetas_nuevas_o_sin_model_info_reentrena(retrain):
    otra = '{"data_source": {"fingerprint": "otra"}}'
    for body in (otra, "", "{rota"):  # huella distinta, API sin modelo o respuesta ilegible
        result = retrain(labels=LABELS_OK, labels_exit="0", MODEL_INFO_BODY=body)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Modelo recargado" in result.stdout
    assert _log(retrain.logs, "docker").count("churn.training.train") == 3


def test_retrain_una_sola_ejecucion_a_la_vez(retrain):
    lock = (retrain.root / ".retrain.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # otra ejecucion en curso
    try:
        result = retrain(labels=LABELS_OK, labels_exit="0", WEBHOOK_URL="http://hook.test/x")
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert result.returncode == 0
    assert "Ya hay un reentrenamiento en curso" in result.stdout
    assert _log(retrain.logs, "docker") == ""
    curl = _log(retrain.logs, "curl")
    assert "/monitoring/drift" not in curl and "http://hook.test/x" in curl  # avisa

    libre = retrain(labels=LABELS_OK, labels_exit="0")  # liberado: vuelve a ejecutarse
    assert libre.returncode == 0 and "Modelo recargado" in libre.stdout


# --------------------------------------------------------------------------- backup.sh

BACKUP_DOCKER_STUB = """#!/usr/bin/env bash
# Doble de docker para backup.sh: registra cada llamada (en una linea) y simula sus salidas.
line="$*"
echo "${line//$'\\n'/ }" >> "$STUB_LOG_DIR/docker.log"
if [ "$1" = volume ]; then
  [ "$3" != "${MISSING_VOLUME:-}" ]; exit $?
fi
case "$*" in
  compose*) exit "${SNAPSHOT_EXIT:-0}" ;;  # instantanea de drift.sqlite
esac
host_b=""; prev=""
for arg in "$@"; do
  case "$arg" in *:/b) [ "$prev" = "-v" ] && host_b="${arg%:/b}" ;; esac
  prev="$arg"
done
case "$*" in
  *"tar czf"*) for arg in "$@"; do case "$arg" in /b/*) : > "$host_b/${arg#/b/}" ;; esac; done ;;
  *"sh -c"*) [ "${SNAPSHOT_EXISTS:-1}" = 1 ] && : > "$host_b/${*: -1}" ;;
esac
exit 0
"""


@pytest.fixture()
def backup(tmp_path):
    """Copia backup.sh a un arbol temporal y lo ejecuta con un docker de mentira."""
    root = tmp_path / "mlops-churn-platform"
    (root / "deploy" / "scripts").mkdir(parents=True)
    script = root / "deploy" / "scripts" / "backup.sh"
    shutil.copy(BACKUP_SCRIPT, script)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "docker").write_text(BACKUP_DOCKER_STUB)
    (stubs / "docker").chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    backups = tmp_path / "backups"

    def _run(**env) -> subprocess.CompletedProcess:
        full_env = {
            **os.environ,
            "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
            "STUB_LOG_DIR": str(logs),
            "BACKUP_DIR": str(backups),
            **env,
        }
        full_env.pop("COMPOSE_PROJECT_NAME", None)
        return subprocess.run(
            ["bash", str(script)], capture_output=True, text=True, timeout=60, env=full_env
        )

    _run.logs = logs  # type: ignore[attr-defined]
    _run.backups = backups  # type: ignore[attr-defined]
    return _run


def test_backup_copia_drift_sqlite_con_una_instantanea_y_no_archiva_el_wal_vivo(backup):
    result = backup()
    assert result.returncode == 0, result.stdout + result.stderr

    calls = _log(backup.logs, "docker").splitlines()
    snapshot = next(i for i, c in enumerate(calls) if c.startswith("compose"))
    tar_models = next(i for i, c in enumerate(calls) if "models:/v" in c and "tar czf" in c)
    assert snapshot < tar_models  # primero la instantanea, despues el archivo del volumen
    assert calls[snapshot].startswith(
        "compose -f docker-compose.prod.yml --profile train run --rm --no-deps -T trainer python -c"
    )
    assert calls[snapshot].endswith("/models/drift.sqlite /models/.backup-drift.sqlite")
    assert "source.backup(target)" in calls[snapshot]  # API de backup de SQLite, no cp ni tar
    for vivo in ("./drift.sqlite", "./drift.sqlite-wal", "./drift.sqlite-shm"):
        assert f"--exclude {vivo} " in calls[tar_models]
    tar_mlflow = next(c for c in calls if "mlflow-data:/v" in c and "tar czf" in c)
    assert "--exclude" not in tar_mlflow
    nombres = sorted(f.name.split("-2")[0] for f in backup.backups.iterdir())
    assert nombres == ["drift", "mlflow-data", "models"]
    assert "instantanea consistente" in result.stdout


def test_backup_sin_drift_sqlite_solo_archiva_los_volumenes(backup):
    result = backup(SNAPSHOT_EXISTS="0")  # API sin predicciones todavia: nada que copiar
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(f.name.split("-2")[0] for f in backup.backups.iterdir()) == [
        "mlflow-data",
        "models",
    ]


def test_backup_falla_si_no_puede_hacer_la_instantanea(backup):
    result = backup(SNAPSHOT_EXIT="1")
    assert result.returncode != 0  # mejor una alerta que una copia inconsistente
    assert "tar czf" not in _log(backup.logs, "docker")


def test_backup_omite_volumenes_inexistentes_y_aplica_la_retencion(backup):
    backup.backups.mkdir()
    vieja = backup.backups / "drift-20200101T000000Z.sqlite"
    vieja.write_text("x")
    hace_un_mes = time.time() - 30 * 86400
    os.utime(vieja, (hace_un_mes, hace_un_mes))

    result = backup(MISSING_VOLUME="mlops-churn-platform_mlflow-data", KEEP_DAYS="14")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "El volumen mlops-churn-platform_mlflow-data no existe; se omite" in result.stdout
    assert not vieja.exists()
    assert sorted(f.name.split("-2")[0] for f in backup.backups.iterdir()) == ["drift", "models"]
