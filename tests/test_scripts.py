import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from churn.data.generator import FEATURE_COLUMNS, TARGET, DriftSpec, generate_dataset

REPO = Path(__file__).resolve().parents[1]
RETRAIN_SCRIPT = REPO / "deploy" / "scripts" / "retrain-if-drift.sh"


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
  */model/reload) exit 0 ;;
  */health) printf '{"model_version": "v-nueva"}' ;;
  *) exit 22 ;;
esac
"""

DOCKER_STUB = """#!/usr/bin/env bash
echo "$*" >> "$STUB_LOG_DIR/docker.log"
exit "${DOCKER_EXIT:-0}"
"""


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

    def _run(*, drift=DRIFT_YES, http="200", docker_exit="0", **env) -> subprocess.CompletedProcess:
        full_env = {
            **os.environ,
            "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
            "STUB_LOG_DIR": str(logs),
            "DRIFT_BODY": drift,
            "DRIFT_HTTP": http,
            "DOCKER_EXIT": docker_exit,
            "CHURN_ADMIN_TOKEN": "token-de-prueba-0123456789",
            **env,
        }
        full_env.pop("CHURN_TRAIN_DATA", None) if "CHURN_TRAIN_DATA" not in env else None
        return subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            timeout=60,
            env=full_env,
            check=False,
        )

    _run.logs = logs  # type: ignore[attr-defined]
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


def test_retrain_con_drift_pero_sin_datos_etiquetados_falla_con_claridad(retrain):
    result = retrain()
    assert result.returncode == 3
    assert "CHURN_TRAIN_DATA no esta definida" in result.stdout
    assert (
        "tenure_months,monthly_charges" in result.stdout
    )  # el motivo del reentreno queda en el aviso
    assert _log(retrain.logs, "docker") == ""  # no se reentrena con el generador sintetico


def test_retrain_forzado_tambien_exige_datos(retrain):
    result = retrain(FORCE="1")
    assert result.returncode == 3
    assert "forzado" in result.stdout
    assert "/monitoring/drift" not in _log(retrain.logs, "curl")


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
