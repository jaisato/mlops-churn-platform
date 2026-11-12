"""mypy limpio con el interprete que ejecuta los tests.

La CI pasa mypy una sola vez, en Python 3.12, pero los tests corren en 3.11, 3.12 y 3.13
y el lockfile no les da las mismas dependencias: numpy 2.4 a 3.11 y numpy 2.5 a las otras
dos, cada uno con sus stubs. `generator.py` tuvo un error de tipos que solo existia en 3.11
y nadie lo veia. Este test ejecuta el `mypy src/churn` de la CI con el interprete en curso,
asi que cada version de la matriz de tests vigila la suya, sin jobs ni instalaciones de mas.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_mypy_no_encuentra_errores_con_este_interprete(tmp_path):
    if "MUTANT_UNDER_TEST" in os.environ:
        # mutmut ejecuta los tests sobre una copia del codigo con los mutantes y sus
        # trampolines dentro: ahi mypy da miles de errores que no son del proyecto.
        pytest.skip("bajo mutmut el codigo que hay en src/ es el mutado")

    # Con su propia cache: el resultado no depende de lo que haya en .mypy_cache
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "src/churn", "--cache-dir", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stdout + result.stderr
