import subprocess
import sys
from pathlib import Path

import pandas as pd

from churn.data.generator import FEATURE_COLUMNS, TARGET

REPO = Path(__file__).resolve().parents[1]


def test_generate_data_funciona_desde_cualquier_cwd(tmp_path):
    out = tmp_path / "salida" / "churn.csv"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "generate_data.py"),
         "--rows", "600", "--seed", "3", "--out", str(out)],
        cwd=tmp_path,  # distinto del raiz del repo: el script no debe depender del cwd
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "600 filas escritas" in result.stdout
    df = pd.read_csv(out)
    assert len(df) == 600
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET]
