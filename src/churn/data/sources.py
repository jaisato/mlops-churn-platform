"""Fuentes de datos de entrenamiento: fichero etiquetado (CSV/Parquet) o generador.

El trainer entrena con lo que le den; la validacion de esquema, rangos y balance la
hace `churn.data.validation` justo antes de entrenar. Aqui solo se resuelve de donde
salen las filas y se deja una huella (`fingerprint`) del dataset en el metadata:
dos entrenamientos con la misma huella han visto exactamente los mismos datos, asi
que un "reentrenamiento" con la misma huella y semilla no puede cambiar el modelo.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from churn.data.generator import DriftSpec, generate_dataset

SUPPORTED_SUFFIXES = (".csv", ".parquet")


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Lee un fichero etiquetado (.csv o .parquet). Lanza `FileNotFoundError` o `ValueError`.

    No valida el contenido (eso lo hace `validate_training_data` en el trainer, que
    rechaza columnas ausentes, nulos, rangos y tasas de churn imposibles); solo garantiza
    que el fichero existe, tiene un formato conocido y contiene alguna fila.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Fichero de datos de entrenamiento no encontrado: {file}")
    suffix = file.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Formato no soportado para {file.name}: se admiten {list(SUPPORTED_SUFFIXES)}"
        )
    try:
        df = pd.read_csv(file) if suffix == ".csv" else pd.read_parquet(file)
    except ImportError as exc:  # parquet sin motor (pyarrow) instalado
        raise ValueError(f"No se puede leer {file.name}: {exc}") from exc
    except (ValueError, OSError) as exc:  # vacio, corrupto o ilegible
        raise ValueError(f"No se puede leer {file.name}: {exc}") from exc
    if df.empty:
        raise ValueError(f"El fichero {file.name} no contiene filas")
    return df


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 estable del contenido (valores y nombres de columna, sin el indice)."""
    digest = hashlib.sha256()
    digest.update(",".join(map(str, df.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def resolve_training_data(
    *,
    data_path: str | Path | None = None,
    rows: int = 20_000,
    seed: int = 42,
    drift_shift: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Devuelve `(df, data_source)`: las filas y la descripcion que va al metadata.

    - Con `data_path`: filas del fichero; `drift_shift` no se aplica (los datos reales
      ya traen su propio drift) y se rechaza si es distinto de cero para evitar confusion.
    - Sin `data_path`: generador sintetico con `rows`, `seed` y, opcionalmente, un
      escenario de drift (`DriftSpec.from_shift`).
    """
    if data_path is not None:
        if drift_shift:
            raise ValueError("--drift-shift solo tiene sentido con datos sinteticos (sin --data)")
        df = load_training_data(data_path)
        source: dict[str, Any] = {"kind": "file", "path": str(data_path)}
    else:
        spec = DriftSpec.from_shift(drift_shift)
        df = generate_dataset(n_rows=rows, seed=seed, drift=spec)
        source = {"kind": "synthetic", "seed": seed, "drift_shift": drift_shift}
    source["rows"] = int(len(df))
    source["fingerprint"] = dataset_fingerprint(df)
    return df, source
