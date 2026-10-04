"""Fuentes de datos de entrenamiento: fichero etiquetado (CSV/Parquet) o generador.

El trainer entrena con lo que le den; la validacion de esquema, rangos y balance la
hace `churn.data.validation` justo antes de entrenar. Aqui solo se resuelve de donde
salen las filas y se deja una huella (`fingerprint`) del dataset en el metadata:
dos entrenamientos con la misma huella han visto exactamente los mismos datos, asi
que un "reentrenamiento" con la misma huella y semilla no puede cambiar el modelo.

Un dataset construido desde las etiquetas de produccion (`churn.data.labels`) va
acompanado de un fichero `<dataset>.meta.json` (sidecar) con su huella y su ventana
temporal; si la huella coincide con el fichero cargado, esa informacion pasa al metadata
del modelo (`data_source.kind = "labels"`) sin cambiar la interfaz `--data`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from churn.data.generator import DriftSpec, generate_dataset

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".csv", ".parquet")
SIDECAR_SUFFIX = ".meta.json"
#: Claves del sidecar que describen la ventana de etiquetas y viajan al metadata del modelo.
LABELS_WINDOW_KEYS = (
    "built_at",
    "predicted_from",
    "predicted_to",
    "observed_from",
    "observed_to",
    "model_versions",
)


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


def sidecar_path(path: str | Path) -> Path:
    """Fichero `<dataset>.meta.json` que acompana a un dataset construido desde etiquetas."""
    return Path(f"{path}{SIDECAR_SUFFIX}")


def read_sidecar(path: str | Path) -> dict[str, Any] | None:
    """Contenido del sidecar del dataset, o None si no existe o no es un objeto JSON."""
    file = sidecar_path(path)
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except OSError:
        return None
    except ValueError:
        logger.warning("Sidecar %s ilegible: se ignora", file)
        return None
    return payload if isinstance(payload, dict) else None


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
      Si el fichero tiene un sidecar cuya huella coincide, la fuente es `labels` y lleva
      la ventana temporal de las etiquetas.
    - Sin `data_path`: generador sintetico con `rows`, `seed` y, opcionalmente, un
      escenario de drift (`DriftSpec.from_shift`).
    """
    if data_path is not None:
        if drift_shift:
            raise ValueError("--drift-shift solo tiene sentido con datos sinteticos (sin --data)")
        df = load_training_data(data_path)
        source: dict[str, Any] = {"kind": "file", "path": str(data_path)}
        fingerprint = dataset_fingerprint(df)
        sidecar = read_sidecar(data_path)
        if sidecar is not None:
            if sidecar.get("fingerprint") == fingerprint:
                source["kind"] = "labels"
                source["labels"] = {k: sidecar.get(k) for k in LABELS_WINDOW_KEYS}
            else:
                logger.warning(
                    "La huella del sidecar de %s no coincide con el fichero (modificado tras "
                    "construirlo): se ignora la procedencia de las etiquetas",
                    data_path,
                )
    else:
        spec = DriftSpec.from_shift(drift_shift)
        df = generate_dataset(n_rows=rows, seed=seed, drift=spec)
        source = {"kind": "synthetic", "seed": seed, "drift_shift": drift_shift}
        fingerprint = dataset_fingerprint(df)
    source["rows"] = int(len(df))
    source["fingerprint"] = fingerprint
    return df, source
