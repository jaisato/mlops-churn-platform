"""Dataset de reentreno a partir de las etiquetas reales recibidas por la API.

Cierra el bucle de etiquetas: la API guarda cada prediccion (features + probabilidad +
`prediction_id`), el negocio devuelve mas tarde el `churn` real de cada una
(POST /labels) y aqui se convierte ese historial etiquetado en un fichero con
exactamente las columnas del contrato de entrenamiento, listo para `--data` /
`CHURN_TRAIN_DATA`. Junto al fichero se escribe un sidecar `<fichero>.meta.json` con
el numero de filas, la ventana temporal y la huella SHA-256, que el trainer incorpora al
metadata del modelo (`data_source.kind = "labels"`).

Uso:
    python -m churn.data.labels --model-dir /models --out /models/datasets/labels.parquet
    python -m churn.data.labels --db models/drift.sqlite --out data/labels.csv --min-rows 800
Salida: una linea JSON en stdout con el resumen (`status`, filas, ventana, huella, ruta);
los logs van a stderr para que un script pueda capturar el resumen.
Codigos de salida: 0 dataset escrito | 2 error (almacen inexistente o dataset invalido
para entrenar) | 3 etiquetas insuficientes (menos de --min-rows / CHURN_LABELS_MIN_ROWS).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from churn.config import get_settings
from churn.data.generator import FEATURE_COLUMNS, TARGET
from churn.data.sources import (
    SUPPORTED_SUFFIXES,
    dataset_fingerprint,
    load_training_data,
    sidecar_path,
)
from churn.data.validation import validate_training_data
from churn.logging_conf import configure_logging
from churn.monitoring.store import (
    OBSERVED_AT_COLUMN,
    PREDICTED_AT_COLUMN,
    VERSION_COLUMN,
    PredictionStore,
    SqlitePredictionStore,
    default_db_path,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_INSUFFICIENT = 3
DEFAULT_DATASET = Path("datasets") / "labels.parquet"  # relativo a --model-dir


class InsufficientLabelsError(ValueError):
    """Hay menos predicciones etiquetadas que el minimo exigido."""

    def __init__(self, rows: int, min_rows: int) -> None:
        super().__init__(f"Etiquetas insuficientes: {rows} de un minimo de {min_rows}")
        self.rows = rows
        self.min_rows = min_rows


@dataclass(frozen=True)
class LabelledWindow:
    """Descripcion del historial etiquetado: filas, ventana temporal y versiones."""

    rows: int
    predicted_from: str | None
    predicted_to: str | None
    observed_from: str | None
    observed_to: str | None
    model_versions: list[str]
    churn_rate: float | None


def labelled_dataset(store: PredictionStore) -> tuple[pd.DataFrame, LabelledWindow]:
    """Todas las predicciones etiquetadas como dataset de entrenamiento + su descripcion.

    El DataFrame tiene exactamente `FEATURE_COLUMNS + [churn]`, en orden cronologico de
    prediccion. Vacio (sin columnas) si no hay etiquetas.
    """
    rows = store.labelled()
    if rows.empty:
        return pd.DataFrame(), LabelledWindow(0, None, None, None, None, [], None)
    df = rows[FEATURE_COLUMNS + [TARGET]].reset_index(drop=True)
    window = LabelledWindow(
        rows=int(len(df)),
        predicted_from=str(rows[PREDICTED_AT_COLUMN].min()),
        predicted_to=str(rows[PREDICTED_AT_COLUMN].max()),
        observed_from=str(rows[OBSERVED_AT_COLUMN].min()),
        observed_to=str(rows[OBSERVED_AT_COLUMN].max()),
        model_versions=sorted(rows[VERSION_COLUMN].astype(str).unique().tolist()),
        churn_rate=round(float(df[TARGET].mean()), 4),
    )
    return df, window


def write_labelled_dataset(
    store: PredictionStore, out: str | Path, *, min_rows: int
) -> dict[str, Any]:
    """Escribe el dataset etiquetado (.csv/.parquet) y su sidecar; devuelve el resumen.

    Lanza `InsufficientLabelsError` si hay menos de `min_rows` filas y `ValueError` si el
    dataset no pasaria la validacion del trainer (asi el fallo se ve aqui, no en el
    reentreno). La huella se calcula releyendo el fichero escrito: es exactamente la que
    el trainer computara al cargarlo, y por eso puede verificar el sidecar.
    """
    path = Path(out)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Formato no soportado para {path.name}: {list(SUPPORTED_SUFFIXES)}")
    df, window = labelled_dataset(store)
    if window.rows < min_rows:
        raise InsufficientLabelsError(window.rows, min_rows)
    report = validate_training_data(df)
    if not report.is_valid:
        raise ValueError(f"El dataset etiquetado no es valido para entrenar: {report.errors}")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    summary: dict[str, Any] = {
        "kind": "labels",
        "path": str(path),
        "built_at": datetime.now(UTC).isoformat(),
        "fingerprint": dataset_fingerprint(load_training_data(path)),
        **asdict(window),
    }
    sidecar_path(path).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        "Dataset etiquetado escrito en %s: %s filas (churn %.1f%%), predicciones %s..%s, "
        "etiquetas observadas %s..%s, huella %s",
        path,
        window.rows,
        100 * (window.churn_rate or 0.0),
        window.predicted_from,
        window.predicted_to,
        window.observed_from,
        window.observed_to,
        summary["fingerprint"][:12],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, stream=sys.stderr)  # stdout queda para el resumen JSON
    parser = argparse.ArgumentParser(
        description="Construye el dataset de reentreno con las etiquetas recibidas por la API"
    )
    parser.add_argument("--model-dir", default=settings.model_dir, help="Volumen de modelos")
    parser.add_argument(
        "--db",
        default=None,
        help="Fichero SQLite de predicciones (por defecto CHURN_DRIFT_DB_PATH o "
        "<model-dir>/drift.sqlite)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"Dataset de salida .csv/.parquet (por defecto <model-dir>/{DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=None,
        help="Minimo de filas etiquetadas (por defecto CHURN_LABELS_MIN_ROWS)",
    )
    args = parser.parse_args(argv)
    db = Path(args.db) if args.db else default_db_path(settings, args.model_dir)
    out = Path(args.out) if args.out else Path(args.model_dir) / DEFAULT_DATASET
    min_rows = args.min_rows if args.min_rows is not None else settings.labels_min_rows

    # Abrir el almacen lo crearia vacio: una ruta equivocada debe verse como error, no
    # como "0 etiquetas".
    if not db.is_file():
        logger.error("No existe el almacen de predicciones %s", db)
        print(json.dumps({"status": "error", "error": f"No existe el almacen {db}"}))
        return EXIT_ERROR
    store = SqlitePredictionStore(db, max_rows=settings.prediction_keep_rows)
    try:
        summary = write_labelled_dataset(store, out, min_rows=min_rows)
    except InsufficientLabelsError as exc:
        logger.error("%s; no se construye el dataset", exc)
        print(json.dumps({"status": "insufficient", "rows": exc.rows, "min_rows": min_rows}))
        return EXIT_INSUFFICIENT
    except ValueError as exc:
        logger.error("Dataset etiquetado rechazado: %s", exc)
        print(json.dumps({"status": "error", "error": str(exc)}))
        return EXIT_ERROR
    print(json.dumps({"status": "ok", **summary}))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
