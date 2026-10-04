"""Dataset de reentreno a partir de las etiquetas reales recibidas por la API.

Cierra el bucle de etiquetas: la API guarda cada prediccion (features + probabilidad +
`prediction_id` + `subject_ref` opcional), el negocio devuelve mas tarde el `churn` real de
cada una (POST /labels) y aqui se convierte ese historial etiquetado en un fichero con las
columnas del contrato de entrenamiento mas unas columnas de trazabilidad (`TRACE_COLUMNS`:
no son features) que el trainer usa para separar el holdout por sujeto y para comparar con
el campeon solo con etiquetas posteriores a las suyas. Listo para `--data` /
`CHURN_TRAIN_DATA`. Junto al fichero se escribe un sidecar `<fichero>.meta.json` con el
numero de filas, la ventana temporal y la huella SHA-256, que el trainer incorpora al
metadata del modelo (bloque `data_source.labels`).

Un ejemplo por sujeto y horizonte: si el mismo cliente (`subject_ref`) se puntuo varias
veces y todas esas predicciones se etiquetaron con la misma observacion (`observed_at`),
solo cuenta la prediccion mas reciente: las demas son casi duplicados que inflarian el peso
de ese cliente. Sin `subject_ref` solo se pueden reconocer las filas identicas (mismas
features y etiqueta), que se cuentan una vez.

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
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    ID_COLUMN,
    OBSERVED_AT_COLUMN,
    PREDICTED_AT_COLUMN,
    SUBJECT_COLUMN,
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
#: Columnas de trazabilidad que acompanan a las del contrato (no son features).
TRACE_COLUMNS = [ID_COLUMN, SUBJECT_COLUMN, PREDICTED_AT_COLUMN, OBSERVED_AT_COLUMN]


class InsufficientLabelsError(ValueError):
    """Hay menos ejemplos etiquetados (tras deduplicar) que el minimo exigido."""

    def __init__(self, rows: int, min_rows: int, labels_total: int) -> None:
        super().__init__(
            f"Etiquetas insuficientes: {rows} ejemplos (de {labels_total} etiquetas) de un "
            f"minimo de {min_rows}"
        )
        self.rows = rows
        self.min_rows = min_rows
        self.labels_total = labels_total


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
    labels_total: int = 0
    subjects: int = 0


def one_label_per_subject_and_horizon(rows: pd.DataFrame) -> pd.DataFrame:
    """Quita los casi duplicados del historial etiquetado, conservando el orden de prediccion.

    - Con `subject_ref`: por cada (sujeto, `observed_at`) solo la prediccion mas reciente
      (las etiquetas no pueden ser anteriores a su prediccion, asi que es la ultima
      anterior a la observacion). Observaciones distintas del mismo sujeto (otro horizonte)
      son ejemplos distintos y se conservan; el trainer las mantiene del mismo lado.
    - Sin `subject_ref`: las filas identicas (features y etiqueta) se cuentan una vez.
    """
    has_subject = rows[SUBJECT_COLUMN].notna()
    by_subject = rows[has_subject].drop_duplicates(
        [SUBJECT_COLUMN, OBSERVED_AT_COLUMN], keep="last"
    )
    anonymous = rows[~has_subject].drop_duplicates(FEATURE_COLUMNS + [TARGET], keep="last")
    return rows.loc[rows.index.isin(by_subject.index.union(anonymous.index))]


def labelled_dataset(store: PredictionStore) -> tuple[pd.DataFrame, LabelledWindow]:
    """Las predicciones etiquetadas como dataset de entrenamiento + su descripcion.

    El DataFrame tiene `FEATURE_COLUMNS + [churn] + TRACE_COLUMNS`, en orden cronologico de
    prediccion y con una etiqueta por sujeto y horizonte. Vacio (sin columnas) si no hay
    etiquetas.
    """
    rows = store.labelled()
    if rows.empty:
        return pd.DataFrame(), LabelledWindow(0, None, None, None, None, [], None)
    kept = one_label_per_subject_and_horizon(rows)
    df = kept[FEATURE_COLUMNS + [TARGET] + TRACE_COLUMNS].reset_index(drop=True)
    window = LabelledWindow(
        rows=int(len(df)),
        predicted_from=str(kept[PREDICTED_AT_COLUMN].min()),
        predicted_to=str(kept[PREDICTED_AT_COLUMN].max()),
        observed_from=str(kept[OBSERVED_AT_COLUMN].min()),
        observed_to=str(kept[OBSERVED_AT_COLUMN].max()),
        model_versions=sorted(kept[VERSION_COLUMN].astype(str).unique().tolist()),
        churn_rate=round(float(df[TARGET].mean()), 4),
        labels_total=int(len(rows)),
        subjects=int(kept[SUBJECT_COLUMN].nunique()),
    )
    return df, window


def _replace_atomically(path: Path, write: Any) -> None:
    """Escribe en un temporal del mismo directorio y lo renombra encima de `path`.

    Un lector (el trainer) ve el fichero anterior completo o el nuevo completo, nunca uno a
    medias; si la escritura falla, el anterior queda intacto y el temporal se borra.
    """
    tmp = path.with_name(f".{path.stem}.{uuid4().hex[:8]}.tmp{path.suffix}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_labelled_dataset(
    store: PredictionStore, out: str | Path, *, min_rows: int
) -> dict[str, Any]:
    """Escribe el dataset etiquetado (.csv/.parquet) y su sidecar; devuelve el resumen.

    Lanza `InsufficientLabelsError` si hay menos de `min_rows` filas y `ValueError` si el
    dataset no pasaria la validacion del trainer (asi el fallo se ve aqui, no en el
    reentreno). La huella se calcula releyendo el fichero escrito: es exactamente la que
    el trainer computara al cargarlo, y por eso puede verificar el sidecar. Ambos ficheros
    se sustituyen de forma atomica.
    """
    path = Path(out)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Formato no soportado para {path.name}: {list(SUPPORTED_SUFFIXES)}")
    df, window = labelled_dataset(store)
    if window.rows < min_rows:
        raise InsufficientLabelsError(window.rows, min_rows, window.labels_total)
    report = validate_training_data(df)
    if not report.is_valid:
        raise ValueError(f"El dataset etiquetado no es valido para entrenar: {report.errors}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprints: list[str] = []

    def write_dataset(tmp: Path) -> None:
        if path.suffix.lower() == ".csv":
            df.to_csv(tmp, index=False)
        else:
            df.to_parquet(tmp, index=False)
        fingerprints.append(dataset_fingerprint(load_training_data(tmp)))

    _replace_atomically(path, write_dataset)
    summary: dict[str, Any] = {
        "kind": "labels",
        "path": str(path),
        "built_at": datetime.now(UTC).isoformat(),
        "fingerprint": fingerprints[0],
        **asdict(window),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    _replace_atomically(sidecar_path(path), lambda tmp: tmp.write_text(text, encoding="utf-8"))
    logger.info(
        "Dataset etiquetado escrito en %s: %s filas de %s etiquetas (%s sujetos, churn %.1f%%), "
        "predicciones %s..%s, etiquetas observadas %s..%s, huella %s",
        path,
        window.rows,
        window.labels_total,
        window.subjects,
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
        help="Minimo de ejemplos etiquetados tras deduplicar (por defecto CHURN_LABELS_MIN_ROWS)",
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
        print(
            json.dumps(
                {
                    "status": "insufficient",
                    "rows": exc.rows,
                    "labels_total": exc.labels_total,
                    "min_rows": min_rows,
                }
            )
        )
        return EXIT_INSUFFICIENT
    except ValueError as exc:
        logger.error("Dataset etiquetado rechazado: %s", exc)
        print(json.dumps({"status": "error", "error": str(exc)}))
        return EXIT_ERROR
    print(json.dumps({"status": "ok", **summary}))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
