"""Almacen de predicciones recientes (drift) y de sus etiquetas reales (bucle de etiquetas).

Dos backends con el mismo contrato:
  - SqlitePredictionStore: fichero SQLite (modo WAL) en el volumen. Sobrevive a los
    reinicios y lo comparten todos los workers/procesos de la API y el trainer.
  - MemoryPredictionStore: estructuras en memoria. Para tests y demos de un solo proceso.

Predicciones: cada fila guarda las features recibidas, la probabilidad devuelta, la
version del modelo que la produjo y un `prediction_id` que la API devuelve al cliente.
El tamano se acota a `max_rows` (las mas antiguas se desalojan); `recent(limit)` es la
ventana que mira el drift.

Etiquetas: `label()` recibe el `churn` real de una prediccion y guarda una COPIA del
ejemplo (features + probabilidad + version + etiqueta) en una tabla aparte, de modo que
el ejemplo etiquetado sobrevive al desalojo de la ventana de predicciones y
`labelled()` devuelve directamente el dataset de reentreno, en el orden en que se
hicieron las predicciones. Reetiquetar sobrescribe: la ultima etiqueta recibida es la
que vale.
"""

from __future__ import annotations

import itertools
import json
import sqlite3
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import pandas as pd

from churn.config import Settings

SCORE_COLUMN = "churn_probability"
VERSION_COLUMN = "model_version"
ID_COLUMN = "prediction_id"
PREDICTED_AT_COLUMN = "predicted_at"
OBSERVED_AT_COLUMN = "observed_at"
LABELLED_AT_COLUMN = "labelled_at"
LABEL_COLUMN = "churn"
DEFAULT_DB_NAME = "drift.sqlite"
#: Columnas de una fila de prediccion que no son features del cliente.
_NON_FEATURE_COLUMNS = frozenset({SCORE_COLUMN, VERSION_COLUMN, ID_COLUMN, PREDICTED_AT_COLUMN})


@dataclass
class LabelOutcome:
    """Resultado de un lote de etiquetas: cuantas eran nuevas, cuantas corrigen una
    etiqueta anterior y que `prediction_id` no estan (o ya no estan) en el almacen."""

    created: int = 0
    updated: int = 0
    unknown: list[str] = field(default_factory=list)


class PredictionStore(Protocol):
    def append(self, rows: Iterable[dict]) -> list[str]: ...

    def recent(self, limit: int) -> pd.DataFrame: ...

    def count(self) -> int: ...

    def label(self, labels: Iterable[dict]) -> LabelOutcome: ...

    def labelled(self, limit: int | None = None) -> pd.DataFrame: ...

    def labelled_count(self) -> int: ...


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _as_utc_iso(value: Any) -> str:
    """ISO-8601 en UTC; un datetime naive se interpreta como UTC."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return _as_utc_iso(datetime.fromisoformat(str(value)))


def _features_of(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in _NON_FEATURE_COLUMNS}


def _labelled_record(
    features: dict,
    score: float,
    version: str,
    prediction_id: str,
    predicted_at: str,
    observed_at: str,
    labelled_at: str,
    churn: int,
) -> dict:
    """Fila de `labelled()`: features + probabilidad + version + ids/tiempos + etiqueta."""
    return {
        **features,
        SCORE_COLUMN: score,
        VERSION_COLUMN: version,
        ID_COLUMN: prediction_id,
        PREDICTED_AT_COLUMN: predicted_at,
        OBSERVED_AT_COLUMN: observed_at,
        LABELLED_AT_COLUMN: labelled_at,
        LABEL_COLUMN: churn,
    }


class MemoryPredictionStore:
    def __init__(self, max_rows: int) -> None:
        # (numero de secuencia, fila): la secuencia ordena las predicciones aunque un lote
        # entero comparta el mismo predicted_at
        self._rows: deque[tuple[int, dict]] = deque(maxlen=max(1, max_rows))
        self._labels: dict[str, tuple[int, dict]] = {}
        self._seq = itertools.count()
        self._lock = threading.Lock()

    def append(self, rows: Iterable[dict]) -> list[str]:
        now = _now()
        ids: list[str] = []
        with self._lock:
            for row in rows:
                prediction_id = uuid4().hex
                stored = {**row, ID_COLUMN: prediction_id, PREDICTED_AT_COLUMN: now}
                self._rows.append((next(self._seq), stored))
                ids.append(prediction_id)
        return ids

    def recent(self, limit: int) -> pd.DataFrame:
        with self._lock:
            rows = [row for _, row in list(self._rows)[-max(0, limit) :]]
        return pd.DataFrame(rows)

    def count(self) -> int:
        return len(self._rows)

    def label(self, labels: Iterable[dict]) -> LabelOutcome:
        now = _now()
        outcome = LabelOutcome()
        with self._lock:
            by_id = {row[ID_COLUMN]: (seq, row) for seq, row in self._rows}
            for item in labels:
                prediction_id = str(item[ID_COLUMN])
                found = by_id.get(prediction_id)
                if found is None:
                    outcome.unknown.append(prediction_id)
                    continue
                seq, row = found
                if prediction_id in self._labels:
                    outcome.updated += 1
                else:
                    outcome.created += 1
                record = _labelled_record(
                    _features_of(row),
                    float(row[SCORE_COLUMN]),
                    str(row[VERSION_COLUMN]),
                    prediction_id,
                    str(row[PREDICTED_AT_COLUMN]),
                    _as_utc_iso(item[OBSERVED_AT_COLUMN]),
                    now,
                    int(item[LABEL_COLUMN]),
                )
                self._labels[prediction_id] = (seq, record)
        return outcome

    def labelled(self, limit: int | None = None) -> pd.DataFrame:
        with self._lock:
            ordered = sorted(self._labels.values(), key=lambda item: item[0])
        if limit is not None:
            ordered = ordered[max(0, len(ordered) - limit) :]  # las mas recientes (0 = ninguna)
        return pd.DataFrame([record for _, record in ordered])

    def labelled_count(self) -> int:
        return len(self._labels)


class SqlitePredictionStore:
    def __init__(self, path: str | Path, max_rows: int) -> None:
        self.path = Path(path)
        self.max_rows = max(1, max_rows)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            self._ensure_schema(conn)

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS predictions ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts TEXT NOT NULL,"
            " model_version TEXT NOT NULL,"
            " churn_probability REAL NOT NULL,"
            " features TEXT NOT NULL,"
            " prediction_id TEXT)"
        )
        # Ficheros anteriores al bucle de etiquetas: se anade la columna; sus filas quedan
        # con prediction_id NULL (no etiquetables) y se desalojan con la ventana.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        if ID_COLUMN not in columns:
            conn.execute("ALTER TABLE predictions ADD COLUMN prediction_id TEXT")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS predictions_prediction_id"
            " ON predictions(prediction_id)"
        )
        # Copia del ejemplo etiquetado; prediction_seq = predictions.id, el orden real de
        # las predicciones (un lote entero comparte predicted_at).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS labels ("
            " prediction_id TEXT PRIMARY KEY,"
            " prediction_seq INTEGER NOT NULL,"
            " churn INTEGER NOT NULL,"
            " observed_at TEXT NOT NULL,"
            " labelled_at TEXT NOT NULL,"
            " predicted_at TEXT NOT NULL,"
            " model_version TEXT NOT NULL,"
            " churn_probability REAL NOT NULL,"
            " features TEXT NOT NULL)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS labels_prediction_seq ON labels(prediction_seq)")

    def _connect(self) -> sqlite3.Connection:
        # Una conexion por operacion: sin estado compartido entre hilos, WAL para lectores
        # concurrentes y timeout para esperar a otros procesos que escriban a la vez.
        return sqlite3.connect(self.path, timeout=10)

    def append(self, rows: Iterable[dict]) -> list[str]:
        now = _now()
        payload = []
        ids: list[str] = []
        for row in rows:
            prediction_id = uuid4().hex
            ids.append(prediction_id)
            payload.append(
                (
                    now,
                    str(row[VERSION_COLUMN]),
                    float(row[SCORE_COLUMN]),
                    json.dumps(_features_of(row), ensure_ascii=False, default=str),
                    prediction_id,
                )
            )
        if not payload:
            return ids
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO predictions"
                " (ts, model_version, churn_probability, features, prediction_id)"
                " VALUES (?, ?, ?, ?, ?)",
                payload,
            )
            # Ventana rodante: borra todo lo anterior a las `max_rows` filas mas recientes.
            conn.execute(
                "DELETE FROM predictions WHERE id <= ("
                " SELECT id FROM predictions ORDER BY id DESC LIMIT 1 OFFSET ?)",
                (self.max_rows,),
            )
        return ids

    def recent(self, limit: int) -> pd.DataFrame:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT features, churn_probability, model_version, prediction_id, ts"
                " FROM predictions ORDER BY id DESC LIMIT ?",
                (max(0, limit),),
            )
            fetched = cursor.fetchall()
        records = [
            {
                **json.loads(features),
                SCORE_COLUMN: score,
                VERSION_COLUMN: version,
                ID_COLUMN: prediction_id,
                PREDICTED_AT_COLUMN: ts,
            }
            for features, score, version, prediction_id, ts in reversed(fetched)  # cronologico
        ]
        return pd.DataFrame(records)

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def label(self, labels: Iterable[dict]) -> LabelOutcome:
        now = _now()
        outcome = LabelOutcome()
        with self._lock, self._connect() as conn:
            for item in labels:
                prediction_id = str(item[ID_COLUMN])
                found = conn.execute(
                    "SELECT id, ts, model_version, churn_probability, features"
                    " FROM predictions WHERE prediction_id = ?",
                    (prediction_id,),
                ).fetchone()
                if found is None:
                    outcome.unknown.append(prediction_id)
                    continue
                seq, predicted_at, version, score, features = found
                already = conn.execute(
                    "SELECT 1 FROM labels WHERE prediction_id = ?", (prediction_id,)
                ).fetchone()
                if already:
                    outcome.updated += 1
                else:
                    outcome.created += 1
                conn.execute(
                    "INSERT OR REPLACE INTO labels (prediction_id, prediction_seq, churn,"
                    " observed_at, labelled_at, predicted_at, model_version,"
                    " churn_probability, features) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        prediction_id,
                        seq,
                        int(item[LABEL_COLUMN]),
                        _as_utc_iso(item[OBSERVED_AT_COLUMN]),
                        now,
                        predicted_at,
                        version,
                        score,
                        features,
                    ),
                )
        return outcome

    def labelled(self, limit: int | None = None) -> pd.DataFrame:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT features, churn_probability, model_version, prediction_id,"
                " predicted_at, observed_at, labelled_at, churn FROM labels"
                " ORDER BY prediction_seq DESC LIMIT ?",
                (-1 if limit is None else max(0, limit),),  # -1 = sin limite en SQLite
            )
            fetched = cursor.fetchall()
        records = [
            _labelled_record(json.loads(features), score, version, *rest)
            for features, score, version, *rest in reversed(fetched)  # orden de prediccion
        ]
        return pd.DataFrame(records)

    def labelled_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0])


def build_prediction_store(settings: Settings) -> PredictionStore:
    if settings.drift_store == "memory":
        return MemoryPredictionStore(max_rows=settings.prediction_keep_rows)
    return SqlitePredictionStore(
        path=default_db_path(settings), max_rows=settings.prediction_keep_rows
    )


def default_db_path(settings: Settings, model_dir: str | Path | None = None) -> Path:
    """Fichero SQLite de predicciones: `drift_db_path` o `<model_dir>/drift.sqlite`."""
    if settings.drift_db_path:
        return Path(settings.drift_db_path)
    return Path(model_dir or settings.model_dir) / DEFAULT_DB_NAME
