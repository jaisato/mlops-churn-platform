"""Almacen de predicciones recientes (drift) y de sus etiquetas reales (bucle de etiquetas).

Dos backends con el mismo contrato:
  - SqlitePredictionStore: fichero SQLite (modo WAL) en el volumen. Sobrevive a los
    reinicios y lo comparten todos los workers/procesos de la API y el trainer.
  - MemoryPredictionStore: estructuras en memoria. Para tests y demos de un solo proceso.

Predicciones: cada fila guarda las features recibidas, la probabilidad devuelta, la
version del modelo que la produjo, un `prediction_id` que la API devuelve al cliente y,
si el cliente la envia, una clave opaca del sujeto (`subject_ref`: el mismo cliente
puntuado varias veces). El tamano se acota a `max_rows` (las mas antiguas se desalojan) y,
opcionalmente, a `max_age` (antiguedad maxima); `recent(limit)` es la ventana que mira el
drift.

Etiquetas: `label()` recibe el `churn` real de una prediccion y guarda una COPIA del
ejemplo (features + probabilidad + version + sujeto + etiqueta) en una tabla aparte, de modo
que el ejemplo etiquetado sobrevive al desalojo de la ventana de predicciones y
`labelled()` devuelve directamente el historial de reentreno, en el orden en que se
hicieron las predicciones. Reetiquetar sobrescribe (la ultima etiqueta recibida es la que
vale), tambien cuando la prediccion ya salio de la ventana: la copia sigue en `labels`. Una
etiqueta cuyo `observed_at` es anterior a la prediccion se rechaza (`rejected`): el
resultado no puede observarse antes de predecirlo.
"""

from __future__ import annotations

import itertools
import json
import sqlite3
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import pandas as pd

from churn.config import Settings

SCORE_COLUMN = "churn_probability"
VERSION_COLUMN = "model_version"
ID_COLUMN = "prediction_id"
SUBJECT_COLUMN = "subject_ref"
PREDICTED_AT_COLUMN = "predicted_at"
OBSERVED_AT_COLUMN = "observed_at"
LABELLED_AT_COLUMN = "labelled_at"
LABEL_COLUMN = "churn"
DEFAULT_DB_NAME = "drift.sqlite"
#: Columnas de una fila de prediccion (o de etiqueta) que no son features del cliente.
_NON_FEATURE_COLUMNS = frozenset(
    {
        SCORE_COLUMN,
        VERSION_COLUMN,
        ID_COLUMN,
        SUBJECT_COLUMN,
        PREDICTED_AT_COLUMN,
        OBSERVED_AT_COLUMN,
        LABELLED_AT_COLUMN,
        LABEL_COLUMN,
    }
)


@dataclass
class LabelOutcome:
    """Resultado de un lote de etiquetas: cuantas eran nuevas, cuantas corrigen una
    etiqueta anterior, que `prediction_id` no estan en el almacen (ni como prediccion ni
    como etiqueta) y cuales se rechazan por tener `observed_at` anterior a la prediccion."""

    created: int = 0
    updated: int = 0
    unknown: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


class PredictionStore(Protocol):
    def append(self, rows: Iterable[dict]) -> list[str]: ...

    def recent(self, limit: int) -> pd.DataFrame: ...

    def count(self) -> int: ...

    def label(self, labels: Iterable[dict]) -> LabelOutcome: ...

    def labelled(self, limit: int | None = None) -> pd.DataFrame: ...

    def labelled_count(self) -> int: ...


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _as_utc_iso(value: Any) -> str:
    """ISO-8601 en UTC; un datetime naive se interpreta como UTC."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return _as_utc_iso(datetime.fromisoformat(str(value)))


def _observed_before_prediction(observed_at: str, predicted_at: str) -> bool:
    return datetime.fromisoformat(observed_at) < datetime.fromisoformat(predicted_at)


def _subject_of(value: Any) -> str | None:
    return None if value is None else str(value)


def _features_of(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in _NON_FEATURE_COLUMNS}


def _labelled_record(
    features: dict,
    score: float,
    version: str,
    prediction_id: str,
    subject_ref: str | None,
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
        SUBJECT_COLUMN: subject_ref,
        PREDICTED_AT_COLUMN: predicted_at,
        OBSERVED_AT_COLUMN: observed_at,
        LABELLED_AT_COLUMN: labelled_at,
        LABEL_COLUMN: churn,
    }


class MemoryPredictionStore:
    def __init__(self, max_rows: int, max_age: timedelta | None = None) -> None:
        # (numero de secuencia, fila): la secuencia ordena las predicciones aunque un lote
        # entero comparta el mismo predicted_at
        self._rows: deque[tuple[int, dict]] = deque(maxlen=max(1, max_rows))
        self._labels: dict[str, tuple[int, dict]] = {}
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self.max_age = max_age

    def append(self, rows: Iterable[dict]) -> list[str]:
        now = _utcnow()
        stamp = now.isoformat()
        ids: list[str] = []
        with self._lock:
            for row in rows:
                prediction_id = uuid4().hex
                stored = {  # mismo formato que una fila de SqlitePredictionStore.recent()
                    **_features_of(row),
                    SCORE_COLUMN: row[SCORE_COLUMN],
                    VERSION_COLUMN: row[VERSION_COLUMN],
                    ID_COLUMN: prediction_id,
                    SUBJECT_COLUMN: _subject_of(row.get(SUBJECT_COLUMN)),
                    PREDICTED_AT_COLUMN: stamp,
                }
                self._rows.append((next(self._seq), stored))
                ids.append(prediction_id)
            if self.max_age is not None:
                cutoff = (now - self.max_age).isoformat()
                while self._rows and self._rows[0][1][PREDICTED_AT_COLUMN] < cutoff:
                    self._rows.popleft()
        return ids

    def recent(self, limit: int) -> pd.DataFrame:
        with self._lock:
            rows = [row for _, row in list(self._rows)[-max(0, limit) :]]
        return pd.DataFrame(rows)

    def count(self) -> int:
        return len(self._rows)

    def label(self, labels: Iterable[dict]) -> LabelOutcome:
        now = _utcnow().isoformat()
        outcome = LabelOutcome()
        with self._lock:
            by_id = {row[ID_COLUMN]: (seq, row) for seq, row in self._rows}
            for item in labels:
                prediction_id = str(item[ID_COLUMN])
                observed_at = _as_utc_iso(item[OBSERVED_AT_COLUMN])
                subject = _subject_of(item.get(SUBJECT_COLUMN))
                existing = self._labels.get(prediction_id)
                previous = existing[1] if existing else None
                found = by_id.get(prediction_id)
                if found is not None:
                    seq, row = found
                    predicted_at = str(row[PREDICTED_AT_COLUMN])
                elif existing is not None:
                    # Prediccion ya desalojada de la ventana: se corrige la copia etiquetada
                    seq, row = existing
                    predicted_at = str(row[PREDICTED_AT_COLUMN])
                else:
                    outcome.unknown.append(prediction_id)
                    continue
                if _observed_before_prediction(observed_at, predicted_at):
                    outcome.rejected.append(prediction_id)
                    continue
                if previous is None:
                    outcome.created += 1
                else:
                    outcome.updated += 1
                self._labels[prediction_id] = (
                    seq,
                    _labelled_record(
                        _features_of(row),
                        float(row[SCORE_COLUMN]),
                        str(row[VERSION_COLUMN]),
                        prediction_id,
                        # la etiqueta manda; si no trae sujeto, el que ya tuviera
                        subject or (previous or {}).get(SUBJECT_COLUMN) or row[SUBJECT_COLUMN],
                        predicted_at,
                        observed_at,
                        now,
                        int(item[LABEL_COLUMN]),
                    ),
                )
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
    def __init__(self, path: str | Path, max_rows: int, max_age: timedelta | None = None) -> None:
        self.path = Path(path)
        self.max_rows = max(1, max_rows)
        self.max_age = max_age
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _migrate(self) -> None:
        """Crea o actualiza el esquema de forma atomica entre procesos.

        Varios workers arrancan a la vez sobre el mismo fichero: comprobar las columnas y
        despues hacer `ALTER TABLE` en dos pasos dejaba que dos procesos vieran la columna
        ausente y el segundo fallara con "duplicate column name". `BEGIN IMMEDIATE` toma el
        cerrojo de escritura antes de mirar el esquema, asi que el segundo espera (timeout)
        y, al entrar, ya ve la columna anadida.
        """
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)  # transaccion manual
        try:
            conn.execute("PRAGMA journal_mode=WAL")  # fuera de la transaccion (SQLite lo exige)
            with conn:  # COMMIT al salir; ROLLBACK (esquema intacto) si algo falla
                conn.execute("BEGIN IMMEDIATE")
                self._ensure_schema(conn)
        finally:
            conn.close()

    @staticmethod
    def _add_missing_columns(conn: sqlite3.Connection, table: str, columns: Iterable[str]) -> None:
        present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column in columns:
            if column not in present:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")

    @classmethod
    def _ensure_schema(cls, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS predictions ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts TEXT NOT NULL,"
            " model_version TEXT NOT NULL,"
            " churn_probability REAL NOT NULL,"
            " features TEXT NOT NULL,"
            " prediction_id TEXT,"
            " subject_ref TEXT)"
        )
        # Ficheros anteriores al bucle de etiquetas: se anaden las columnas; sus filas quedan
        # con prediction_id NULL (no etiquetables) y se desalojan con la ventana.
        cls._add_missing_columns(conn, "predictions", (ID_COLUMN, SUBJECT_COLUMN))
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
            " features TEXT NOT NULL,"
            " subject_ref TEXT)"
        )
        cls._add_missing_columns(conn, "labels", (SUBJECT_COLUMN,))
        conn.execute("CREATE INDEX IF NOT EXISTS labels_prediction_seq ON labels(prediction_seq)")

    def _connect(self) -> sqlite3.Connection:
        # Una conexion por operacion: sin estado compartido entre hilos, WAL para lectores
        # concurrentes y timeout para esperar a otros procesos que escriban a la vez.
        return sqlite3.connect(self.path, timeout=10)

    def append(self, rows: Iterable[dict]) -> list[str]:
        now = _utcnow()
        stamp = now.isoformat()
        payload = []
        ids: list[str] = []
        for row in rows:
            prediction_id = uuid4().hex
            ids.append(prediction_id)
            payload.append(
                (
                    stamp,
                    str(row[VERSION_COLUMN]),
                    float(row[SCORE_COLUMN]),
                    json.dumps(_features_of(row), ensure_ascii=False, default=str),
                    prediction_id,
                    _subject_of(row.get(SUBJECT_COLUMN)),
                )
            )
        if not payload:
            return ids
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO predictions"
                " (ts, model_version, churn_probability, features, prediction_id, subject_ref)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                payload,
            )
            # Ventana rodante: los ids (AUTOINCREMENT) crecen de uno en uno, asi que las
            # `max_rows` mas recientes son las de id > MAX(id) - max_rows. MAX(id) es O(1) y el
            # DELETE recorre solo las filas desalojadas (antes, `ORDER BY id DESC OFFSET
            # max_rows` recorria toda la ventana en cada prediccion).
            conn.execute(
                "DELETE FROM predictions WHERE id <= (SELECT MAX(id) FROM predictions) - ?",
                (self.max_rows,),
            )
            if self.max_age is not None:
                # Antiguedad maxima: las predicciones entran en orden de id y de ts, asi que
                # basta con encontrar la primera vigente (recorre solo las caducadas).
                conn.execute(
                    "DELETE FROM predictions WHERE id < ("
                    " SELECT id FROM predictions WHERE ts >= ? ORDER BY id LIMIT 1)",
                    ((now - self.max_age).isoformat(),),
                )
        return ids

    def recent(self, limit: int) -> pd.DataFrame:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT features, churn_probability, model_version, prediction_id, subject_ref,"
                " ts FROM predictions ORDER BY id DESC LIMIT ?",
                (max(0, limit),),
            )
            fetched = cursor.fetchall()
        records = [
            {
                **json.loads(features),
                SCORE_COLUMN: score,
                VERSION_COLUMN: version,
                ID_COLUMN: prediction_id,
                SUBJECT_COLUMN: subject,
                PREDICTED_AT_COLUMN: ts,
            }
            for features, score, version, prediction_id, subject, ts in reversed(fetched)
        ]  # orden cronologico
        return pd.DataFrame(records)

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def label(self, labels: Iterable[dict]) -> LabelOutcome:
        now = _utcnow().isoformat()
        outcome = LabelOutcome()
        with self._lock, self._connect() as conn:
            for item in labels:
                prediction_id = str(item[ID_COLUMN])
                observed_at = _as_utc_iso(item[OBSERVED_AT_COLUMN])
                subject = _subject_of(item.get(SUBJECT_COLUMN))
                found = conn.execute(
                    "SELECT id, ts, model_version, churn_probability, features, subject_ref"
                    " FROM predictions WHERE prediction_id = ?",
                    (prediction_id,),
                ).fetchone()
                existing = conn.execute(
                    "SELECT predicted_at, subject_ref FROM labels WHERE prediction_id = ?",
                    (prediction_id,),
                ).fetchone()
                if found is None and existing is None:
                    outcome.unknown.append(prediction_id)
                    continue
                predicted_at = found[1] if found else existing[0]
                if _observed_before_prediction(observed_at, predicted_at):
                    outcome.rejected.append(prediction_id)
                    continue
                if found is None:
                    # Prediccion ya desalojada de la ventana: la copia etiquetada sigue en
                    # `labels`, asi que una correccion tardia actualiza esa fila.
                    cursor = conn.execute(
                        "UPDATE labels SET churn = ?, observed_at = ?, labelled_at = ?,"
                        " subject_ref = COALESCE(?, subject_ref) WHERE prediction_id = ?",
                        (int(item[LABEL_COLUMN]), observed_at, now, subject, prediction_id),
                    )
                    outcome.updated += cursor.rowcount
                    continue
                seq, predicted_at, version, score, features, predicted_subject = found
                if existing:
                    outcome.updated += 1
                else:
                    outcome.created += 1
                conn.execute(
                    "INSERT OR REPLACE INTO labels (prediction_id, prediction_seq, churn,"
                    " observed_at, labelled_at, predicted_at, model_version,"
                    " churn_probability, features, subject_ref)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        prediction_id,
                        seq,
                        int(item[LABEL_COLUMN]),
                        observed_at,
                        now,
                        predicted_at,
                        version,
                        score,
                        features,
                        subject or (existing[1] if existing else None) or predicted_subject,
                    ),
                )
        return outcome

    def labelled(self, limit: int | None = None) -> pd.DataFrame:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT features, churn_probability, model_version, prediction_id, subject_ref,"
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


def max_age_of(settings: Settings) -> timedelta | None:
    """Antiguedad maxima de las predicciones (`CHURN_PREDICTION_KEEP_DAYS`; 0 = sin limite)."""
    return timedelta(days=settings.prediction_keep_days) if settings.prediction_keep_days else None


def build_prediction_store(settings: Settings) -> PredictionStore:
    max_age = max_age_of(settings)
    if settings.drift_store == "memory":
        return MemoryPredictionStore(max_rows=settings.prediction_keep_rows, max_age=max_age)
    return SqlitePredictionStore(
        path=default_db_path(settings), max_rows=settings.prediction_keep_rows, max_age=max_age
    )


def default_db_path(settings: Settings, model_dir: str | Path | None = None) -> Path:
    """Fichero SQLite de predicciones: `drift_db_path` o `<model_dir>/drift.sqlite`."""
    if settings.drift_db_path:
        return Path(settings.drift_db_path)
    return Path(model_dir or settings.model_dir) / DEFAULT_DB_NAME
