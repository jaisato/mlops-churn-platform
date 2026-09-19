"""Almacen de predicciones recientes para monitorizar drift.

Dos backends con el mismo contrato:
  - SqlitePredictionStore: fichero SQLite (modo WAL) en el volumen. Sobrevive a los
    reinicios y lo comparten todos los workers/procesos de la API.
  - MemoryPredictionStore: deque en memoria. Para tests y demos de un solo proceso.

Cada fila guarda las features recibidas, la probabilidad devuelta y la version del
modelo que la produjo; el tamano se acota a `max_rows` (ventana rodante).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import deque
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from churn.config import Settings

SCORE_COLUMN = "churn_probability"
VERSION_COLUMN = "model_version"
DEFAULT_DB_NAME = "drift.sqlite"


class PredictionStore(Protocol):
    def append(self, rows: Iterable[dict]) -> None: ...

    def recent(self, limit: int) -> pd.DataFrame: ...

    def count(self) -> int: ...


class MemoryPredictionStore:
    def __init__(self, max_rows: int) -> None:
        self._rows: deque[dict] = deque(maxlen=max(1, max_rows))
        self._lock = threading.Lock()

    def append(self, rows: Iterable[dict]) -> None:
        with self._lock:
            self._rows.extend(dict(row) for row in rows)

    def recent(self, limit: int) -> pd.DataFrame:
        with self._lock:
            rows = list(self._rows)[-max(0, limit) :]
        return pd.DataFrame(rows)

    def count(self) -> int:
        return len(self._rows)


class SqlitePredictionStore:
    def __init__(self, path: str | Path, max_rows: int) -> None:
        self.path = Path(path)
        self.max_rows = max(1, max_rows)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS predictions ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " ts TEXT NOT NULL,"
                " model_version TEXT NOT NULL,"
                " churn_probability REAL NOT NULL,"
                " features TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        # Una conexion por operacion: sin estado compartido entre hilos, WAL para lectores
        # concurrentes y timeout para esperar a otros procesos que escriban a la vez.
        return sqlite3.connect(self.path, timeout=10)

    def append(self, rows: Iterable[dict]) -> None:
        now = datetime.now(UTC).isoformat()
        payload = []
        for row in rows:
            features = {k: v for k, v in row.items() if k not in (SCORE_COLUMN, VERSION_COLUMN)}
            payload.append(
                (
                    now,
                    str(row[VERSION_COLUMN]),
                    float(row[SCORE_COLUMN]),
                    json.dumps(features, ensure_ascii=False, default=str),
                )
            )
        if not payload:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO predictions (ts, model_version, churn_probability, features)"
                " VALUES (?, ?, ?, ?)",
                payload,
            )
            # Ventana rodante: borra todo lo anterior a las `max_rows` filas mas recientes.
            conn.execute(
                "DELETE FROM predictions WHERE id <= ("
                " SELECT id FROM predictions ORDER BY id DESC LIMIT 1 OFFSET ?)",
                (self.max_rows,),
            )

    def recent(self, limit: int) -> pd.DataFrame:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT features, churn_probability, model_version FROM predictions"
                " ORDER BY id DESC LIMIT ?",
                (max(0, limit),),
            )
            fetched = cursor.fetchall()
        records = [
            {**json.loads(features), SCORE_COLUMN: score, VERSION_COLUMN: version}
            for features, score, version in reversed(fetched)  # orden cronologico
        ]
        return pd.DataFrame(records)

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])


def build_prediction_store(settings: Settings) -> PredictionStore:
    if settings.drift_store == "memory":
        return MemoryPredictionStore(max_rows=settings.drift_buffer_size)
    path = settings.drift_db_path or str(Path(settings.model_dir) / DEFAULT_DB_NAME)
    return SqlitePredictionStore(path=path, max_rows=settings.drift_buffer_size)
