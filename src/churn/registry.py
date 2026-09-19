"""Almacen de artefactos del modelo.

Contrato de artefactos en `model_dir`:
  - model.joblib      pipeline sklearn completo (preprocesado + clasificador)
  - metadata.json     version, metricas, esquema de features, timestamps
  - reference.csv     muestra de datos de entrenamiento para monitorizar drift

Los tres ficheros se escriben primero en un directorio de staging y despues se
mueven a su sitio con `os.replace` (atomico por fichero en el mismo sistema de
ficheros). Asi un reload de la API o un fallo a mitad de escritura nunca ve un
fichero truncado. El modelo se mueve el ultimo porque `exists()` lo usa como senal
de "artefactos completos".
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REFERENCE_FILE = "reference.csv"
ARTIFACT_FILES = (METADATA_FILE, REFERENCE_FILE, MODEL_FILE)  # el modelo, el ultimo
REQUIRED_METADATA_KEYS = ("model_version", "numeric_features", "categorical_features")


class ModelArtifactError(RuntimeError):
    """Artefactos ausentes, incompletos o corruptos en el directorio del modelo."""


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Any
    metadata: dict[str, Any]
    reference: pd.DataFrame

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata["numeric_features"]) + list(self.metadata["categorical_features"])


class LocalModelStore:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)

    @property
    def model_path(self) -> Path:
        return self.model_dir / MODEL_FILE

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / METADATA_FILE

    @property
    def reference_path(self) -> Path:
        return self.model_dir / REFERENCE_FILE

    def missing_files(self) -> list[str]:
        return [name for name in ARTIFACT_FILES if not (self.model_dir / name).exists()]

    def exists(self) -> bool:
        """True solo si el juego de artefactos esta completo."""
        return not self.missing_files()

    def save(self, pipeline: Any, metadata: dict[str, Any], reference: pd.DataFrame) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.model_dir))
        try:
            joblib.dump(pipeline, staging / MODEL_FILE)
            _write_json(staging / METADATA_FILE, metadata)
            reference.to_csv(staging / REFERENCE_FILE, index=False)
            for name in ARTIFACT_FILES:
                os.replace(staging / name, self.model_dir / name)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        """Reescribe solo metadata.json de forma atomica (p. ej. para anadir el run de MLflow)."""
        tmp = self.metadata_path.with_suffix(".json.tmp")
        _write_json(tmp, metadata)
        os.replace(tmp, self.metadata_path)

    def load(self) -> LoadedModel:
        missing = self.missing_files()
        if missing:
            raise FileNotFoundError(
                f"Artefactos ausentes en {self.model_dir}: {missing}. Ejecuta antes el "
                "entrenamiento (python -m churn.training.train)."
            )
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelArtifactError(f"metadata.json ilegible en {self.model_dir}: {exc}") from exc
        if not isinstance(metadata, dict) or any(k not in metadata for k in REQUIRED_METADATA_KEYS):
            raise ModelArtifactError(
                f"metadata.json incompleto: se esperan las claves {list(REQUIRED_METADATA_KEYS)}"
            )
        try:
            pipeline = joblib.load(self.model_path)
        except Exception as exc:  # joblib/pickle lanzan tipos muy variados
            raise ModelArtifactError(f"model.joblib corrupto en {self.model_dir}: {exc}") from exc
        try:
            reference = pd.read_csv(self.reference_path)
        except (OSError, ValueError) as exc:
            raise ModelArtifactError(f"reference.csv ilegible en {self.model_dir}: {exc}") from exc
        return LoadedModel(pipeline=pipeline, metadata=metadata, reference=reference)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
