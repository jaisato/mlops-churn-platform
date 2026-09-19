"""Almacen de artefactos del modelo, versionado y con rollback.

Layout (a partir de la 1.1):

    model_dir/
      current                      nombre de la version en servicio (fichero de texto)
      versions/<model_version>/    un directorio inmutable por entrenamiento
        model.joblib               pipeline sklearn completo (preprocesado + clasificador)
        metadata.json              version, metricas, esquema de features, runtime, timestamps
        reference.csv              muestra de datos de entrenamiento para monitorizar drift

Publicar una version = escribir su directorio en staging, renombrarlo (atomico) y
reemplazar el puntero `current` (atomico): un reload nunca ve un juego a medias.
Rollback = apuntar `current` a otra version ya existente. Las versiones antiguas se
podan segun `keep_versions`, sin tocar nunca la que esta en servicio.

Compatibilidad: si no hay puntero pero existen los tres ficheros sueltos en
`model_dir` (layout plano de la 1.0), se cargan tal cual; el siguiente
entrenamiento ya publica en el layout versionado.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from churn.data.generator import FEATURE_COLUMNS

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REFERENCE_FILE = "reference.csv"
ARTIFACT_FILES = (METADATA_FILE, REFERENCE_FILE, MODEL_FILE)
REQUIRED_METADATA_KEYS = ("model_version", "numeric_features", "categorical_features")
CURRENT_FILE = "current"
VERSIONS_DIR = "versions"
DEFAULT_KEEP_VERSIONS = 5


class ModelArtifactError(RuntimeError):
    """Artefactos ausentes, incompletos o corruptos en el directorio del modelo."""


class ModelCompatibilityError(ModelArtifactError):
    """Los artefactos existen pero este codigo/runtime no puede servirlos con garantias."""


def _minor(version: str) -> str:
    return ".".join(str(version).split(".")[:2])


def verify_compatibility(
    metadata: dict[str, Any],
    *,
    strict_runtime: bool = True,
    expected_features: list[str] | None = None,
) -> list[str]:
    """Comprueba que el modelo puede servirse con el codigo y el runtime actuales.

    Devuelve la lista de avisos (no bloqueantes). Lanza `ModelCompatibilityError` si las
    features del modelo no son las que espera el contrato de la API, o si (en modo
    estricto) el pipeline se serializo con otra version menor de scikit-learn: los
    pickles no son portables entre versiones y el fallo seria silencioso.
    """
    expected = sorted(expected_features or FEATURE_COLUMNS)
    actual = sorted(
        list(metadata.get("numeric_features", [])) + list(metadata.get("categorical_features", []))
    )
    if actual != expected:
        raise ModelCompatibilityError(
            f"Las features del modelo {actual} no coinciden con las del codigo {expected}: "
            "reentrena con esta version del codigo"
        )

    warnings: list[str] = []
    runtime = metadata.get("runtime") or {}
    if not runtime:
        warnings.append(
            "metadata sin versiones de runtime (modelo anterior a la 1.1): no se puede "
            "comprobar la compatibilidad de scikit-learn"
        )
        return warnings

    trained_sklearn = str(runtime.get("scikit_learn", ""))
    current_sklearn = sklearn.__version__
    if _minor(trained_sklearn) != _minor(current_sklearn):
        message = (
            f"scikit-learn del modelo ({trained_sklearn}) distinto del instalado "
            f"({current_sklearn}); los pickles no son portables entre versiones menores"
        )
        if strict_runtime:
            raise ModelCompatibilityError(
                message + ". Reentrena o arranca con CHURN_STRICT_ARTIFACT_COMPAT=false"
            )
        warnings.append(message)
    elif trained_sklearn != current_sklearn:
        warnings.append(
            f"scikit-learn con distinto parche: modelo {trained_sklearn}, "
            f"instalado {current_sklearn}"
        )

    trained_python = str(runtime.get("python", ""))
    if trained_python and _minor(trained_python) != _minor(platform.python_version()):
        warnings.append(
            f"Python del modelo ({trained_python}) distinto del actual "
            f"({platform.python_version()})"
        )
    return warnings


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
    def __init__(self, model_dir: str | Path, keep_versions: int = DEFAULT_KEEP_VERSIONS) -> None:
        self.model_dir = Path(model_dir)
        self.keep_versions = max(1, int(keep_versions))

    # ------------------------------------------------------------------ layout

    @property
    def versions_dir(self) -> Path:
        return self.model_dir / VERSIONS_DIR

    @property
    def current_file(self) -> Path:
        return self.model_dir / CURRENT_FILE

    def version_dir(self, version: str) -> Path:
        return self.versions_dir / version

    def current_version(self) -> str | None:
        """Version apuntada por `current`, o None si no hay puntero (vacio o layout plano)."""
        try:
            text = self.current_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return text or None

    def list_versions(self) -> list[str]:
        """Versiones publicadas, de la mas antigua a la mas reciente.

        Se ordenan por `trained_at` del metadata (con microsegundos) y no por nombre: dos
        entrenamientos en el mismo segundo comparten prefijo de fecha y el sufijo aleatorio
        no dice nada del orden. Sobrevive a copias/restauraciones que pierden los mtimes.
        """
        if not self.versions_dir.is_dir():
            return []
        names = [
            p.name for p in self.versions_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        ]
        return sorted(names, key=lambda name: (self._trained_at(name), name))

    def _trained_at(self, version: str) -> str:
        metadata_file = self.version_dir(version) / METADATA_FILE
        try:
            meta = json.loads(metadata_file.read_text(encoding="utf-8"))
            return str(meta.get("trained_at") or "")
        except (OSError, ValueError, AttributeError):
            return ""

    def has_legacy_layout(self) -> bool:
        return all((self.model_dir / name).exists() for name in ARTIFACT_FILES)

    def resolve_dir(self, version: str | None = None) -> Path | None:
        """Directorio con los artefactos de `version` (o de la version en servicio)."""
        if version is not None:
            return self.version_dir(version)
        current = self.current_version()
        if current:
            return self.version_dir(current)
        if self.has_legacy_layout():
            return self.model_dir
        return None

    def missing_files(self, version: str | None = None) -> list[str]:
        directory = self.resolve_dir(version)
        if directory is None:
            return list(ARTIFACT_FILES)
        return [name for name in ARTIFACT_FILES if not (directory / name).exists()]

    def exists(self, version: str | None = None) -> bool:
        """True solo si el juego de artefactos de esa version (o la actual) esta completo."""
        return not self.missing_files(version)

    def _path_of(self, name: str) -> Path:
        directory = self.resolve_dir()
        return (directory if directory is not None else self.model_dir) / name

    @property
    def model_path(self) -> Path:
        return self._path_of(MODEL_FILE)

    @property
    def metadata_path(self) -> Path:
        return self._path_of(METADATA_FILE)

    @property
    def reference_path(self) -> Path:
        return self._path_of(REFERENCE_FILE)

    # ------------------------------------------------------------------ escritura

    def save(self, pipeline: Any, metadata: dict[str, Any], reference: pd.DataFrame) -> str:
        """Publica una version nueva y la deja en servicio. Devuelve el nombre de la version."""
        version = str(metadata["model_version"])
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.versions_dir))
        try:
            joblib.dump(pipeline, staging / MODEL_FILE)
            _write_json(staging / METADATA_FILE, metadata)
            reference.to_csv(staging / REFERENCE_FILE, index=False)
            target = self.version_dir(version)
            if target.exists():
                shutil.rmtree(target)
            os.replace(staging, target)  # renombrar el directorio es atomico
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        self.set_current(version)
        self.prune()
        return version

    def set_current(self, version: str) -> None:
        """Cambia el puntero de la version en servicio de forma atomica."""
        tmp = self.current_file.with_suffix(".tmp")
        tmp.write_text(version + "\n", encoding="utf-8")
        os.replace(tmp, self.current_file)

    def prune(self) -> list[str]:
        """Borra las versiones mas antiguas que exceden `keep_versions`; nunca la actual."""
        versions = self.list_versions()
        keep = set(versions[-self.keep_versions :])
        current = self.current_version()
        if current:
            keep.add(current)
        removed = [v for v in versions if v not in keep]
        for version in removed:
            shutil.rmtree(self.version_dir(version), ignore_errors=True)
        return removed

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        """Reescribe metadata.json de esa version de forma atomica (p. ej. tras el run MLflow)."""
        directory = self.version_dir(str(metadata["model_version"]))
        if not directory.is_dir() and self.has_legacy_layout():
            directory = self.model_dir
        tmp = (directory / METADATA_FILE).with_suffix(".json.tmp")
        _write_json(tmp, metadata)
        os.replace(tmp, directory / METADATA_FILE)

    # ------------------------------------------------------------------ lectura

    def load(self, version: str | None = None) -> LoadedModel:
        directory = self.resolve_dir(version)
        missing = self.missing_files(version)
        if directory is None or missing:
            where = directory if directory is not None else self.model_dir
            raise FileNotFoundError(
                f"Artefactos ausentes en {where}: {missing}. Ejecuta antes el "
                "entrenamiento (python -m churn.training.train)."
            )
        try:
            metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelArtifactError(f"metadata.json ilegible en {directory}: {exc}") from exc
        if not isinstance(metadata, dict) or any(k not in metadata for k in REQUIRED_METADATA_KEYS):
            raise ModelArtifactError(
                f"metadata.json incompleto: se esperan las claves {list(REQUIRED_METADATA_KEYS)}"
            )
        try:
            pipeline = joblib.load(directory / MODEL_FILE)
        except Exception as exc:  # joblib/pickle lanzan tipos muy variados
            raise ModelArtifactError(f"model.joblib corrupto en {directory}: {exc}") from exc
        try:
            reference = pd.read_csv(directory / REFERENCE_FILE)
        except (OSError, ValueError) as exc:
            raise ModelArtifactError(f"reference.csv ilegible en {directory}: {exc}") from exc
        return LoadedModel(pipeline=pipeline, metadata=metadata, reference=reference)

    # ------------------------------------------------------------------ rollback

    def resolve_rollback_target(self, version: str | None = None) -> str:
        """Version a la que volver: la indicada, o la inmediatamente anterior a la actual.

        Lanza `LookupError` si no existe (o no hay anterior) y `ModelArtifactError` si esta
        incompleta. No toca el puntero: el llamador decide cuando (p. ej. tras cargarla).
        """
        versions = self.list_versions()
        current = self.current_version()
        if version is None:
            # "Anterior" por orden de publicacion; sin puntero (layout plano) es la ultima publicada
            position = versions.index(current) if current in versions else len(versions)
            if position == 0:
                raise LookupError("No hay una version anterior a la que volver")
            version = versions[position - 1]
        elif version not in versions:
            raise LookupError(f"Version desconocida: {version}. Disponibles: {versions}")
        missing = self.missing_files(version)
        if missing:
            raise ModelArtifactError(f"La version {version} esta incompleta: {missing}")
        return version

    def rollback(self, version: str | None = None) -> str:
        """Apunta `current` a otra version publicada y la devuelve."""
        target = self.resolve_rollback_target(version)
        self.set_current(target)
        return target

    def describe_versions(self) -> list[dict[str, Any]]:
        """Resumen de cada version publicada, de la mas antigua a la mas reciente."""
        current = self.current_version()
        summary = []
        for version in self.list_versions():
            entry: dict[str, Any] = {
                "version": version,
                "current": version == current,
                "complete": not self.missing_files(version),
                "trained_at": None,
                "roc_auc": None,
            }
            try:
                meta = json.loads(
                    (self.version_dir(version) / METADATA_FILE).read_text(encoding="utf-8")
                )
                entry["trained_at"] = meta.get("trained_at")
                entry["roc_auc"] = (meta.get("metrics") or {}).get("roc_auc")
            except (OSError, ValueError, AttributeError):
                pass
            summary.append(entry)
        return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
