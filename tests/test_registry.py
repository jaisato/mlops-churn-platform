import json
import platform

import pandas as pd
import pytest
import sklearn

from churn.data.generator import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from churn.registry import (
    ARTIFACT_FILES,
    LoadedModel,
    LocalModelStore,
    ModelArtifactError,
    ModelCompatibilityError,
    verify_compatibility,
)
from churn.training.train import runtime_versions


class FakePipeline:
    """Sustituto serializable del pipeline sklearn (joblib lo pickle-a por referencia)."""

    def __init__(self, tag: str = "v1"):
        self.tag = tag


METADATA = {
    "model_version": "20260101-000000-abc123",
    "numeric_features": ["a"],
    "categorical_features": ["c"],
}
REFERENCE = pd.DataFrame({"a": [1, 2, 3], "c": ["x", "y", "x"]})


@pytest.fixture()
def store(tmp_path) -> LocalModelStore:
    return LocalModelStore(tmp_path / "models")


def test_load_sin_artefactos_lanza_filenotfound(store):
    assert not store.exists()
    with pytest.raises(FileNotFoundError, match="Ejecuta antes el entrenamiento"):
        store.load()


def test_save_crea_el_directorio_y_los_tres_ficheros(store):
    store.save(FakePipeline(), METADATA, REFERENCE)
    assert store.exists()
    assert store.missing_files() == []
    assert sorted(p.name for p in store.model_dir.iterdir()) == sorted(ARTIFACT_FILES)


def test_save_no_deja_residuos_de_staging(store):
    store.save(FakePipeline(), METADATA, REFERENCE)
    store.save(FakePipeline("v2"), {**METADATA, "model_version": "v2"}, REFERENCE)
    assert not list(store.model_dir.glob(".staging-*"))
    assert not list(store.model_dir.glob("*.tmp"))


def test_roundtrip(store):
    store.save(FakePipeline("v1"), METADATA, REFERENCE)
    loaded = store.load()
    assert isinstance(loaded, LoadedModel)
    assert loaded.pipeline.tag == "v1"
    assert loaded.metadata == METADATA
    assert loaded.version == METADATA["model_version"]
    assert loaded.feature_columns == ["a", "c"]
    pd.testing.assert_frame_equal(loaded.reference, REFERENCE)


def test_save_sobrescribe_la_version_anterior(store):
    store.save(FakePipeline("v1"), METADATA, REFERENCE)
    store.save(FakePipeline("v2"), {**METADATA, "model_version": "v2"}, REFERENCE.head(1))
    loaded = store.load()
    assert loaded.pipeline.tag == "v2"
    assert loaded.version == "v2"
    assert len(loaded.reference) == 1


def test_exists_requiere_el_juego_completo(store):
    store.save(FakePipeline(), METADATA, REFERENCE)
    store.reference_path.unlink()
    assert not store.exists()
    assert store.missing_files() == ["reference.csv"]
    with pytest.raises(FileNotFoundError, match="reference.csv"):
        store.load()


def test_metadata_corrupto(store):
    store.save(FakePipeline(), METADATA, REFERENCE)
    store.metadata_path.write_text("{no es json", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="metadata.json ilegible"):
        store.load()


def test_metadata_incompleto(store):
    store.save(FakePipeline(), {"model_version": "x"}, REFERENCE)
    with pytest.raises(ModelArtifactError, match="incompleto"):
        store.load()


def test_metadata_no_es_un_objeto(store):
    store.save(FakePipeline(), METADATA, REFERENCE)
    store.metadata_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="incompleto"):
        store.load()


def test_modelo_corrupto(store):
    store.save(FakePipeline(), METADATA, REFERENCE)
    store.model_path.write_bytes(b"\x00\x01basura")
    with pytest.raises(ModelArtifactError, match="model.joblib corrupto"):
        store.load()


def test_reference_ilegible(store):
    store.save(FakePipeline(), METADATA, REFERENCE)
    store.reference_path.write_text("", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="reference.csv ilegible"):
        store.load()


# ------------------------------------------------------------------ compatibilidad


def _compatible_metadata(**overrides) -> dict:
    metadata = {
        "model_version": "v1",
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "runtime": runtime_versions(),
    }
    metadata.update(overrides)
    return metadata


def _bump_minor(version: str) -> str:
    major, minor, *rest = version.split(".")
    return ".".join([major, str(int(minor) + 1), *rest])


def test_compatibilidad_modelo_actual_sin_avisos():
    assert verify_compatibility(_compatible_metadata()) == []


def test_compatibilidad_rechaza_features_distintas():
    metadata = _compatible_metadata(numeric_features=list(NUMERIC_FEATURES[:-1]))
    with pytest.raises(ModelCompatibilityError, match="features del modelo"):
        verify_compatibility(metadata)
    extra = _compatible_metadata(categorical_features=[*CATEGORICAL_FEATURES, "region"])
    with pytest.raises(ModelCompatibilityError, match="features del modelo"):
        verify_compatibility(extra, strict_runtime=False)


def test_compatibilidad_el_orden_de_las_features_no_importa():
    metadata = _compatible_metadata(
        numeric_features=list(reversed(NUMERIC_FEATURES)),
        categorical_features=list(reversed(CATEGORICAL_FEATURES)),
    )
    assert verify_compatibility(metadata) == []


def test_compatibilidad_metadata_antiguo_sin_runtime_solo_avisa():
    metadata = _compatible_metadata()
    del metadata["runtime"]
    warnings = verify_compatibility(metadata)
    assert len(warnings) == 1 and "sin versiones de runtime" in warnings[0]


def test_compatibilidad_sklearn_otra_version_menor():
    runtime = {**runtime_versions(), "scikit_learn": _bump_minor(sklearn.__version__)}
    metadata = _compatible_metadata(runtime=runtime)
    with pytest.raises(ModelCompatibilityError, match="CHURN_STRICT_ARTIFACT_COMPAT"):
        verify_compatibility(metadata, strict_runtime=True)
    warnings = verify_compatibility(metadata, strict_runtime=False)
    assert any("no son portables" in w for w in warnings)


def test_compatibilidad_sklearn_otro_parche_solo_avisa():
    runtime = {**runtime_versions(), "scikit_learn": sklearn.__version__ + ".post1"}
    warnings = verify_compatibility(_compatible_metadata(runtime=runtime))
    assert len(warnings) == 1 and "distinto parche" in warnings[0]


def test_compatibilidad_python_distinto_solo_avisa():
    runtime = {**runtime_versions(), "python": _bump_minor(platform.python_version())}
    warnings = verify_compatibility(_compatible_metadata(runtime=runtime))
    assert len(warnings) == 1 and "Python del modelo" in warnings[0]


def test_save_metadata_actualiza_solo_metadata(store):
    store.save(FakePipeline("v1"), METADATA, REFERENCE)
    store.save_metadata({**METADATA, "mlflow_run_id": "run-1"})
    loaded = store.load()
    assert loaded.metadata["mlflow_run_id"] == "run-1"
    assert loaded.pipeline.tag == "v1"
    assert json.loads(store.metadata_path.read_text())["mlflow_run_id"] == "run-1"
    assert not list(store.model_dir.glob("*.tmp"))
