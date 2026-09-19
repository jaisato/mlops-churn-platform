import json

import pandas as pd
import pytest

from churn.registry import ARTIFACT_FILES, LoadedModel, LocalModelStore, ModelArtifactError


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


def test_save_metadata_actualiza_solo_metadata(store):
    store.save(FakePipeline("v1"), METADATA, REFERENCE)
    store.save_metadata({**METADATA, "mlflow_run_id": "run-1"})
    loaded = store.load()
    assert loaded.metadata["mlflow_run_id"] == "run-1"
    assert loaded.pipeline.tag == "v1"
    assert json.loads(store.metadata_path.read_text())["mlflow_run_id"] == "run-1"
    assert not list(store.model_dir.glob("*.tmp"))
