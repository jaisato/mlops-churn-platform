import json
import platform
import shutil

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


def _metadata(version: str = "20260101-000000-abc123", **extra) -> dict:
    return {
        "model_version": version,
        "numeric_features": ["a"],
        "categorical_features": ["c"],
        **extra,
    }


METADATA = _metadata()
REFERENCE = pd.DataFrame({"a": [1, 2, 3], "c": ["x", "y", "x"]})


@pytest.fixture()
def store(tmp_path) -> LocalModelStore:
    return LocalModelStore(tmp_path / "models", keep_versions=3)


def _publish(store: LocalModelStore, version: str) -> str:
    metadata = _metadata(version, trained_at=f"t-{version}")
    return store.save(FakePipeline(version), metadata, REFERENCE)


def test_save_fallido_no_deja_staging_ni_version(store):
    _publish(store, "v1")
    with pytest.raises(Exception, match="pickle"):
        store.save(lambda x: x, _metadata("v2"), REFERENCE)  # una lambda no se puede serializar
    assert store.list_versions() == ["v1"]
    assert store.current_version() == "v1"
    assert not list(store.versions_dir.glob(".staging-*"))


def test_prune_sin_puntero_conserva_las_mas_recientes(store):
    for v in ["v1", "v2", "v3", "v4"]:
        _publish(store, v)
    store.current_file.unlink()
    store.keep_versions = 2
    assert store.prune() == ["v2"]
    assert store.list_versions() == ["v3", "v4"]
    assert store.current_version() is None


# ------------------------------------------------------------------ layout y publicacion


def test_load_sin_artefactos_lanza_filenotfound(store):
    assert not store.exists()
    assert store.current_version() is None
    with pytest.raises(FileNotFoundError, match="Ejecuta antes el entrenamiento"):
        store.load()


def test_save_publica_una_version_y_la_deja_en_servicio(store):
    version = _publish(store, "v1")
    assert version == "v1"
    assert store.exists()
    assert store.current_version() == "v1"
    assert store.current_file.read_text().strip() == "v1"
    assert sorted(p.name for p in store.version_dir("v1").iterdir()) == sorted(ARTIFACT_FILES)
    assert store.model_path == store.version_dir("v1") / "model.joblib"


def test_save_no_deja_residuos_de_staging(store):
    _publish(store, "v1")
    _publish(store, "v2")
    assert not list(store.versions_dir.glob(".staging-*"))
    assert not list(store.model_dir.glob("*.tmp"))


def test_roundtrip(store):
    _publish(store, "v1")
    loaded = store.load()
    assert isinstance(loaded, LoadedModel)
    assert loaded.pipeline.tag == "v1"
    assert loaded.metadata["model_version"] == "v1"
    assert loaded.version == "v1"
    assert loaded.feature_columns == ["a", "c"]
    pd.testing.assert_frame_equal(loaded.reference, REFERENCE)


def test_cada_save_crea_una_version_nueva_y_apunta_a_la_ultima(store):
    _publish(store, "v1")
    _publish(store, "v2")
    assert store.list_versions() == ["v1", "v2"]
    assert store.current_version() == "v2"
    assert store.load().pipeline.tag == "v2"
    assert store.load("v1").pipeline.tag == "v1"


def test_republicar_la_misma_version_la_sustituye(store):
    _publish(store, "v1")
    store.save(FakePipeline("v1-bis"), _metadata("v1"), REFERENCE.head(1))
    assert store.list_versions() == ["v1"]
    assert store.load().pipeline.tag == "v1-bis"
    assert len(store.load().reference) == 1


def test_retencion_poda_las_mas_antiguas(store):
    for v in ["v1", "v2", "v3", "v4"]:
        _publish(store, v)
    assert store.list_versions() == ["v2", "v3", "v4"]  # keep_versions=3
    assert not store.version_dir("v1").exists()


def test_retencion_nunca_borra_la_version_en_servicio(store):
    _publish(store, "v1")
    _publish(store, "v2")
    store.set_current("v1")
    store.keep_versions = 1
    assert store.prune() == []  # v1 es la actual: se conserva aunque exceda la ventana
    assert store.list_versions() == ["v1", "v2"]
    store.set_current("v2")
    assert store.prune() == ["v1"]
    assert store.list_versions() == ["v2"]


def test_las_versiones_se_ordenan_por_fecha_de_entrenamiento_no_por_nombre(store):
    store.save(FakePipeline("b"), _metadata("zzz", trained_at="2026-01-01T00:00:00"), REFERENCE)
    store.save(FakePipeline("a"), _metadata("aaa", trained_at="2026-01-02T00:00:00"), REFERENCE)
    assert store.list_versions() == ["zzz", "aaa"]
    assert store.rollback() == "zzz"


def test_exists_requiere_el_juego_completo(store):
    _publish(store, "v1")
    store.reference_path.unlink()
    assert not store.exists()
    assert store.missing_files() == ["reference.csv"]
    with pytest.raises(FileNotFoundError, match="reference.csv"):
        store.load()


def test_exists_de_una_version_concreta(store):
    _publish(store, "v1")
    assert store.exists("v1") and not store.exists("v9")
    assert store.missing_files("v9") == list(ARTIFACT_FILES)


# ------------------------------------------------------------------ layout plano (1.0)


def test_layout_plano_se_carga_y_migra_al_siguiente_save(store, tmp_path):
    legacy = LocalModelStore(tmp_path / "legacy")
    _publish(store, "old")
    legacy.model_dir.mkdir()
    for name in ARTIFACT_FILES:
        shutil.copy(store.version_dir("old") / name, legacy.model_dir / name)

    assert legacy.has_legacy_layout()
    assert legacy.current_version() is None
    assert legacy.exists()
    assert legacy.metadata_path == legacy.model_dir / "metadata.json"
    assert legacy.load().pipeline.tag == "old"
    assert legacy.describe_versions() == []

    legacy.save_metadata({**_metadata("old"), "mlflow_run_id": "r1"})  # va al fichero plano
    assert json.loads((legacy.model_dir / "metadata.json").read_text())["mlflow_run_id"] == "r1"

    _publish(legacy, "new")
    assert legacy.current_version() == "new"
    assert legacy.load().pipeline.tag == "new"
    assert legacy.load("new").version == "new"


# ------------------------------------------------------------------ artefactos corruptos


def test_metadata_corrupto(store):
    _publish(store, "v1")
    store.metadata_path.write_text("{no es json", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="metadata.json ilegible"):
        store.load()


def test_metadata_incompleto(store):
    store.save(FakePipeline(), {"model_version": "x"}, REFERENCE)
    with pytest.raises(ModelArtifactError, match="incompleto"):
        store.load()


def test_metadata_no_es_un_objeto(store):
    _publish(store, "v1")
    store.metadata_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="incompleto"):
        store.load()


def test_modelo_corrupto(store):
    _publish(store, "v1")
    store.model_path.write_bytes(b"\x00\x01basura")
    with pytest.raises(ModelArtifactError, match="model.joblib corrupto"):
        store.load()


def test_reference_ilegible(store):
    _publish(store, "v1")
    store.reference_path.write_text("", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="reference.csv ilegible"):
        store.load()


def test_save_metadata_actualiza_solo_metadata(store):
    _publish(store, "v1")
    store.save_metadata({**_metadata("v1"), "mlflow_run_id": "run-1"})
    loaded = store.load()
    assert loaded.metadata["mlflow_run_id"] == "run-1"
    assert loaded.pipeline.tag == "v1"
    assert json.loads(store.metadata_path.read_text())["mlflow_run_id"] == "run-1"
    assert not list(store.version_dir("v1").glob("*.tmp"))


# ------------------------------------------------------------------ rollback


def test_rollback_por_defecto_vuelve_a_la_anterior(store):
    for v in ["v1", "v2", "v3"]:
        _publish(store, v)
    assert store.rollback() == "v2"
    assert store.current_version() == "v2"
    assert store.load().pipeline.tag == "v2"
    assert store.rollback() == "v1"  # anterior a la actual, no a la ultima publicada


def test_rollback_a_una_version_concreta_en_cualquier_direccion(store):
    for v in ["v1", "v2", "v3"]:
        _publish(store, v)
    assert store.rollback("v1") == "v1"
    assert store.rollback("v3") == "v3"


def test_rollback_sin_anterior_o_desconocida(store):
    with pytest.raises(LookupError, match="No hay una version anterior"):
        store.rollback()
    _publish(store, "v1")
    with pytest.raises(LookupError, match="No hay una version anterior"):
        store.rollback()
    with pytest.raises(LookupError, match="Version desconocida: v9"):
        store.rollback("v9")
    assert store.current_version() == "v1"


def test_rollback_a_version_incompleta_no_mueve_el_puntero(store):
    _publish(store, "v1")
    _publish(store, "v2")
    (store.version_dir("v1") / "model.joblib").unlink()
    with pytest.raises(ModelArtifactError, match="incompleta"):
        store.rollback()
    assert store.current_version() == "v2"


def test_describe_versions(store):
    _publish(store, "v1")
    _publish(store, "v2")
    (store.version_dir("v1") / "metadata.json").write_text("{roto", encoding="utf-8")
    described = store.describe_versions()
    assert [d["version"] for d in described] == ["v1", "v2"]
    assert described[1] == {
        "version": "v2",
        "current": True,
        "complete": True,
        "trained_at": "t-v2",
        "roc_auc": None,
    }
    assert described[0]["current"] is False and described[0]["trained_at"] is None


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
