import contextlib
import json
import platform
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sklearn

from churn import __version__
from churn.config import Settings
from churn.data.generator import CATEGORICAL_FEATURES, NUMERIC_FEATURES, generate_dataset
from churn.data.labels import write_labelled_dataset
from churn.monitoring.store import SqlitePredictionStore
from churn.registry import LocalModelStore
from churn.serving.schemas import ModelInfoResponse
from churn.training import train as train_module
from churn.training.train import (
    ModelQualityError,
    build_pipeline,
    main,
    new_model_version,
    runtime_versions,
    train,
)
from tests.conftest import SMALL_ROWS, score_and_label, weak_pipeline


def test_entrenamiento_genera_artefactos(tmp_path):
    df = generate_dataset(n_rows=2000, seed=11)
    metadata = train(df, model_dir=str(tmp_path), seed=11)

    store = LocalModelStore(tmp_path)
    assert store.exists()
    assert store.current_version() == metadata["model_version"]
    version_dir = Path(tmp_path) / "versions" / metadata["model_version"]
    assert (version_dir / "model.joblib").exists()
    assert (version_dir / "metadata.json").exists()
    assert (version_dir / "reference.csv").exists()

    persisted = json.loads(store.metadata_path.read_text())
    assert persisted["model_version"] == metadata["model_version"]


def test_cada_entrenamiento_publica_una_version_y_conserva_las_anteriores(tmp_path, train_small):
    primero = train_small(tmp_path, seed=12)
    segundo = train_small(tmp_path, seed=13)
    store = LocalModelStore(tmp_path)
    assert store.list_versions() == [primero["model_version"], segundo["model_version"]]
    assert store.current_version() == segundo["model_version"]
    assert store.load(primero["model_version"]).metadata["seed"] == 12


def test_retencion_de_versiones_sale_de_settings(tmp_path, monkeypatch, train_small):
    monkeypatch.setattr(
        train_module, "get_settings", lambda: Settings(_env_file=None, model_keep_versions=1)
    )
    train_small(tmp_path, seed=14)
    ultimo = train_small(tmp_path, seed=15)
    assert LocalModelStore(tmp_path).list_versions() == [ultimo["model_version"]]


def test_metricas_minimas_de_calidad(tmp_path):
    """Gate de calidad: el modelo debe superar un AUC minimo sobre datos sinteticos."""
    df = generate_dataset(n_rows=6000, seed=13)
    metadata = train(df, model_dir=str(tmp_path), seed=13)
    assert metadata["metrics"]["roc_auc"] > 0.80
    assert 0 < metadata["metrics"]["f1"] <= 1
    # Las mismas metricas (y umbral 0.5) que /monitoring/performance mide con etiquetas reales
    assert {"roc_auc", "accuracy", "precision", "recall", "f1", "brier"} <= set(metadata["metrics"])
    assert 0 < metadata["metrics"]["precision"] <= 1 and 0 < metadata["metrics"]["recall"] <= 1


def test_entrenamiento_rechaza_datos_invalidos(tmp_path):
    df = generate_dataset(n_rows=2000, seed=11).drop(columns=["churn"])
    with pytest.raises(ValueError):
        train(df, model_dir=str(tmp_path))
    assert not any(tmp_path.iterdir())


def test_store_load_roundtrip(tmp_path):
    df = generate_dataset(n_rows=2000, seed=17)
    train(df, model_dir=str(tmp_path), seed=17)
    loaded = LocalModelStore(tmp_path).load()
    cols = loaded.metadata["numeric_features"] + loaded.metadata["categorical_features"]
    proba = loaded.pipeline.predict_proba(df.head(3)[cols])
    assert proba.shape == (3, 2)


def test_metadata_contiene_trazabilidad_completa(tmp_path, train_small):
    metadata = train_small(tmp_path, seed=19)
    assert metadata["code_version"] == __version__
    assert metadata["seed"] == 19
    assert metadata["numeric_features"] == NUMERIC_FEATURES
    assert metadata["categorical_features"] == CATEGORICAL_FEATURES
    assert metadata["quality_gate"] == {"min_roc_auc": 0.75, "passed": True}
    assert metadata["mlflow_run_id"] is None
    assert metadata["mlflow_model_version"] is None
    assert metadata["runtime"]["scikit_learn"] == sklearn.__version__
    assert metadata["runtime"]["python"] == platform.python_version()
    assert set(metadata["runtime"]) == {"python", "scikit_learn", "numpy", "pandas"}
    assert metadata["metrics"]["n_train"] + metadata["metrics"]["n_test"] == 1200
    datetime.fromisoformat(metadata["trained_at"])  # ISO-8601 valido
    reference = LocalModelStore(tmp_path).load().reference
    assert list(reference.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert len(reference) == metadata["metrics"]["n_train"]  # < 2000 -> toda la muestra


def test_runtime_versions_son_cadenas_no_vacias():
    versions = runtime_versions()
    assert all(isinstance(v, str) and v for v in versions.values())


# ------------------------------------------------------------------ gate de calidad


def test_gate_de_calidad_no_sobrescribe_artefactos_anteriores(tmp_path, train_small):
    bueno = train_small(tmp_path, seed=23)
    with pytest.raises(ModelQualityError, match="por debajo del minimo"):
        train(generate_dataset(1200, seed=24), model_dir=str(tmp_path), seed=24, min_roc_auc=0.999)
    store = LocalModelStore(tmp_path)
    assert json.loads(store.metadata_path.read_text())["model_version"] == bueno["model_version"]
    assert store.list_versions() == [bueno["model_version"]]
    assert not list(store.versions_dir.glob(".staging-*"))


def test_gate_de_calidad_en_directorio_vacio_no_crea_nada(tmp_path):
    with pytest.raises(ModelQualityError):
        train(generate_dataset(1200, seed=25), model_dir=str(tmp_path / "nuevo"), min_roc_auc=1.0)
    assert not (tmp_path / "nuevo").exists()


def test_gate_por_defecto_sale_de_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train_module, "get_settings", lambda: Settings(_env_file=None, min_roc_auc=0.999)
    )
    with pytest.raises(ModelQualityError):
        train(generate_dataset(1200, seed=26), model_dir=str(tmp_path))


# ------------------------------------------------------------------ versiones y pipeline


def test_versiones_unicas_incluso_en_el_mismo_segundo():
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    a, b = new_model_version(now), new_model_version(now)
    assert a != b
    assert a.startswith("20260102-030405-") and len(a) == len("20260102-030405-") + 6


def test_build_pipeline_es_reproducible():
    df = generate_dataset(1000, seed=27)
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    p1 = build_pipeline(seed=1).fit(df[cols], df["churn"]).predict_proba(df[cols])
    p2 = build_pipeline(seed=1).fit(df[cols], df["churn"]).predict_proba(df[cols])
    assert (p1 == p2).all()


# ------------------------------------------------------------------ MLflow (opcional)


class _FakeRun:
    def __init__(self, run_id: str):
        self.info = types.SimpleNamespace(run_id=run_id)


def _install_fake_mlflow(
    monkeypatch,
    *,
    fail: bool = False,
    run_id: str = "run-abc123",
    registered_version: str | None = "3",
    search_result: str | None = None,
    alias_fail: bool = False,
) -> dict:
    """Sustituye mlflow en sys.modules por un doble que registra las llamadas."""
    calls: dict = {}
    fake = types.ModuleType("mlflow")
    fake_sklearn = types.ModuleType("mlflow.sklearn")

    def set_tracking_uri(uri):
        if fail:
            raise ConnectionError("mlflow caido")
        calls["uri"] = uri

    @contextlib.contextmanager
    def start_run(run_name=None):
        calls["run_name"] = run_name
        yield _FakeRun(run_id)

    def log_model(*args, **kwargs):
        calls["log_model"] = kwargs
        return types.SimpleNamespace(registered_model_version=registered_version)

    class FakeClient:
        def search_model_versions(self, filter_string):
            calls["search"] = filter_string
            if search_result is None:
                return []
            return [types.SimpleNamespace(version=search_result)]

        def set_registered_model_alias(self, name, alias, version):
            if alias_fail:
                raise RuntimeError("registry caido")
            calls["alias"] = (name, alias, version)

    fake.set_tracking_uri = set_tracking_uri
    fake.set_experiment = lambda name: calls.__setitem__("experiment", name)
    fake.start_run = start_run
    fake.log_params = lambda params: calls.__setitem__("params", params)
    fake.log_metrics = lambda metrics: calls.setdefault("metrics", {}).update(metrics)
    fake.MlflowClient = FakeClient
    fake_sklearn.log_model = log_model
    fake.sklearn = fake_sklearn
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.sklearn", fake_sklearn)
    return calls


def _settings_con_mlflow(monkeypatch, **overrides) -> None:
    settings = Settings(_env_file=None, mlflow_tracking_uri="http://mlflow.test:5000", **overrides)
    monkeypatch.setattr(train_module, "get_settings", lambda: settings)


def test_mlflow_desactivado_sin_tracking_uri(tmp_path, monkeypatch, train_small):
    monkeypatch.setitem(sys.modules, "mlflow", None)  # importarlo lanzaria ImportError
    metadata = train_small(tmp_path, seed=28)
    assert metadata["mlflow_run_id"] is None


def test_mlflow_no_instalado_solo_avisa(tmp_path, monkeypatch, train_small, caplog):
    _settings_con_mlflow(monkeypatch)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    metadata = train_small(tmp_path, seed=29)
    assert metadata["mlflow_run_id"] is None
    assert "MLflow no instalado" in caplog.text


def test_mlflow_caido_no_tumba_el_entrenamiento(tmp_path, monkeypatch, train_small, caplog):
    _settings_con_mlflow(monkeypatch)
    _install_fake_mlflow(monkeypatch, fail=True)
    metadata = train_small(tmp_path, seed=30)
    assert LocalModelStore(tmp_path).exists()
    assert metadata["mlflow_run_id"] is None
    assert "Fallo al registrar en MLflow" in caplog.text
    assert "mlflow caido" in caplog.text


def test_mlflow_ok_registra_run_alias_y_lo_persiste(tmp_path, monkeypatch, train_small):
    _settings_con_mlflow(monkeypatch)
    calls = _install_fake_mlflow(monkeypatch, run_id="run-xyz", registered_version="3")
    metadata = train_small(tmp_path, seed=31)

    assert metadata["mlflow_run_id"] == "run-xyz"
    assert metadata["mlflow_model_version"] == "3"
    persisted = json.loads(LocalModelStore(tmp_path).metadata_path.read_text())
    assert persisted["mlflow_run_id"] == "run-xyz"
    assert persisted["mlflow_model_version"] == "3"
    assert calls["uri"] == "http://mlflow.test:5000"
    assert calls["experiment"] == "churn"
    assert calls["run_name"] == f"train-{metadata['model_version']}"
    assert calls["params"]["seed"] == 31 and calls["params"]["min_roc_auc"] == 0.75
    assert calls["params"]["scikit_learn"] == sklearn.__version__
    assert calls["params"]["promotion_decision"] == "no_champion"
    assert calls["params"]["promotion_margin"] == 1.0  # train_small desactiva la comparacion
    assert calls["params"]["champion_version"] is None
    assert calls["params"]["data_kind"] == "dataframe"
    assert calls["params"]["data_fingerprint"] == metadata["data_source"]["fingerprint"]
    assert "champion_roc_auc" not in calls["metrics"]
    assert calls["metrics"]["roc_auc"] == metadata["metrics"]["roc_auc"]
    assert calls["log_model"]["registered_model_name"] == "churn-classifier"
    assert len(calls["log_model"]["input_example"]) == 5
    assert calls["alias"] == ("churn-classifier", "champion", "3")
    assert "search" not in calls  # log_model ya devolvio la version registrada


def test_mlflow_busca_la_version_si_log_model_no_la_devuelve(tmp_path, monkeypatch, train_small):
    _settings_con_mlflow(monkeypatch)
    calls = _install_fake_mlflow(
        monkeypatch, run_id="run-old", registered_version=None, search_result="7"
    )
    metadata = train_small(tmp_path, seed=32)
    assert "run_id='run-old'" in calls["search"]
    assert calls["alias"] == ("churn-classifier", "champion", "7")
    assert metadata["mlflow_model_version"] == "7"


def test_mlflow_sin_version_registrada_no_asigna_alias(tmp_path, monkeypatch, train_small, caplog):
    _settings_con_mlflow(monkeypatch)
    calls = _install_fake_mlflow(monkeypatch, registered_version=None, search_result=None)
    metadata = train_small(tmp_path, seed=33)
    assert "alias" not in calls
    assert metadata["mlflow_run_id"] == "run-abc123"
    assert metadata["mlflow_model_version"] is None
    assert "No se encontro la version registrada" in caplog.text


def test_mlflow_alias_fallido_conserva_el_run_id(tmp_path, monkeypatch, train_small, caplog):
    _settings_con_mlflow(monkeypatch)
    _install_fake_mlflow(monkeypatch, alias_fail=True)
    metadata = train_small(tmp_path, seed=34)
    assert metadata["mlflow_run_id"] == "run-abc123"
    assert metadata["mlflow_model_version"] is None
    assert "No se pudo asignar el alias" in caplog.text


def test_mlflow_alias_desactivado_por_configuracion(tmp_path, monkeypatch, train_small):
    _settings_con_mlflow(monkeypatch, mlflow_champion_alias="")
    calls = _install_fake_mlflow(monkeypatch)
    metadata = train_small(tmp_path, seed=35)
    assert "alias" not in calls
    assert metadata["mlflow_run_id"] == "run-abc123"
    assert metadata["mlflow_model_version"] is None


def test_mlflow_registra_al_campeon_y_al_retador_promovido(tmp_path, monkeypatch, train_small):
    _settings_con_mlflow(monkeypatch)
    calls = _install_fake_mlflow(monkeypatch, registered_version="4")
    campeon = train_small(tmp_path, seed=36)
    retador = train_small(tmp_path, seed=37)  # margen 1.0: se promueve
    assert calls["params"]["promotion_decision"] == "promoted"
    assert calls["params"]["champion_version"] == campeon["model_version"]
    assert calls["metrics"]["champion_roc_auc"] == retador["promotion"]["champion_roc_auc"]
    assert calls["alias"] == ("churn-classifier", "champion", "4")


def test_mlflow_registra_el_run_pero_no_mueve_el_alias_si_no_promueve(
    tmp_path, monkeypatch, train_small, caplog
):
    _settings_con_mlflow(monkeypatch)
    calls = _install_fake_mlflow(monkeypatch, registered_version="5")
    train_small(tmp_path, seed=38)
    calls.clear()
    monkeypatch.setattr(train_module, "build_pipeline", weak_pipeline)
    df = generate_dataset(SMALL_ROWS, seed=39)
    metadata = train(df, model_dir=str(tmp_path), seed=39, min_roc_auc=0.5)
    assert metadata["promotion"]["decision"] == "rejected"
    assert calls["params"]["promotion_decision"] == "rejected"
    assert "alias" not in calls
    assert metadata["mlflow_run_id"] == "run-abc123"
    assert metadata["mlflow_model_version"] is None
    assert "el alias de campeon no cambia" in caplog.text
    persisted = json.loads(
        (
            LocalModelStore(tmp_path).version_dir(metadata["model_version"]) / "metadata.json"
        ).read_text()
    )
    assert persisted["mlflow_run_id"] == "run-abc123"


# ------------------------------------------------------------------ CLI


def test_cli_entrena_y_devuelve_0(tmp_path, capsys):
    code = main(["--rows", "800", "--seed", "5", "--model-dir", str(tmp_path)])
    assert code == 0
    assert LocalModelStore(tmp_path).exists()
    assert "Entrenamiento OK" in capsys.readouterr().out


def test_cli_gate_fallido_devuelve_2_sin_artefactos(tmp_path):
    argv = ["--rows", "800", "--seed", "5", "--model-dir", str(tmp_path), "--min-auc", "0.999"]
    code = main(argv)
    assert code == 2
    assert not LocalModelStore(tmp_path).exists()


def test_cli_registra_la_fuente_sintetica_y_el_drift_shift(tmp_path):
    argv = ["--rows", "800", "--seed", "5", "--drift-shift", "1.0", "--model-dir", str(tmp_path)]
    assert main(argv) == 0
    metadata = LocalModelStore(tmp_path).load().metadata
    assert metadata["data_source"]["kind"] == "synthetic"
    assert metadata["data_source"]["seed"] == 5
    assert metadata["data_source"]["drift_shift"] == 1.0
    assert metadata["data_source"]["rows"] == 800


def test_cli_entrena_desde_un_fichero(tmp_path, capsys):
    data = tmp_path / "clientes.parquet"
    generate_dataset(900, seed=6).to_parquet(data, index=False)
    argv = ["--data", str(data), "--seed", "6", "--model-dir", str(tmp_path / "models")]
    assert main(argv) == 0
    assert "(no_champion)" in capsys.readouterr().out
    metadata = LocalModelStore(tmp_path / "models").load().metadata
    assert metadata["data_source"]["kind"] == "file"
    assert metadata["data_source"]["path"] == str(data)
    assert metadata["data_source"]["rows"] == 900
    assert metadata["metrics"]["n_train"] + metadata["metrics"]["n_test"] == 900


def test_cli_entrena_desde_el_dataset_de_etiquetas_y_registra_su_ventana(tmp_path, capsys):
    store = SqlitePredictionStore(tmp_path / "drift.sqlite", max_rows=5000)
    score_and_label(
        store, generate_dataset(900, seed=8), version="v-anterior", observed_at="2026-10-01T00:00Z"
    )
    summary = write_labelled_dataset(store, tmp_path / "datasets" / "labels.parquet", min_rows=500)

    argv = ["--data", summary["path"], "--seed", "8", "--model-dir", str(tmp_path / "models")]
    assert main(argv) == 0

    metadata = LocalModelStore(tmp_path / "models").load().metadata
    source = metadata["data_source"]
    assert source["kind"] == "file"  # la procedencia de etiquetas va en el bloque `labels`
    assert source["path"] == summary["path"]
    assert source["rows"] == 900
    assert source["fingerprint"] == summary["fingerprint"]  # la huella del sidecar es la real
    assert source["labels"] == {
        "built_at": summary["built_at"],
        "predicted_from": summary["predicted_from"],
        "predicted_to": summary["predicted_to"],
        "observed_from": "2026-10-01T00:00:00+00:00",
        "observed_to": "2026-10-01T00:00:00+00:00",
        "model_versions": ["v-anterior"],
        "labels_total": 900,
        "subjects": 0,
    }
    assert ModelInfoResponse(**metadata).data_source.labels.model_versions == ["v-anterior"]
    assert "labels" in capsys.readouterr().out  # la procedencia queda en el log del trainer


def test_cli_fichero_inexistente_devuelve_2(tmp_path, capsys):
    assert main(["--data", str(tmp_path / "nada.csv"), "--model-dir", str(tmp_path)]) == 2
    assert "no disponibles" in capsys.readouterr().out  # el log JSON va a stdout
    assert not (tmp_path / "versions").exists()


def test_cli_rechaza_drift_shift_con_fichero(tmp_path):
    data = tmp_path / "clientes.csv"
    generate_dataset(600, seed=6).to_csv(data, index=False)
    argv = ["--data", str(data), "--drift-shift", "0.5", "--model-dir", str(tmp_path)]
    assert main(argv) == 2


def test_cli_fichero_con_datos_invalidos_devuelve_2(tmp_path, capsys):
    data = tmp_path / "sin_etiqueta.csv"
    generate_dataset(600, seed=6).drop(columns=["churn"]).to_csv(data, index=False)
    assert main(["--data", str(data), "--model-dir", str(tmp_path)]) == 2
    assert "Faltan columnas" in capsys.readouterr().out
