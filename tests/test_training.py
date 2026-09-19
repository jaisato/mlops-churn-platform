import contextlib
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from churn import __version__
from churn.config import Settings
from churn.data.generator import CATEGORICAL_FEATURES, NUMERIC_FEATURES, generate_dataset
from churn.registry import LocalModelStore
from churn.training import train as train_module
from churn.training.train import (
    ModelQualityError,
    build_pipeline,
    main,
    new_model_version,
    train,
)


def test_entrenamiento_genera_artefactos(tmp_path):
    df = generate_dataset(n_rows=2000, seed=11)
    metadata = train(df, model_dir=str(tmp_path), seed=11)

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "reference.csv").exists()

    persisted = json.loads(Path(tmp_path / "metadata.json").read_text())
    assert persisted["model_version"] == metadata["model_version"]


def test_metricas_minimas_de_calidad(tmp_path):
    """Gate de calidad: el modelo debe superar un AUC minimo sobre datos sinteticos."""
    df = generate_dataset(n_rows=6000, seed=13)
    metadata = train(df, model_dir=str(tmp_path), seed=13)
    assert metadata["metrics"]["roc_auc"] > 0.80
    assert 0 < metadata["metrics"]["f1"] <= 1


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
    assert metadata["metrics"]["n_train"] + metadata["metrics"]["n_test"] == 1200
    datetime.fromisoformat(metadata["trained_at"])  # ISO-8601 valido
    reference = LocalModelStore(tmp_path).load().reference
    assert list(reference.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert len(reference) == metadata["metrics"]["n_train"]  # < 2000 -> toda la muestra


# ------------------------------------------------------------------ gate de calidad


def test_gate_de_calidad_no_sobrescribe_artefactos_anteriores(tmp_path, train_small):
    bueno = train_small(tmp_path, seed=23)
    with pytest.raises(ModelQualityError, match="por debajo del minimo"):
        train(generate_dataset(1200, seed=24), model_dir=str(tmp_path), seed=24, min_roc_auc=0.999)
    persisted = json.loads((tmp_path / "metadata.json").read_text())
    assert persisted["model_version"] == bueno["model_version"]
    assert not list(tmp_path.glob(".staging-*"))


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
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
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


def _install_fake_mlflow(monkeypatch, *, fail: bool = False, run_id: str = "run-abc123") -> dict:
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

    fake.set_tracking_uri = set_tracking_uri
    fake.set_experiment = lambda name: calls.__setitem__("experiment", name)
    fake.start_run = start_run
    fake.log_params = lambda params: calls.__setitem__("params", params)
    fake.log_metrics = lambda metrics: calls.__setitem__("metrics", metrics)
    fake_sklearn.log_model = lambda *args, **kwargs: calls.__setitem__("log_model", kwargs)
    fake.sklearn = fake_sklearn
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.sklearn", fake_sklearn)
    return calls


def _settings_con_mlflow(monkeypatch) -> None:
    monkeypatch.setattr(
        train_module,
        "get_settings",
        lambda: Settings(_env_file=None, mlflow_tracking_uri="http://mlflow.test:5000"),
    )


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
    assert (tmp_path / "model.joblib").exists()
    assert metadata["mlflow_run_id"] is None
    assert "Fallo al registrar en MLflow" in caplog.text
    assert "mlflow caido" in caplog.text


def test_mlflow_ok_registra_run_y_lo_persiste(tmp_path, monkeypatch, train_small):
    _settings_con_mlflow(monkeypatch)
    calls = _install_fake_mlflow(monkeypatch, run_id="run-xyz")
    metadata = train_small(tmp_path, seed=31)

    assert metadata["mlflow_run_id"] == "run-xyz"
    persisted = json.loads((tmp_path / "metadata.json").read_text())
    assert persisted["mlflow_run_id"] == "run-xyz"
    assert calls["uri"] == "http://mlflow.test:5000"
    assert calls["experiment"] == "churn"
    assert calls["run_name"] == f"train-{metadata['model_version']}"
    assert calls["params"]["seed"] == 31 and calls["params"]["min_roc_auc"] == 0.75
    assert calls["metrics"]["roc_auc"] == metadata["metrics"]["roc_auc"]
    assert calls["log_model"]["registered_model_name"] == "churn-classifier"
    assert len(calls["log_model"]["input_example"]) == 5


# ------------------------------------------------------------------ CLI


def test_cli_entrena_y_devuelve_0(tmp_path, capsys):
    code = main(["--rows", "800", "--seed", "5", "--model-dir", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "model.joblib").exists()
    assert "Entrenamiento OK" in capsys.readouterr().out


def test_cli_gate_fallido_devuelve_2_sin_artefactos(tmp_path):
    argv = ["--rows", "800", "--seed", "5", "--model-dir", str(tmp_path), "--min-auc", "0.999"]
    code = main(argv)
    assert code == 2
    assert not (tmp_path / "model.joblib").exists()
