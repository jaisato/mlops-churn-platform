"""Dataset de reentreno construido desde las etiquetas del almacen (`churn.data.labels`)."""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from churn.config import Settings
from churn.data import labels as labels_module
from churn.data.generator import FEATURE_COLUMNS, TARGET, generate_dataset
from churn.data.labels import (
    EXIT_ERROR,
    EXIT_INSUFFICIENT,
    EXIT_OK,
    InsufficientLabelsError,
    LabelledWindow,
    labelled_dataset,
    main,
    write_labelled_dataset,
)
from churn.data.sources import (
    dataset_fingerprint,
    load_training_data,
    read_sidecar,
    resolve_training_data,
    sidecar_path,
)
from churn.monitoring.store import MemoryPredictionStore, SqlitePredictionStore
from tests.conftest import score_and_label

OBSERVED = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return MemoryPredictionStore(max_rows=5000)
    return SqlitePredictionStore(tmp_path / "drift.sqlite", max_rows=5000)


# --------------------------------------------------------------------------- dataset


def test_sin_etiquetas_devuelve_un_dataset_vacio(store):
    store.append([{"tenure_months": 1, "churn_probability": 0.5, "model_version": "v1"}])
    df, window = labelled_dataset(store)
    assert df.empty and list(df.columns) == []
    assert window == LabelledWindow(0, None, None, None, None, [], None)


def test_dataset_etiquetado_tiene_las_columnas_del_contrato_en_orden_de_prediccion(store):
    datos = generate_dataset(30, seed=1)
    score_and_label(store, datos.head(20), version="v1", observed_at=OBSERVED)
    score_and_label(store, datos.tail(10), version="v2", observed_at="2026-10-05T00:00:00Z")

    df, window = labelled_dataset(store)

    assert list(df.columns) == FEATURE_COLUMNS + [TARGET]
    pd.testing.assert_frame_equal(df, datos.reset_index(drop=True), check_dtype=False)
    assert window.rows == 30
    assert window.model_versions == ["v1", "v2"]
    assert window.churn_rate == round(float(datos[TARGET].mean()), 4)
    assert window.observed_from == "2026-10-01T12:00:00+00:00"
    assert window.observed_to == "2026-10-05T00:00:00+00:00"
    assert window.predicted_from <= window.predicted_to
    assert datetime.fromisoformat(window.predicted_to).tzinfo is not None


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_write_escribe_dataset_y_sidecar_con_la_huella_del_fichero(store, tmp_path, suffix):
    score_and_label(store, generate_dataset(600, seed=2), observed_at=OBSERVED)
    out = tmp_path / "datasets" / f"labels{suffix}"

    summary = write_labelled_dataset(store, out, min_rows=500)

    df = load_training_data(out)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET] and len(df) == 600
    assert summary["kind"] == "labels"
    assert summary["path"] == str(out)
    assert summary["rows"] == 600
    assert summary["fingerprint"] == dataset_fingerprint(df)  # la que calculara el trainer
    assert summary["model_versions"] == ["v1"]
    assert datetime.fromisoformat(summary["built_at"]).tzinfo is not None
    sidecar = read_sidecar(out)
    assert sidecar == summary
    assert sidecar_path(out) == out.parent / f"labels{suffix}.meta.json"

    # El trainer reconoce la procedencia a traves del sidecar
    _, source = resolve_training_data(data_path=out)
    assert source["kind"] == "labels"
    assert source["labels"]["model_versions"] == ["v1"]
    assert source["labels"]["observed_from"] == "2026-10-01T12:00:00+00:00"


def test_write_rechaza_etiquetas_insuficientes_sin_escribir_nada(store, tmp_path):
    score_and_label(store, generate_dataset(120, seed=3))
    out = tmp_path / "labels.parquet"
    with pytest.raises(InsufficientLabelsError, match="120 de un minimo de 500") as exc:
        write_labelled_dataset(store, out, min_rows=500)
    assert (exc.value.rows, exc.value.min_rows) == (120, 500)
    assert not out.exists() and not sidecar_path(out).exists()


def test_write_rechaza_un_dataset_que_el_trainer_no_aceptaria(store, tmp_path):
    datos = generate_dataset(600, seed=4)
    datos[TARGET] = 1  # todas bajas: tasa de churn fuera de los limites del validador
    score_and_label(store, datos)
    out = tmp_path / "labels.csv"
    with pytest.raises(ValueError, match="Tasa de churn sospechosa"):
        write_labelled_dataset(store, out, min_rows=500)
    assert not out.exists()


def test_write_rechaza_formatos_no_soportados(store, tmp_path):
    with pytest.raises(ValueError, match="Formato no soportado"):
        write_labelled_dataset(store, tmp_path / "labels.xlsx", min_rows=1)


def test_min_rows_por_debajo_del_validador_falla_en_la_validacion(store, tmp_path):
    score_and_label(store, generate_dataset(100, seed=5))
    with pytest.raises(ValueError, match="demasiado pequeno"):
        write_labelled_dataset(store, tmp_path / "labels.csv", min_rows=50)


# --------------------------------------------------------------------------- CLI


def _summary(capsys) -> dict:
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert len(lines) == 1  # stdout = solo el resumen JSON; los logs van a stderr
    assert '"logger"' in captured.err
    return json.loads(lines[0])


def test_cli_construye_el_dataset_por_defecto_dentro_de_model_dir(tmp_path, capsys):
    model_dir = tmp_path / "models"
    store = SqlitePredictionStore(model_dir / "drift.sqlite", max_rows=5000)
    score_and_label(store, generate_dataset(600, seed=6), observed_at=OBSERVED)

    assert main(["--model-dir", str(model_dir)]) == EXIT_OK

    summary = _summary(capsys)
    assert summary["status"] == "ok"
    assert summary["path"] == str(model_dir / "datasets" / "labels.parquet")
    assert summary["rows"] == 600
    assert len(load_training_data(summary["path"])) == 600
    assert read_sidecar(summary["path"])["fingerprint"] == summary["fingerprint"]


def test_cli_acepta_db_out_y_min_rows_explicitos(tmp_path, capsys):
    db = tmp_path / "otro" / "predicciones.sqlite"
    score_and_label(SqlitePredictionStore(db, max_rows=5000), generate_dataset(700, seed=7))
    out = tmp_path / "salida" / "labels.csv"

    argv = ["--db", str(db), "--out", str(out), "--min-rows", "650"]
    assert main(argv) == EXIT_OK
    assert _summary(capsys)["path"] == str(out)
    assert len(pd.read_csv(out)) == 700

    assert main([*argv[:-1], "750"]) == EXIT_INSUFFICIENT
    assert _summary(capsys) == {"status": "insufficient", "rows": 700, "min_rows": 750}


def test_cli_etiquetas_insuficientes_devuelve_3(tmp_path, capsys):
    model_dir = tmp_path / "models"
    score_and_label(SqlitePredictionStore(model_dir / "drift.sqlite", 50), generate_dataset(20, 8))

    assert main(["--model-dir", str(model_dir)]) == EXIT_INSUFFICIENT

    assert _summary(capsys) == {"status": "insufficient", "rows": 20, "min_rows": 500}
    assert not (model_dir / "datasets").exists()


def test_cli_almacen_inexistente_devuelve_2_sin_crearlo(tmp_path, capsys):
    assert main(["--model-dir", str(tmp_path / "vacio")]) == EXIT_ERROR
    summary = _summary(capsys)
    assert summary["status"] == "error" and "No existe el almacen" in summary["error"]
    assert not (tmp_path / "vacio").exists()


def test_cli_dataset_invalido_devuelve_2(tmp_path, capsys):
    db = tmp_path / "drift.sqlite"
    datos = generate_dataset(600, seed=9)
    datos[TARGET] = 0
    score_and_label(SqlitePredictionStore(db, max_rows=5000), datos)

    assert main(["--db", str(db), "--out", str(tmp_path / "labels.parquet")]) == EXIT_ERROR
    summary = _summary(capsys)
    assert summary["status"] == "error" and "Tasa de churn" in summary["error"]


def test_cli_lee_los_valores_por_defecto_de_settings(tmp_path, monkeypatch, capsys):
    settings = Settings(
        _env_file=None,
        model_dir=str(tmp_path / "ignorado"),
        drift_db_path=str(tmp_path / "configurado.sqlite"),
        labels_min_rows=10,
    )
    monkeypatch.setattr(labels_module, "get_settings", lambda: settings)
    store = SqlitePredictionStore(settings.drift_db_path, max_rows=settings.prediction_keep_rows)
    score_and_label(store, generate_dataset(12, seed=10))

    assert main(["--out", str(tmp_path / "labels.csv")]) == EXIT_ERROR  # 12 >= 10 pero < 500
    assert "demasiado pequeno" in _summary(capsys)["error"]
    assert main(["--out", str(tmp_path / "labels.csv"), "--min-rows", "20"]) == EXIT_INSUFFICIENT
    assert _summary(capsys)["min_rows"] == 20
