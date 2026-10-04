import json
from pathlib import Path

import pandas as pd
import pytest

from churn.data.generator import FEATURE_COLUMNS, TARGET, DriftSpec, generate_dataset
from churn.data.sources import (
    LABELS_WINDOW_KEYS,
    dataset_fingerprint,
    load_training_data,
    read_sidecar,
    resolve_training_data,
    sidecar_path,
)


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return generate_dataset(600, seed=3)


# ------------------------------------------------------------------ carga de ficheros


def test_carga_csv(tmp_path, sample):
    path = tmp_path / "datos.csv"
    sample.to_csv(path, index=False)
    df = load_training_data(path)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET]
    assert len(df) == 600


def test_carga_parquet_y_conserva_dtypes(tmp_path, sample):
    path = tmp_path / "datos.parquet"
    sample.to_parquet(path, index=False)
    df = load_training_data(str(path))  # acepta str y Path
    assert df.equals(sample)


def test_extension_insensible_a_mayusculas(tmp_path, sample):
    path = tmp_path / "DATOS.CSV"
    sample.to_csv(path, index=False)
    assert len(load_training_data(path)) == 600


def test_fichero_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError, match="no encontrado"):
        load_training_data(tmp_path / "nada.csv")


def test_directorio_no_es_un_fichero(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_training_data(tmp_path)


def test_formato_no_soportado(tmp_path):
    path = tmp_path / "datos.xlsx"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="Formato no soportado"):
        load_training_data(path)


def test_csv_vacio(tmp_path):
    path = tmp_path / "vacio.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="No se puede leer"):
        load_training_data(path)


def test_csv_solo_cabecera(tmp_path):
    path = tmp_path / "cabecera.csv"
    path.write_text(",".join(FEATURE_COLUMNS + [TARGET]) + "\n")
    with pytest.raises(ValueError, match="no contiene filas"):
        load_training_data(path)


def test_parquet_corrupto(tmp_path):
    path = tmp_path / "roto.parquet"
    path.write_text("esto no es parquet")
    with pytest.raises(ValueError, match="No se puede leer"):
        load_training_data(path)


def test_parquet_sin_motor_instalado(tmp_path, monkeypatch):
    path = tmp_path / "datos.parquet"
    path.write_bytes(b"PAR1")

    def sin_pyarrow(*args, **kwargs):
        raise ImportError("Unable to find a usable engine")

    monkeypatch.setattr(pd, "read_parquet", sin_pyarrow)
    with pytest.raises(ValueError, match="usable engine"):
        load_training_data(path)


# ------------------------------------------------------------------ huella


def test_fingerprint_estable_y_sensible(sample):
    assert dataset_fingerprint(sample) == dataset_fingerprint(sample.copy())
    assert dataset_fingerprint(sample) == dataset_fingerprint(generate_dataset(600, seed=3))
    assert len(dataset_fingerprint(sample)) == 64

    otra_fila = sample.copy()
    otra_fila.loc[0, "tenure_months"] = otra_fila.loc[0, "tenure_months"] + 1
    assert dataset_fingerprint(otra_fila) != dataset_fingerprint(sample)
    assert dataset_fingerprint(sample.head(599)) != dataset_fingerprint(sample)
    assert dataset_fingerprint(generate_dataset(600, seed=4)) != dataset_fingerprint(sample)


def test_fingerprint_ignora_el_indice_pero_no_los_nombres(sample):
    reindexado = sample.set_index(pd.RangeIndex(1000, 1600))
    assert dataset_fingerprint(reindexado) == dataset_fingerprint(sample)
    renombrado = sample.rename(columns={"tenure_months": "antiguedad"})
    assert dataset_fingerprint(renombrado) != dataset_fingerprint(sample)


# ------------------------------------------------------------------ resolucion


def test_resolve_sintetico_describe_la_fuente():
    df, source = resolve_training_data(rows=700, seed=9, drift_shift=0.5)
    assert df.equals(generate_dataset(700, seed=9, drift=DriftSpec.from_shift(0.5)))
    assert source == {
        "kind": "synthetic",
        "seed": 9,
        "drift_shift": 0.5,
        "rows": 700,
        "fingerprint": dataset_fingerprint(df),
    }


def test_resolve_fichero_describe_la_fuente(tmp_path, sample):
    path = tmp_path / "q3.csv"
    sample.to_csv(path, index=False)
    df, source = resolve_training_data(data_path=path)
    assert len(df) == 600
    assert source == {
        "kind": "file",
        "path": str(path),
        "rows": 600,
        "fingerprint": dataset_fingerprint(df),
    }


def test_resolve_rechaza_drift_shift_con_fichero(tmp_path, sample):
    path = tmp_path / "q3.csv"
    sample.to_csv(path, index=False)
    with pytest.raises(ValueError, match="solo tiene sentido con datos sinteticos"):
        resolve_training_data(data_path=path, drift_shift=1.0)


def test_resolve_propaga_errores_del_fichero(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_training_data(data_path=tmp_path / "nada.parquet")


# ------------------------------------------------------------------ sidecar de etiquetas


def _sidecar(path, fingerprint: str) -> dict:
    payload = {
        "kind": "labels",
        "fingerprint": fingerprint,
        "rows": 600,
        "built_at": "2026-10-04T10:00:00+00:00",
        "predicted_from": "2026-09-01T00:00:00+00:00",
        "predicted_to": "2026-09-30T23:59:59+00:00",
        "observed_from": "2026-09-15T00:00:00+00:00",
        "observed_to": "2026-10-03T00:00:00+00:00",
        "model_versions": ["v1", "v2"],
        "churn_rate": 0.27,
        "labels_total": 640,
        "subjects": 580,
    }
    sidecar_path(path).write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_sidecar_path_acompana_al_dataset():
    assert sidecar_path("data/labels.parquet") == Path("data/labels.parquet.meta.json")
    assert sidecar_path(Path("/models/datasets/labels.csv")).name == "labels.csv.meta.json"


def test_resolve_con_sidecar_coincidente_describe_la_fuente_como_labels(tmp_path, sample):
    path = tmp_path / "labels.parquet"
    sample.to_parquet(path, index=False)
    payload = _sidecar(path, dataset_fingerprint(load_training_data(path)))

    df, source = resolve_training_data(data_path=path)

    assert len(df) == 600
    assert source == {
        "kind": "file",  # no "labels": una imagen anterior no conoceria ese valor (rollback)
        "path": str(path),
        "rows": 600,
        "fingerprint": payload["fingerprint"],
        "labels": {k: payload[k] for k in LABELS_WINDOW_KEYS},  # sin rows/churn_rate/kind
    }
    assert set(source["labels"]) == {
        "built_at",
        "predicted_from",
        "predicted_to",
        "observed_from",
        "observed_to",
        "model_versions",
        "labels_total",
        "subjects",
    }


def test_resolve_ignora_un_sidecar_cuya_huella_no_coincide(tmp_path, sample, caplog):
    path = tmp_path / "labels.csv"
    sample.to_csv(path, index=False)
    _sidecar(path, "0" * 64)  # el fichero se edito despues de construirlo

    _, source = resolve_training_data(data_path=path)

    assert source["kind"] == "file" and "labels" not in source
    assert "no coincide" in caplog.text


def test_read_sidecar_tolera_ausente_ilegible_o_que_no_es_un_objeto(tmp_path, caplog):
    path = tmp_path / "labels.csv"
    assert read_sidecar(path) is None  # no existe

    sidecar_path(path).write_text("{rota", encoding="utf-8")
    assert read_sidecar(path) is None
    assert "ilegible" in caplog.text

    sidecar_path(path).write_text("[1, 2, 3]", encoding="utf-8")
    assert read_sidecar(path) is None

    sidecar_path(path).write_text('{"fingerprint": "x"}', encoding="utf-8")
    assert read_sidecar(path) == {"fingerprint": "x"}


def test_sidecar_incompleto_rellena_la_ventana_con_nulos(tmp_path, sample):
    path = tmp_path / "labels.csv"
    sample.to_csv(path, index=False)
    fingerprint = dataset_fingerprint(load_training_data(path))
    sidecar_path(path).write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")

    _, source = resolve_training_data(data_path=path)

    assert source["kind"] == "file"
    assert source["labels"] == dict.fromkeys(LABELS_WINDOW_KEYS)
