import pandas as pd
import pytest

from churn.data.generator import FEATURE_COLUMNS, TARGET, DriftSpec, generate_dataset
from churn.data.sources import dataset_fingerprint, load_training_data, resolve_training_data


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
