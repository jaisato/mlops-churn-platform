import threading

import pytest

from churn.config import Settings
from churn.monitoring.store import (
    MemoryPredictionStore,
    SqlitePredictionStore,
    build_prediction_store,
)


def _row(i: int, version: str = "v1") -> dict:
    return {
        "tenure_months": i,
        "monthly_charges": 10.5 + i,
        "contract_type": "mensual",
        "churn_probability": i / 100,
        "model_version": version,
    }


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return MemoryPredictionStore(max_rows=20)
    return SqlitePredictionStore(tmp_path / "drift.sqlite", max_rows=20)


def test_vacio(store):
    assert store.count() == 0
    assert store.recent(10).empty


def test_append_y_recent_en_orden_cronologico(store):
    store.append(_row(i) for i in range(5))
    df = store.recent(10)
    assert store.count() == 5
    assert list(df["tenure_months"]) == [0, 1, 2, 3, 4]
    assert set(df.columns) == {
        "tenure_months", "monthly_charges", "contract_type", "churn_probability", "model_version",
    }


def test_recent_devuelve_las_ultimas_n(store):
    store.append(_row(i) for i in range(10))
    assert list(store.recent(3)["tenure_months"]) == [7, 8, 9]


def test_ventana_rodante_acota_el_tamano(store):
    store.append(_row(i) for i in range(30))
    assert store.count() == 20
    assert list(store.recent(100)["tenure_months"]) == list(range(10, 30))


def test_conserva_tipos_y_version(store):
    store.append([_row(3, version="v2")])
    df = store.recent(1)
    assert df.loc[0, "monthly_charges"] == 13.5
    assert df.loc[0, "contract_type"] == "mensual"
    assert df.loc[0, "model_version"] == "v2"
    assert df.loc[0, "churn_probability"] == 0.03


def test_append_vacio_no_falla(store):
    store.append([])
    assert store.count() == 0


def test_sqlite_persiste_entre_instancias(tmp_path):
    path = tmp_path / "drift.sqlite"
    SqlitePredictionStore(path, max_rows=50).append(_row(i) for i in range(7))
    reabierto = SqlitePredictionStore(path, max_rows=50)
    assert reabierto.count() == 7
    assert list(reabierto.recent(2)["tenure_months"]) == [5, 6]


def test_sqlite_crea_el_directorio_padre(tmp_path):
    store = SqlitePredictionStore(tmp_path / "sub" / "dir" / "drift.sqlite", max_rows=5)
    store.append([_row(1)])
    assert (tmp_path / "sub" / "dir" / "drift.sqlite").exists()


def test_sqlite_escrituras_concurrentes(tmp_path):
    store = SqlitePredictionStore(tmp_path / "drift.sqlite", max_rows=10_000)

    def worker(offset: int) -> None:
        store.append(_row(offset * 100 + i) for i in range(50))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.count() == 400


def test_build_prediction_store_segun_settings(tmp_path):
    memoria = build_prediction_store(
        Settings(_env_file=None, drift_store="memory", model_dir=str(tmp_path))
    )
    assert isinstance(memoria, MemoryPredictionStore)

    por_defecto = build_prediction_store(Settings(_env_file=None, model_dir=str(tmp_path)))
    assert isinstance(por_defecto, SqlitePredictionStore)
    assert por_defecto.path == tmp_path / "drift.sqlite"

    explicito = build_prediction_store(
        Settings(_env_file=None, model_dir=str(tmp_path), drift_db_path=str(tmp_path / "x.db"))
    )
    assert explicito.path == tmp_path / "x.db"
