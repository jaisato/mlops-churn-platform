import sqlite3
import threading
from datetime import UTC, datetime, timedelta, timezone

import pytest

from churn.config import Settings
from churn.monitoring.store import (
    LabelOutcome,
    MemoryPredictionStore,
    SqlitePredictionStore,
    build_prediction_store,
)

OBSERVED = "2026-10-01T12:00:00+00:00"


def _row(i: int, version: str = "v1") -> dict:
    return {
        "tenure_months": i,
        "monthly_charges": 10.5 + i,
        "contract_type": "mensual",
        "churn_probability": i / 100,
        "model_version": version,
    }


def _label(prediction_id: str, churn: int = 1, observed_at=OBSERVED) -> dict:
    return {"prediction_id": prediction_id, "churn": churn, "observed_at": observed_at}


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return MemoryPredictionStore(max_rows=20)
    return SqlitePredictionStore(tmp_path / "drift.sqlite", max_rows=20)


# --------------------------------------------------------------------------- predicciones


def test_vacio(store):
    assert store.count() == 0
    assert store.recent(10).empty


def test_append_y_recent_en_orden_cronologico(store):
    ids = store.append(_row(i) for i in range(5))
    df = store.recent(10)
    assert store.count() == 5
    assert list(df["tenure_months"]) == [0, 1, 2, 3, 4]
    assert set(df.columns) == {
        "tenure_months",
        "monthly_charges",
        "contract_type",
        "churn_probability",
        "model_version",
        "prediction_id",
        "predicted_at",
    }
    assert list(df["prediction_id"]) == ids
    for predicted_at in df["predicted_at"]:
        assert datetime.fromisoformat(predicted_at).tzinfo is not None


def test_append_devuelve_un_id_unico_por_prediccion(store):
    ids = store.append(_row(i) for i in range(50))
    assert len(ids) == 50 and len(set(ids)) == 50
    assert all(len(i) == 32 and int(i, 16) >= 0 for i in ids)  # uuid4 en hexadecimal
    assert not set(ids) & set(store.append([_row(1), _row(2)]))


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
    assert store.append([]) == []
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
    assert por_defecto.max_rows == 100_000  # retencion de etiquetado, no la ventana de drift

    explicito = build_prediction_store(
        Settings(_env_file=None, model_dir=str(tmp_path), drift_db_path=str(tmp_path / "x.db"))
    )
    assert explicito.path == tmp_path / "x.db"


# --------------------------------------------------------------------------- etiquetas


def test_sin_etiquetas(store):
    store.append([_row(1)])
    assert store.labelled_count() == 0
    assert store.labelled().empty
    assert store.label([]) == LabelOutcome()


def test_label_crea_corrige_y_reporta_desconocidos(store):
    ids = store.append(_row(i) for i in range(3))

    primero = store.label([_label(ids[0], churn=1), _label("inexistente", churn=0)])
    assert primero == LabelOutcome(created=1, updated=0, unknown=["inexistente"])
    assert store.labelled_count() == 1

    # Idempotente por prediction_id: reenviar no duplica, corregir sobrescribe
    segundo = store.label([_label(ids[0], churn=0), _label(ids[1], churn=1)])
    assert segundo == LabelOutcome(created=1, updated=1, unknown=[])
    assert store.labelled_count() == 2
    etiquetas = store.labelled().set_index("prediction_id")["churn"].to_dict()
    assert etiquetas == {ids[0]: 0, ids[1]: 1}  # la ultima etiqueta recibida es la que vale


def test_labelled_lleva_features_probabilidad_version_tiempos_y_etiqueta(store):
    [prediction_id] = store.append([_row(3, version="v2")])
    predicted_at = store.recent(1).loc[0, "predicted_at"]
    store.label([_label(prediction_id, churn=1)])

    fila = store.labelled().iloc[0].to_dict()
    assert fila["tenure_months"] == 3
    assert fila["monthly_charges"] == 13.5
    assert fila["contract_type"] == "mensual"
    assert fila["churn_probability"] == 0.03
    assert fila["model_version"] == "v2"
    assert fila["prediction_id"] == prediction_id
    assert fila["predicted_at"] == predicted_at
    assert fila["observed_at"] == OBSERVED
    assert datetime.fromisoformat(fila["labelled_at"]) >= datetime.fromisoformat(predicted_at)
    assert fila["churn"] == 1


def test_observed_at_se_normaliza_a_utc(store):
    ids = store.append(_row(i) for i in range(4))
    madrid = timezone(timedelta(hours=2))
    store.label(
        [
            _label(ids[0], observed_at=datetime(2026, 10, 1, 14, 0, tzinfo=madrid)),
            _label(ids[1], observed_at="2026-10-01T14:00:00+02:00"),
            _label(ids[2], observed_at=datetime(2026, 10, 1, 12, 0)),  # naive = UTC
            _label(ids[3], observed_at=datetime(2026, 10, 1, 12, 0, tzinfo=UTC)),
        ]
    )
    assert set(store.labelled()["observed_at"]) == {OBSERVED}


def test_las_etiquetas_sobreviven_al_desalojo_de_la_ventana(store):
    ids = store.append(_row(i) for i in range(5))
    store.label([_label(ids[0], churn=1)])

    store.append(_row(i) for i in range(100, 130))  # desaloja las 5 primeras (max_rows=20)

    assert store.count() == 20
    assert store.labelled_count() == 1
    assert store.labelled().loc[0, "tenure_months"] == 0
    # ...pero una prediccion ya desalojada no se puede etiquetar (ni reetiquetar)
    assert store.label([_label(ids[1]), _label(ids[0], churn=0)]).unknown == [ids[1], ids[0]]
    assert store.labelled().loc[0, "churn"] == 1


def test_labelled_en_orden_de_prediccion_y_con_limite(store):
    ids = store.append(_row(i) for i in range(6))
    store.label(_label(i) for i in reversed(ids))  # etiquetadas en orden inverso

    assert list(store.labelled()["tenure_months"]) == [0, 1, 2, 3, 4, 5]  # por prediccion
    assert list(store.labelled(2)["tenure_months"]) == [4, 5]  # las mas recientes
    assert store.labelled(0).empty
    assert len(store.labelled(100)) == 6


def test_sqlite_las_etiquetas_persisten_entre_instancias(tmp_path):
    path = tmp_path / "drift.sqlite"
    primero = SqlitePredictionStore(path, max_rows=50)
    ids = primero.append(_row(i) for i in range(3))
    primero.label([_label(ids[2], churn=1)])

    reabierto = SqlitePredictionStore(path, max_rows=50)
    assert reabierto.labelled_count() == 1
    assert reabierto.labelled().loc[0, "prediction_id"] == ids[2]
    assert reabierto.label([_label(ids[2], churn=0)]) == LabelOutcome(updated=1)


def test_sqlite_migra_un_fichero_anterior_al_bucle_de_etiquetas(tmp_path):
    path = tmp_path / "drift.sqlite"
    with sqlite3.connect(path) as conn:  # esquema de la version 1.2: sin prediction_id ni labels
        conn.execute(
            "CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
            " model_version TEXT NOT NULL, churn_probability REAL NOT NULL, features TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO predictions (ts, model_version, churn_probability, features)"
            " VALUES ('2026-01-01T00:00:00+00:00', 'v0', 0.5, '{\"tenure_months\": 9}')"
        )
    conn.close()

    store = SqlitePredictionStore(path, max_rows=20)
    antigua = store.recent(10).iloc[0].to_dict()
    assert antigua["tenure_months"] == 9 and antigua["model_version"] == "v0"
    assert antigua["prediction_id"] is None  # no etiquetable: se desalojara con la ventana
    assert antigua["predicted_at"] == "2026-01-01T00:00:00+00:00"

    ids = store.append([_row(1)])
    assert store.count() == 2
    assert store.label([_label(ids[0])]) == LabelOutcome(created=1)
    assert SqlitePredictionStore(path, max_rows=20).labelled_count() == 1  # reabrir es inocuo


def test_sqlite_etiquetado_concurrente(tmp_path):
    store = SqlitePredictionStore(tmp_path / "drift.sqlite", max_rows=10_000)
    ids = store.append(_row(i) for i in range(200))

    def worker(chunk: list[str]) -> None:
        store.label(_label(i) for i in chunk)

    threads = [threading.Thread(target=worker, args=(ids[n::4],)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.labelled_count() == 200
