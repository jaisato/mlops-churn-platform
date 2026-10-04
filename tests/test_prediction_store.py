import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone

import pytest

from churn.config import Settings
from churn.monitoring import store as store_module
from churn.monitoring.store import (
    LabelOutcome,
    MemoryPredictionStore,
    SqlitePredictionStore,
    build_prediction_store,
)

REAL_UTCNOW = store_module._utcnow  # capturado al importar, antes del reloj de conftest
OBSERVED = "2026-10-01T12:00:00+00:00"
# Esquema de drift.sqlite en la version 1.2 (antes del bucle de etiquetas)
SCHEMA_1_2 = (
    "CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
    " model_version TEXT NOT NULL, churn_probability REAL NOT NULL, features TEXT NOT NULL)"
)


def _row(i: int, version: str = "v1", subject: str | None = None) -> dict:
    row = {
        "tenure_months": i,
        "monthly_charges": 10.5 + i,
        "contract_type": "mensual",
        "churn_probability": i / 100,
        "model_version": version,
    }
    if subject is not None:
        row["subject_ref"] = subject
    return row


def _label(prediction_id: str, churn: int = 1, observed_at=OBSERVED, **extra) -> dict:
    return {"prediction_id": prediction_id, "churn": churn, "observed_at": observed_at, **extra}


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
        "subject_ref",
        "predicted_at",
    }
    assert list(df["prediction_id"]) == ids
    assert df["subject_ref"].isna().all()  # el cliente no envio sujeto
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


def test_ventana_rodante_prediccion_a_prediccion(store):
    for i in range(45):  # como /predict: una fila por llamada
        store.append([_row(i)])
    assert store.count() == 20
    assert list(store.recent(100)["tenure_months"]) == list(range(25, 45))


def test_retencion_por_antiguedad(tmp_path, store_clock):
    start = store_clock.now
    store_clock.step = timedelta(0)  # el test fija el instante de cada operacion
    for store in (
        MemoryPredictionStore(max_rows=100, max_age=timedelta(days=30)),
        SqlitePredictionStore(tmp_path / "drift.sqlite", max_rows=100, max_age=timedelta(days=30)),
    ):
        ids = []
        for day in range(0, 60, 10):  # una prediccion cada diez dias
            store_clock.now = start + timedelta(days=day)
            ids.append(store.append([_row(day)])[0])
            if day == 0:
                store.label([_label(ids[0], observed_at=start + timedelta(days=1))])
        # el dia 50 caducan las anteriores al dia 20 (la del dia 20 justo entra)
        assert store.count() == 4
        assert list(store.recent(100)["prediction_id"]) == ids[2:]
        assert store.labelled_count() == 1  # las etiquetadas se conservan aparte
        assert store.label([_label(ids[1])]).unknown == [ids[1]]  # caducada sin etiquetar


def test_retencion_por_antiguedad_desactivada_por_defecto(store, store_clock):
    store_clock.step = timedelta(days=365)
    store.append(_row(i) for i in range(3))
    store.append([_row(3)])
    assert store.count() == 4 and store.max_age is None


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
    assert memoria.max_age is None

    por_defecto = build_prediction_store(Settings(_env_file=None, model_dir=str(tmp_path)))
    assert isinstance(por_defecto, SqlitePredictionStore)
    assert por_defecto.path == tmp_path / "drift.sqlite"
    assert por_defecto.max_rows == 100_000  # retencion de etiquetado, no la ventana de drift
    assert por_defecto.max_age is None

    con_dias = build_prediction_store(
        Settings(_env_file=None, model_dir=str(tmp_path), prediction_keep_days=120)
    )
    assert con_dias.max_age == timedelta(days=120)  # type: ignore[attr-defined]

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
    # una prediccion desalojada sin etiquetar ya no se puede etiquetar...
    assert store.label([_label(ids[1])]) == LabelOutcome(unknown=[ids[1]])
    assert store.labelled_count() == 1


def test_una_etiqueta_se_puede_corregir_aunque_su_prediccion_ya_se_desalojara(store):
    ids = store.append(_row(i, subject="c-1" if i == 0 else None) for i in range(5))
    store.label([_label(ids[0], churn=1)])
    antes = store.labelled().iloc[0].to_dict()
    store.append(_row(i) for i in range(100, 130))  # desaloja ids[0]
    assert ids[0] not in set(store.recent(100)["prediction_id"])

    # ...pero la copia etiquetada sigue en el almacen: la correccion tardia la actualiza
    corregida = store.label([_label(ids[0], churn=0, observed_at="2026-10-02T00:00:00Z")])

    assert corregida == LabelOutcome(updated=1)
    despues = store.labelled().iloc[0].to_dict()
    assert despues["churn"] == 0
    assert despues["observed_at"] == "2026-10-02T00:00:00+00:00"
    assert despues["labelled_at"] > antes["labelled_at"]
    assert despues["subject_ref"] == "c-1"  # sin sujeto en la correccion, se conserva
    sin_cambios = {"tenure_months", "churn_probability", "model_version", "predicted_at"}
    assert {k: despues[k] for k in sin_cambios} == {k: antes[k] for k in sin_cambios}
    assert store.labelled_count() == 1
    assert store.label([_label(ids[0], subject_ref="c-9")]) == LabelOutcome(updated=1)
    assert store.labelled().loc[0, "subject_ref"] == "c-9"


def test_rechaza_etiquetas_observadas_antes_de_la_prediccion(store):
    ids = store.append(_row(i) for i in range(3))
    predicted_at = datetime.fromisoformat(store.recent(1).loc[0, "predicted_at"])

    outcome = store.label(
        [
            _label(ids[0], observed_at=predicted_at - timedelta(days=1)),  # imposible
            _label(ids[1], observed_at=predicted_at),  # en el mismo instante: vale
            _label("desconocido"),
        ]
    )

    assert outcome == LabelOutcome(created=1, unknown=["desconocido"], rejected=[ids[0]])
    assert list(store.labelled()["prediction_id"]) == [ids[1]]
    # tambien al corregir una etiqueta cuya prediccion ya se desalojo
    store.append(_row(i) for i in range(100, 130))
    tarde = store.label([_label(ids[1], observed_at=predicted_at - timedelta(hours=1))])
    assert tarde == LabelOutcome(rejected=[ids[1]])
    assert store.labelled().loc[0, "observed_at"] == predicted_at.isoformat()  # sin cambios


def test_subject_ref_viaja_con_la_prediccion_y_la_etiqueta(store):
    ids = store.append([_row(1, subject="c-1"), _row(2), _row(3, subject="c-3")])
    assert list(store.recent(3)["subject_ref"]) == ["c-1", None, "c-3"]

    store.label(
        [
            _label(ids[0]),  # hereda el sujeto de la prediccion
            _label(ids[1], subject_ref="c-2"),  # el cliente lo aporta al etiquetar
            _label(ids[2], subject_ref="c-3b"),  # la etiqueta manda
        ]
    )
    sujetos = store.labelled().set_index("prediction_id")["subject_ref"].to_dict()
    assert sujetos == {ids[0]: "c-1", ids[1]: "c-2", ids[2]: "c-3b"}

    store.label([_label(ids[1], churn=0)])  # correccion sin sujeto: se conserva el anterior
    assert store.labelled().set_index("prediction_id").loc[ids[1], "subject_ref"] == "c-2"


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
        conn.execute(SCHEMA_1_2)
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

    ids = store.append([_row(1, subject="c-1")])
    assert store.count() == 2
    assert store.label([_label(ids[0])]) == LabelOutcome(created=1)
    assert SqlitePredictionStore(path, max_rows=20).labelled_count() == 1  # reabrir es inocuo
    assert store.labelled().loc[0, "subject_ref"] == "c-1"


def test_sqlite_migra_etiquetas_sin_subject_ref(tmp_path):
    """Un drift.sqlite con la tabla labels pero sin subject_ref (version previa del bucle)."""
    path = tmp_path / "drift.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute(SCHEMA_1_2)
        conn.execute("ALTER TABLE predictions ADD COLUMN prediction_id TEXT")
        conn.execute(
            "CREATE TABLE labels (prediction_id TEXT PRIMARY KEY, prediction_seq INTEGER NOT NULL,"
            " churn INTEGER NOT NULL, observed_at TEXT NOT NULL, labelled_at TEXT NOT NULL,"
            " predicted_at TEXT NOT NULL, model_version TEXT NOT NULL,"
            " churn_probability REAL NOT NULL, features TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO labels VALUES ('p0', 1, 1, '2026-09-02T00:00:00+00:00',"
            " '2026-09-02T00:00:00+00:00', '2026-09-01T00:00:00+00:00', 'v0', 0.4,"
            " '{\"tenure_months\": 5}')"
        )
    conn.close()

    store = SqlitePredictionStore(path, max_rows=20)

    antigua = store.labelled().iloc[0].to_dict()
    assert antigua["prediction_id"] == "p0" and antigua["subject_ref"] is None
    assert store.label([_label("p0", churn=0, subject_ref="c-0")]) == LabelOutcome(updated=1)
    assert store.labelled().loc[0, "subject_ref"] == "c-0"


def test_sqlite_migracion_concurrente_espera_y_no_duplica_columnas(tmp_path):
    """Varios workers arrancan a la vez sobre un drift.sqlite antiguo.

    Determinista: otro proceso tiene el cerrojo de escritura y ya ha anadido la columna sin
    confirmar. Antes, el store leia el esquema sin la columna, esperaba al cerrojo para el
    ALTER y fallaba con "duplicate column name"; ahora espera ANTES de mirar el esquema.
    """
    path = tmp_path / "drift.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(SCHEMA_1_2)
    conn.close()
    otro_worker = sqlite3.connect(path, isolation_level=None)
    otro_worker.execute("BEGIN IMMEDIATE")
    otro_worker.execute("ALTER TABLE predictions ADD COLUMN prediction_id TEXT")

    with ThreadPoolExecutor(max_workers=1) as pool:
        arranque = pool.submit(SqlitePredictionStore, path, 10)
        time.sleep(0.3)  # el store queda esperando al cerrojo
        assert not arranque.done()
        otro_worker.execute("COMMIT")
        store = arranque.result(timeout=30)  # sin "duplicate column name"
    otro_worker.close()

    ids = store.append([_row(1)])
    assert store.label([_label(ids[0])]) == LabelOutcome(created=1)


def test_sqlite_migracion_fallida_no_deja_el_esquema_a_medias(tmp_path, monkeypatch):
    path = tmp_path / "drift.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute(SCHEMA_1_2)
    conn.close()
    real = SqlitePredictionStore._add_missing_columns

    def falla_a_mitad(conn, table, columns):
        real(conn, table, columns)
        raise sqlite3.OperationalError("disco lleno")

    monkeypatch.setattr(SqlitePredictionStore, "_add_missing_columns", falla_a_mitad)
    with pytest.raises(sqlite3.OperationalError, match="disco lleno"):
        SqlitePredictionStore(path, max_rows=10)

    with sqlite3.connect(path) as conn:  # ROLLBACK: ni columnas nuevas ni tabla labels
        columnas = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        tablas = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    conn.close()
    assert "prediction_id" not in columnas and "labels" not in tablas
    monkeypatch.setattr(SqlitePredictionStore, "_add_missing_columns", real)
    assert SqlitePredictionStore(path, max_rows=10).count() == 0  # el siguiente arranque migra


def test_el_reloj_real_del_almacen_es_utc():
    # conftest sustituye store._utcnow por un reloj determinista; aqui, el de verdad
    antes = datetime.now(UTC)
    ahora = REAL_UTCNOW()
    assert ahora.tzinfo is UTC and antes <= ahora <= datetime.now(UTC)
    assert store_module._utcnow is not REAL_UTCNOW


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
