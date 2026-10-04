"""Reentrenamiento real: drift -> datos etiquetados nuevos -> retador -> promocion (o no).

Demuestra el problema que motiva esta pieza y su solucion:
  - reentrenar con el mismo generador y semilla (lo que hacia el cron) produce un modelo
    identico (misma huella del dataset, mismas predicciones, misma referencia) y el drift
    medido contra su referencia sigue exactamente igual;
  - reentrenar con datos etiquetados del mundo nuevo produce otro modelo que gana al
    campeon sobre el holdout nuevo, se promueve, y el drift medido contra su referencia
    desaparece;
  - un retador peor que el campeon se guarda sin promover y el servicio no cambia.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from churn.data.generator import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    DriftSpec,
    generate_dataset,
)
from churn.data.labels import write_labelled_dataset
from churn.data.sources import resolve_training_data
from churn.monitoring.drift import drift_report
from churn.monitoring.store import SqlitePredictionStore
from churn.registry import LocalModelStore
from churn.serving.schemas import ModelInfoResponse
from churn.training import train as train_module
from churn.training.train import (
    EXIT_NOT_PROMOTED,
    EXIT_OK,
    MIN_COMPARISON_GROUPS,
    holdout_mask,
    main,
    split_groups,
    train,
)
from tests.conftest import ADMIN_HEADERS, score_and_label, weak_pipeline

ROWS = 3000
DRIFT = DriftSpec.from_shift(1.0)
SYNTHETIC_42 = {"kind": "synthetic", "seed": 42, "drift_shift": 0.0}


@pytest.fixture(scope="module")
def worlds():
    """`old`: mundo con el que se entreno el campeon; `new`: datos etiquetados del mundo
    desplazado; `traffic`: lo que llega hoy a la API (mismo mundo nuevo, otra muestra)."""
    old = generate_dataset(ROWS, seed=42)
    new = generate_dataset(ROWS, seed=43, drift=DRIFT)
    traffic = generate_dataset(1500, seed=44, drift=DRIFT)[FEATURE_COLUMNS]
    return old, new, traffic


def _drifted_features(reference, traffic) -> list[str]:
    report = drift_report(reference, traffic, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    return report["drifted_features"]


# ------------------------------------------------------------------ antes / despues


def test_reentrenar_con_el_mismo_generador_no_cambia_nada(tmp_path, worlds, caplog):
    """Comportamiento anterior: el cron reentrenaba con generate_dataset(seed=42) otra vez."""
    old, _, traffic = worlds
    store = LocalModelStore(tmp_path)
    v1 = train(old, model_dir=str(tmp_path), seed=42, data_source=SYNTHETIC_42)
    reference_1 = store.load().reference
    drifted_before = _drifted_features(reference_1, traffic)
    assert {"tenure_months", "monthly_charges"} <= set(drifted_before)  # hay drift real

    v2 = train(old, model_dir=str(tmp_path), seed=42, data_source=SYNTHETIC_42)  # "reentreno"

    assert v2["data_source"]["fingerprint"] == v1["data_source"]["fingerprint"]
    assert v2["promotion"]["identical_training_data"] is True
    assert v2["promotion"]["champion_version"] == v1["model_version"]
    # Ahora se detecta y NO se promueve: el campeon sigue en servicio
    assert v2["promotion"]["decision"] == "rejected"
    assert "mismos datos" in v2["promotion"]["reason"]
    assert "el retador no aporta nada" in caplog.text
    assert store.current_version() == v1["model_version"]
    p1 = store.load(v1["model_version"]).pipeline.predict_proba(traffic)[:, 1]
    p2 = store.load(v2["model_version"]).pipeline.predict_proba(traffic)[:, 1]
    assert np.array_equal(p1, p2)  # mismo modelo con otro nombre
    reference_2 = store.load().reference
    assert reference_2.reset_index(drop=True).equals(reference_1.reset_index(drop=True))
    assert _drifted_features(reference_2, traffic) == drifted_before  # el drift sigue ahi


def test_reentrenar_con_datos_del_mundo_nuevo_corrige_el_drift(tmp_path, worlds):
    old, new, traffic = worlds
    store = LocalModelStore(tmp_path)
    v1 = train(old, model_dir=str(tmp_path), seed=42, data_source=SYNTHETIC_42)
    assert _drifted_features(store.load().reference, traffic)  # drift contra la referencia vieja

    v2 = train(
        new, model_dir=str(tmp_path), seed=42, data_source={"kind": "file", "path": "q3.csv"}
    )

    promotion = v2["promotion"]
    assert promotion["decision"] == "promoted"
    assert promotion["champion_version"] == v1["model_version"]
    assert promotion["identical_training_data"] is False
    assert promotion["champion_roc_auc"] < promotion["challenger_roc_auc"] - 0.03
    assert promotion["challenger_roc_auc"] == v2["metrics"]["roc_auc"]
    assert v2["data_source"] == {
        "kind": "file",
        "path": "q3.csv",
        "rows": ROWS,
        "fingerprint": v2["data_source"]["fingerprint"],
    }
    assert v2["data_source"]["fingerprint"] != v1["data_source"]["fingerprint"]
    assert store.current_version() == v2["model_version"]

    p1 = store.load(v1["model_version"]).pipeline.predict_proba(traffic)[:, 1]
    p2 = store.load().pipeline.predict_proba(traffic)[:, 1]
    assert not np.allclose(p1, p2)  # otro modelo de verdad
    assert _drifted_features(store.load().reference, traffic) == []  # y el drift desaparece


def test_primera_version_se_promueve_sin_campeon(tmp_path, worlds):
    old, _, _ = worlds
    metadata = train(old, model_dir=str(tmp_path), seed=42)
    assert metadata["promotion"] == {
        "decision": "no_champion",
        "margin": 0.0,
        "challenger_roc_auc": metadata["metrics"]["roc_auc"],
        "champion_version": None,
        "champion_roc_auc": None,
        "identical_training_data": False,
        "evaluated_rows": metadata["metrics"]["n_test"],
        "evaluated_groups": metadata["split"]["groups_test"],
        "champion_cutoff": None,
        "reason": "se promueve: no hay modelo en servicio",
    }
    assert metadata["data_source"]["kind"] == "dataframe"  # llamada directa, sin CLI
    assert metadata["data_source"]["rows"] == ROWS
    assert len(metadata["data_source"]["fingerprint"]) == 64


# ------------------------------------------------------------------ promocion


def test_retador_peor_se_guarda_sin_promover(tmp_path, worlds, monkeypatch, caplog):
    old, new, _ = worlds
    store = LocalModelStore(tmp_path)
    champion = train(old, model_dir=str(tmp_path), seed=42)

    monkeypatch.setattr(train_module, "build_pipeline", weak_pipeline)
    challenger = train(new, model_dir=str(tmp_path), seed=42, min_roc_auc=0.5)

    promotion = challenger["promotion"]
    assert promotion["decision"] == "rejected"
    assert promotion["champion_version"] == champion["model_version"]
    assert promotion["challenger_roc_auc"] < promotion["champion_roc_auc"]
    assert "se mantiene el campeon" in promotion["reason"]
    assert "guardada SIN promover" in caplog.text
    # el servicio no cambia, pero la version queda auditable y recuperable
    assert store.current_version() == champion["model_version"]
    assert store.list_versions() == [champion["model_version"], challenger["model_version"]]
    persisted = json.loads(
        (store.version_dir(challenger["model_version"]) / "metadata.json").read_text()
    )
    assert persisted["promotion"]["decision"] == "rejected"
    described = {d["version"]: d for d in store.describe_versions()}
    assert described[challenger["model_version"]] | {"trained_at": None} == {
        "version": challenger["model_version"],
        "current": False,
        "complete": True,
        "trained_at": None,
        "roc_auc": challenger["metrics"]["roc_auc"],
        "promotion": "rejected",
    }
    assert described[champion["model_version"]]["promotion"] == "no_champion"


def test_el_margen_permite_promover_un_retador_algo_peor(tmp_path, worlds, monkeypatch):
    old, new, _ = worlds
    champion = train(old, model_dir=str(tmp_path), seed=42)
    monkeypatch.setattr(train_module, "build_pipeline", weak_pipeline)
    challenger = train(new, model_dir=str(tmp_path), seed=42, min_roc_auc=0.5, promotion_margin=1.0)
    assert challenger["promotion"]["decision"] == "promoted"
    assert challenger["promotion"]["margin"] == 1.0
    assert challenger["promotion"]["champion_version"] == champion["model_version"]
    assert LocalModelStore(tmp_path).current_version() == challenger["model_version"]


def test_margen_por_defecto_sale_de_settings(tmp_path, worlds, monkeypatch):
    from churn.config import Settings

    old, new, _ = worlds
    train(old, model_dir=str(tmp_path), seed=42)
    monkeypatch.setattr(
        train_module, "get_settings", lambda: Settings(_env_file=None, promotion_margin=1.0)
    )
    monkeypatch.setattr(train_module, "build_pipeline", weak_pipeline)
    challenger = train(new, model_dir=str(tmp_path), seed=42, min_roc_auc=0.5)
    assert challenger["promotion"]["decision"] == "promoted"
    assert challenger["promotion"]["margin"] == 1.0


def test_campeon_ilegible_no_bloquea_la_promocion(tmp_path, worlds):
    old, new, _ = worlds
    store = LocalModelStore(tmp_path)
    champion = train(old, model_dir=str(tmp_path), seed=42)
    (store.version_dir(champion["model_version"]) / "model.joblib").write_bytes(b"roto")

    challenger = train(new, model_dir=str(tmp_path), seed=42)
    promotion = challenger["promotion"]
    assert promotion["decision"] == "no_champion"
    assert promotion["champion_version"] is None
    assert "no se puede cargar" in promotion["reason"]
    assert store.current_version() == challenger["model_version"]


def test_campeon_con_otras_features_no_bloquea_la_promocion(tmp_path, worlds):
    old, new, _ = worlds
    store = LocalModelStore(tmp_path)
    champion = train(old, model_dir=str(tmp_path), seed=42)
    metadata_file = store.version_dir(champion["model_version"]) / "metadata.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["numeric_features"] = ["feature_que_ya_no_existe"]
    metadata_file.write_text(json.dumps(metadata))

    challenger = train(new, model_dir=str(tmp_path), seed=42)
    promotion = challenger["promotion"]
    assert promotion["decision"] == "no_champion"
    assert promotion["champion_version"] == champion["model_version"]
    assert promotion["champion_roc_auc"] is None
    assert "no se puede evaluar" in promotion["reason"]
    assert store.current_version() == challenger["model_version"]


# ------------------------------------------------------------------ CLI


def test_cli_devuelve_3_si_no_promueve(tmp_path, worlds, monkeypatch, capsys):
    old, new, _ = worlds
    data = tmp_path / "nuevo.csv"
    new.to_csv(data, index=False)
    train(old, model_dir=str(tmp_path / "models"), seed=42)

    monkeypatch.setattr(train_module, "build_pipeline", weak_pipeline)
    argv = ["--data", str(data), "--model-dir", str(tmp_path / "models"), "--min-auc", "0.5"]
    assert main(argv) == EXIT_NOT_PROMOTED
    out = capsys.readouterr().out
    assert "NO promovido" in out and "AUC_campeon=" in out

    monkeypatch.undo()
    assert main(argv) == EXIT_OK
    assert "(promoted)" in capsys.readouterr().out


# ------------------------------------------------------------------ API


def test_api_sigue_sirviendo_al_campeon_si_el_retador_no_se_promueve(
    client_factory, worlds, monkeypatch
):
    _, new, _ = worlds
    c, model_dir = client_factory(seed=42)
    champion = c.get("/health").json()["model_version"]

    monkeypatch.setattr(train_module, "build_pipeline", weak_pipeline)
    rejected = train(new, model_dir=model_dir, seed=42, min_roc_auc=0.5)["model_version"]

    reload = c.post("/model/reload", headers=ADMIN_HEADERS).json()
    assert reload["model_version"] == champion  # `current` no se movio
    versions = {v["version"]: v for v in c.get("/model/versions").json()["versions"]}
    assert versions[rejected]["promotion"] == "rejected" and not versions[rejected]["current"]
    info = c.get("/model/info").json()
    assert info["promotion"]["decision"] == "no_champion"
    assert info["data_source"]["kind"] == "dataframe"

    # el rollback por defecto ignora al retador (no es "anterior" a current)...
    assert c.post("/model/rollback", headers=ADMIN_HEADERS).status_code == 409
    # ...pero un operador puede ponerlo en servicio explicitamente si lo decide
    resp = c.post("/model/rollback", json={"version": rejected}, headers=ADMIN_HEADERS)
    assert resp.status_code == 200 and resp.json()["model_version"] == rejected
    assert c.get("/model/info").json()["promotion"]["decision"] == "rejected"


# ------------------------------------------------------------------ holdout estable por grupos


def _labels_dataset(store, tmp_path, name: str):
    """Dataset de reentreno desde las etiquetas del almacen, como retrain-if-drift.sh."""
    summary = write_labelled_dataset(store, tmp_path / f"{name}.parquet", min_rows=500)
    return resolve_training_data(data_path=summary["path"])


def test_holdout_estable_al_acumular_etiquetas_y_el_split_aleatorio_anterior_no(tmp_path):
    """Causa del bloqueo: el dataset de etiquetas es ACUMULADO y el split aleatorio cambia con
    el tamano, asi que el holdout del 2o reentreno contenia filas con las que entreno el
    campeon. El split por hash del grupo deja cada fila siempre del mismo lado."""
    store = SqlitePredictionStore(tmp_path / "drift.sqlite", max_rows=100_000)
    score_and_label(store, generate_dataset(3000, seed=1, drift=DRIFT))
    d1, _ = _labels_dataset(store, tmp_path, "d1")
    score_and_label(store, generate_dataset(3000, seed=2, drift=DRIFT))
    d2, _ = _labels_dataset(store, tmp_path, "d2")

    def old_split(df):  # el de antes: train_test_split(random_state=seed, stratify=y)
        train_ids, test_ids = train_test_split(
            df["prediction_id"], test_size=0.2, random_state=42, stratify=df["churn"]
        )
        return set(train_ids), set(test_ids)

    old_train_1, _ = old_split(d1)
    _, old_test_2 = old_split(d2)
    # aqui ~23 % del holdout del 2o reentreno son filas de entrenamiento del campeon
    assert len(old_train_1 & old_test_2) / len(old_test_2) > 0.15

    def new_split(df):
        in_test = holdout_mask(split_groups(df))
        return set(df.loc[~in_test, "prediction_id"]), set(df.loc[in_test, "prediction_id"])

    new_train_1, new_test_1 = new_split(d1)
    new_train_2, new_test_2 = new_split(d2)
    assert new_train_1 & new_test_2 == set()  # nada de lo que vio el campeon en el holdout
    assert new_train_1 <= new_train_2 and new_test_1 <= new_test_2  # nadie cambia de lado
    assert 0.17 < len(new_test_2) / len(d2) < 0.23
    assert np.array_equal(holdout_mask(split_groups(d2)), holdout_mask(split_groups(d2.copy())))


def test_split_por_sujeto_mantiene_juntas_las_filas_del_mismo_cliente():
    clientes = generate_dataset(400, seed=3)
    meses = pd.concat(
        [clientes.assign(tenure_months=clientes["tenure_months"] + m) for m in range(3)],
        ignore_index=True,
    )
    meses["subject_ref"] = [f"c-{i}" for i in range(400)] * 3
    meses.loc[meses.index % 7 == 0, "subject_ref"] = None  # sin sujeto: se agrupan por filas

    groups = split_groups(meses)
    in_test = pd.Series(holdout_mask(groups), index=meses.index)

    con_sujeto = meses["subject_ref"].notna()
    lados = in_test[con_sujeto].groupby(meses.loc[con_sujeto, "subject_ref"]).nunique()
    assert (lados == 1).all()  # ningun cliente a ambos lados del holdout
    assert groups[con_sujeto].str.startswith("s:").all()
    assert groups[~con_sujeto].str.startswith("f:").all()
    # filas identicas sin sujeto: mismo grupo, mismo lado
    duplicadas = pd.concat([meses.head(5), meses.head(5)], ignore_index=True)
    duplicadas["subject_ref"] = None
    assert holdout_mask(split_groups(duplicadas))[:5].tolist() == (
        holdout_mask(split_groups(duplicadas))[5:].tolist()
    )


def test_dos_reentrenos_sucesivos_promueven_al_retador_mejor(tmp_path):
    """Bloqueo: en el 2o reentreno el campeon (entrenado con las etiquetas del 1o) puntuaba su
    propio entrenamiento dentro del holdout y un retador mejor se rechazaba. Ahora el holdout
    es estable y el campeon solo se mide con las etiquetas posteriores a las suyas."""
    model_dir = str(tmp_path / "models")
    models = LocalModelStore(model_dir)
    store = SqlitePredictionStore(tmp_path / "drift.sqlite", max_rows=100_000)
    m0 = train(generate_dataset(3000, seed=100), model_dir=model_dir, seed=42)

    # 1er reentreno: etiquetas del mundo desplazado puntuadas por m0
    score_and_label(store, generate_dataset(3000, seed=101, drift=DRIFT), version="m0")
    d1, source_1 = _labels_dataset(store, tmp_path, "d1")
    m1 = train(d1, model_dir=model_dir, seed=42, data_source=source_1)
    assert m1["promotion"]["decision"] == "promoted"
    assert m1["promotion"]["champion_version"] == m0["model_version"]
    cutoff = m1["data_source"]["labels"]["predicted_to"]

    # 2o reentreno: el mundo sigue moviendose; el dataset acumula las etiquetas de ambos
    world_2 = DriftSpec.from_shift(1.5)
    score_and_label(store, generate_dataset(3000, seed=102, drift=world_2), version="m1")
    d2, source_2 = _labels_dataset(store, tmp_path, "d2")
    assert len(d2) == 6000
    m2 = train(d2, model_dir=model_dir, seed=42, data_source=source_2)

    promotion = m2["promotion"]
    assert promotion["decision"] == "promoted", promotion["reason"]
    assert promotion["champion_version"] == m1["model_version"]
    assert promotion["champion_cutoff"] == cutoff
    # solo cuentan las filas del holdout predichas despues de las etiquetas del campeon
    in_test = holdout_mask(split_groups(d2))
    fresh = in_test & (pd.to_datetime(d2["predicted_at"]) > pd.Timestamp(cutoff)).to_numpy()
    assert promotion["evaluated_rows"] == int(fresh.sum()) == promotion["evaluated_groups"]
    assert 0 < promotion["evaluated_rows"] < m2["metrics"]["n_test"]
    assert promotion["challenger_roc_auc"] > promotion["champion_roc_auc"]
    assert models.current_version() == m2["model_version"]
    # ...y de verdad es mejor: datos nuevos independientes del mundo de hoy
    independiente = generate_dataset(5000, seed=999, drift=world_2)

    def auc(version):
        pipeline = models.load(version).pipeline
        proba = pipeline.predict_proba(independiente[FEATURE_COLUMNS])[:, 1]
        return roc_auc_score(independiente["churn"], proba)

    assert auc(m2["model_version"]) > auc(m1["model_version"]) + 0.01


def test_sin_evidencia_nueva_suficiente_se_mantiene_el_campeon(tmp_path):
    model_dir = str(tmp_path / "models")
    store = SqlitePredictionStore(tmp_path / "drift.sqlite", max_rows=100_000)
    score_and_label(store, generate_dataset(2000, seed=110, drift=DRIFT))
    d1, source_1 = _labels_dataset(store, tmp_path, "d1")
    champion = train(d1, model_dir=model_dir, seed=42, data_source=source_1)
    score_and_label(store, generate_dataset(150, seed=111, drift=DRIFT))  # pocas etiquetas nuevas
    d2, source_2 = _labels_dataset(store, tmp_path, "d2")

    challenger = train(d2, model_dir=model_dir, seed=42, data_source=source_2)

    promotion = challenger["promotion"]
    assert promotion["decision"] == "rejected"
    assert promotion["evaluated_groups"] < MIN_COMPARISON_GROUPS
    assert promotion["champion_roc_auc"] is None  # no se le llega a medir
    assert "no hay evidencia suficiente" in promotion["reason"]
    assert "posteriores a las etiquetas del campeon" in promotion["reason"]
    assert LocalModelStore(model_dir).current_version() == champion["model_version"]

    # margen 1 = promover siempre que pase el gate: tampoco aqui se compara
    forced = train(d2, model_dir=model_dir, seed=7, data_source=source_2, promotion_margin=1.0)
    assert forced["promotion"]["decision"] == "promoted"
    assert "promueve sin comparar" in forced["promotion"]["reason"]


def test_datos_sin_predicted_at_evaluan_al_campeon_con_todo_el_holdout(tmp_path, caplog):
    model_dir = str(tmp_path / "models")
    store = SqlitePredictionStore(tmp_path / "drift.sqlite", max_rows=100_000)
    score_and_label(store, generate_dataset(2000, seed=120, drift=DRIFT))
    d1, source_1 = _labels_dataset(store, tmp_path, "d1")
    train(d1, model_dir=model_dir, seed=42, data_source=source_1)

    export = generate_dataset(2000, seed=121, drift=DRIFT)  # fichero del CRM: sin trazabilidad
    challenger = train(export, model_dir=model_dir, seed=42)

    promotion = challenger["promotion"]
    assert promotion["champion_cutoff"] is None
    assert promotion["evaluated_rows"] == challenger["metrics"]["n_test"]
    assert promotion["champion_roc_auc"] is not None
    assert "no traen predicted_at" in caplog.text


def test_mismos_datos_con_otra_semilla_tampoco_se_promueve(tmp_path, worlds):
    _, new, _ = worlds
    champion = train(new, model_dir=str(tmp_path), seed=42)
    again = train(new, model_dir=str(tmp_path), seed=7, promotion_margin=1.0)
    assert again["promotion"]["decision"] == "rejected"
    assert again["promotion"]["identical_training_data"] is True
    assert LocalModelStore(tmp_path).current_version() == champion["model_version"]


def test_holdout_sin_ambas_clases_es_un_error_claro(tmp_path):
    df = generate_dataset(800, seed=130)
    in_test = holdout_mask(split_groups(df))
    df["churn"] = 0
    train_rows = np.flatnonzero(~in_test)
    df.loc[train_rows[:40], "churn"] = 1  # todas las bajas caen fuera del holdout
    with pytest.raises(ValueError, match="no tiene ambas clases"):
        train(df, model_dir=str(tmp_path), seed=42)
    assert not LocalModelStore(tmp_path).exists()


def test_split_queda_en_el_metadata(tmp_path):
    df = generate_dataset(1000, seed=131)
    df["subject_ref"] = [f"c-{i // 2}" for i in range(1000)]  # dos filas por cliente
    metadata = train(df, model_dir=str(tmp_path), seed=42)
    split = metadata["split"]
    assert split["method"] == "group_hash" and split["test_size"] == 0.2
    assert split["subject_rows"] == 1000
    assert split["groups_train"] + split["groups_test"] == 500
    assert metadata["metrics"]["n_test"] == 2 * split["groups_test"]
    info = ModelInfoResponse(**metadata)
    assert info.split is not None and info.split.groups_test == split["groups_test"]


# ------------------------------------------------------------------ cliente puntuado varias veces


def _monthly_scorings(customers: pd.DataFrame, months: int) -> pd.DataFrame:
    """El mismo cliente puntuado cada mes (casi duplicados) con su churn final."""
    snaps = []
    for month in range(months):
        snap = customers.copy()
        snap["tenure_months"] = (snap["tenure_months"] + month).clip(upper=120)
        snap["total_charges"] = (snap["total_charges"] + month * snap["monthly_charges"]).clip(
            upper=60_000
        )
        snap["subject_ref"] = [f"c-{i}" for i in range(len(customers))]
        snaps.append(snap)
    return pd.concat(snaps, ignore_index=True)


@pytest.mark.parametrize("customers", [300, 1000])
def test_cliente_puntuado_varias_veces_no_promueve_un_modelo_peor(tmp_path, customers):
    """Bloqueo: 300 clientes x 4 puntuaciones, todas etiquetadas. Con el split por filas, sus
    casi duplicados caian a ambos lados y el retador (AUC de holdout 0.997, real 0.89) se
    promovia sobre un campeon mejor (real 0.957). Con subject_ref el holdout se separa por
    cliente: con 300 no hay evidencia para sustituir al campeon; con 1000 la comparacion,
    ya honesta, lo descarta."""
    champion = train(generate_dataset(20_000, seed=7, drift=DRIFT), model_dir=str(tmp_path))
    repeated = _monthly_scorings(generate_dataset(customers, seed=5, drift=DRIFT), months=4)

    challenger = train(repeated, model_dir=str(tmp_path), seed=42, min_roc_auc=0.5)

    promotion = challenger["promotion"]
    assert promotion["decision"] == "rejected", promotion["reason"]
    assert challenger["split"]["subject_rows"] == len(repeated)
    assert promotion["evaluated_rows"] == 4 * promotion["evaluated_groups"]
    assert challenger["metrics"]["roc_auc"] < 0.99  # sin la fuga de casi duplicados
    if customers == 300:
        assert promotion["evaluated_groups"] < MIN_COMPARISON_GROUPS
    else:
        assert promotion["challenger_roc_auc"] < promotion["champion_roc_auc"]
    assert LocalModelStore(tmp_path).current_version() == champion["model_version"]
