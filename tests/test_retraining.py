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
import pytest

from churn.data.generator import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    DriftSpec,
    generate_dataset,
)
from churn.monitoring.drift import drift_report
from churn.registry import LocalModelStore
from churn.training import train as train_module
from churn.training.train import EXIT_NOT_PROMOTED, EXIT_OK, main, train
from tests.conftest import ADMIN_HEADERS, weak_pipeline

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
    assert "el retador es identico" in caplog.text
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
