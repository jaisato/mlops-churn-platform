"""Bucle de etiquetas de extremo a extremo, sin Docker ni dobles.

El mismo camino que recorre `retrain-if-drift.sh` en el VPS: la API puntua y devuelve
`prediction_id`, el negocio devuelve el churn real por POST /labels, la CLI de etiquetas
construye el dataset desde el mismo SQLite que usa la API, el trainer entrena con `--data`
y, si gana, la API lo recarga. El metadata del modelo nuevo deja escrito de que etiquetas
sale.
"""

import json

from churn.data.generator import DriftSpec, generate_dataset
from churn.data.labels import main as build_labels_dataset
from churn.data.sources import read_sidecar
from churn.training.train import main as train_main
from tests.conftest import ADMIN_HEADERS
from tests.test_api import _score_and_label

DRIFT = DriftSpec.from_shift(1.0)


def test_bucle_de_etiquetas_de_extremo_a_extremo(client_factory, capsys):
    c, model_dir = client_factory(seed=42, drift_min_rows=50)  # SQLite real dentro de model_dir
    v1 = c.get("/health").json()["model_version"]
    holdout_auc = c.get("/model/info").json()["metrics"]["roc_auc"]
    world = generate_dataset(1500, seed=43, drift=DRIFT)  # el mundo ha cambiado

    # 1. Se puntua el trafico nuevo y, semanas despues, llega su churn real (lotes <= 1000)
    for chunk in (world.iloc[:1000], world.iloc[1000:]):
        _score_and_label(c, chunk)
    assert c.get("/monitoring/drift").json()["drift_detected"] is True
    before = c.get("/monitoring/performance").json()
    assert before["labelled_total"] == 1500
    assert before["by_version"] == {v1: before["overall"]}
    assert before["holdout_roc_auc"] == holdout_auc
    assert before["overall"]["roc_auc"] < holdout_auc - 0.03  # acierta menos que en su holdout

    # 2. El dataset de reentreno sale de esas etiquetas (lo que hace retrain-if-drift.sh)
    assert build_labels_dataset(["--model-dir", model_dir]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])  # como `tail -n 1`
    assert summary["rows"] == 1500 and summary["model_versions"] == [v1]
    assert read_sidecar(summary["path"])["fingerprint"] == summary["fingerprint"]

    # 3. Reentreno con --data: el retador gana al campeon sobre el holdout nuevo y se promueve
    assert train_main(["--data", summary["path"], "--model-dir", model_dir, "--seed", "42"]) == 0
    v2 = c.post("/model/reload", headers=ADMIN_HEADERS).json()["model_version"]
    assert v2 != v1
    info = c.get("/model/info").json()
    assert info["promotion"]["decision"] == "promoted"
    assert info["promotion"]["champion_version"] == v1
    assert info["data_source"]["kind"] == "labels"
    assert info["data_source"]["rows"] == 1500
    assert info["data_source"]["fingerprint"] == summary["fingerprint"]
    assert info["data_source"]["labels"]["model_versions"] == [v1]
    assert info["data_source"]["labels"]["built_at"] == summary["built_at"]
    assert info["data_source"]["labels"]["predicted_to"] == summary["predicted_to"]

    # 4. El modelo nuevo acierta mas en el mundo nuevo; el drift contra su referencia desaparece
    _score_and_label(c, generate_dataset(400, seed=44, drift=DRIFT))
    after = c.get("/monitoring/performance").json()
    assert after["serving_version"] == v2
    assert after["labelled_total"] == 1900
    assert set(after["by_version"]) == {v1, v2}
    assert after["by_version"][v2]["n"] == 400
    assert after["by_version"][v2]["roc_auc"] > before["overall"]["roc_auc"] + 0.03
    drift = c.get("/monitoring/drift").json()
    assert drift["drifted_features"] == []
