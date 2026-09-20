import numpy as np
import pandas as pd
import pytest

from churn.data.generator import generate_dataset
from churn.monitoring.drift import (
    bin_edges,
    drift_report,
    prediction_drift,
    psi_categorical,
    psi_numeric,
)

NUM = ["tenure_months", "monthly_charges"]
CAT = ["contract_type"]


def test_sin_drift_misma_distribucion():
    ref = generate_dataset(4000, seed=1)
    cur = generate_dataset(4000, seed=99)  # misma distribucion, otra muestra
    report = drift_report(ref, cur, NUM, CAT)
    assert report["drift_detected"] is False
    assert all(f["psi"] < 0.1 for f in report["features"].values())


def test_drift_detectado_con_shift():
    ref = generate_dataset(4000, seed=1)
    cur = generate_dataset(4000, seed=2)
    cur["monthly_charges"] = cur["monthly_charges"] * 2.0 + 40  # shift fuerte
    report = drift_report(ref, cur, NUM, CAT)
    assert report["drift_detected"] is True
    assert "monthly_charges" in report["drifted_features"]
    assert report["features"]["monthly_charges"]["psi"] > 0.2


def test_psi_numeric_cero_para_identicas():
    ref = generate_dataset(3000, seed=5)
    assert psi_numeric(ref["tenure_months"], ref["tenure_months"]) < 0.01


def test_psi_categorical_detecta_cambio_de_mix():
    ref = generate_dataset(3000, seed=5)
    cur = ref.copy()
    cur["contract_type"] = "mensual"  # todo el trafico pasa a mensual
    assert psi_categorical(ref["contract_type"], cur["contract_type"]) > 0.2


# --------------------------------------------------------------- binning robusto (bug fix)


def test_psi_numeric_binaria_desbalanceada_detecta_shift():
    """Con bins por cuantiles, una binaria 70/30 caia entera en un bin y el PSI era 0."""
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.choice([0, 1], 5000, p=[0.7, 0.3]))
    cur = pd.Series(rng.choice([0, 1], 5000, p=[0.3, 0.7]))
    assert psi_numeric(ref, cur) > 0.2


def test_psi_numeric_binaria_estable_es_baja():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.choice([0, 1], 5000, p=[0.7, 0.3]))
    cur = pd.Series(rng.choice([0, 1], 5000, p=[0.7, 0.3]))
    assert psi_numeric(ref, cur) < 0.05


def test_psi_numeric_referencia_constante():
    ref = pd.Series(np.zeros(1000))
    assert psi_numeric(ref, pd.Series(np.zeros(500))) < 1e-6
    assert psi_numeric(ref, pd.Series(np.ones(500))) > 0.2
    assert psi_numeric(ref, pd.Series(-np.ones(500))) > 0.2


def test_psi_numeric_baja_cardinalidad_distingue_valores():
    """num_products (1..4): con cuantiles, 3 y 4 compartian bin y eran indistinguibles."""
    rng = np.random.default_rng(2)
    ref = pd.Series(rng.integers(1, 5, 5000))
    todo_3 = psi_numeric(ref, pd.Series(np.full(5000, 3)))
    todo_4 = psi_numeric(ref, pd.Series(np.full(5000, 4)))
    assert todo_3 > 0.2 and todo_4 > 0.2
    assert todo_3 != todo_4


def test_psi_numeric_ignora_nulos():
    ref = generate_dataset(2000, seed=3)["monthly_charges"]
    cur = ref.copy()
    cur.iloc[:50] = np.nan
    assert psi_numeric(ref, cur) < 0.05


def test_bin_edges_cuantiles_para_continuas():
    ref = np.random.default_rng(4).uniform(0, 100, 5000)
    edges = bin_edges(ref, bins=10)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert len(edges) == 11
    assert np.all(np.diff(edges) > 0)


def test_bin_edges_un_bin_por_valor_en_baja_cardinalidad():
    edges = bin_edges(np.array([0, 1, 0, 1, 1]), bins=10)
    assert list(edges) == [-np.inf, 0.5, np.inf]


def test_bin_edges_referencia_vacia():
    assert list(bin_edges(np.array([]))) == [-np.inf, np.inf]


def test_psi_numeric_cuantiles_degenerados_por_valor_dominante():
    """95% ceros + cola larga: los cuantiles colapsan en [0, max]; se cae a un bin por valor."""
    ref = pd.Series(np.concatenate([np.zeros(950), np.linspace(1, 50, 50)]))
    assert len(bin_edges(ref.to_numpy())) > 3
    assert psi_numeric(ref, ref) < 1e-6
    assert psi_numeric(ref, pd.Series(np.full(500, 25.0))) > 0.2


# --------------------------------------------------------------- categoricas


def test_psi_categorical_tipos_mezclados_no_revienta():
    psi = psi_categorical(pd.Series(["a", "b", "a"]), pd.Series(["a", 1, None]))
    assert psi > 0


def test_psi_categorical_identicas_es_cero():
    ref = generate_dataset(2000, seed=6)["payment_method"]
    assert psi_categorical(ref, ref) < 1e-6


def test_psi_categorical_categoria_nueva_en_produccion():
    ref = pd.Series(["a"] * 900 + ["b"] * 100)
    cur = pd.Series(["a"] * 800 + ["b"] * 100 + ["c"] * 100)
    assert psi_categorical(ref, cur) > 0.1


# --------------------------------------------------------------- prediction drift


def test_prediction_drift_estable_y_con_shift():
    rng = np.random.default_rng(7)
    ref = rng.beta(2, 8, 3000)
    estable = prediction_drift(ref, rng.beta(2, 8, 3000))
    shift = prediction_drift(ref, rng.beta(8, 2, 3000))
    assert estable["drift"] is False and estable["psi"] < 0.1
    assert shift["drift"] is True and shift["psi"] > 0.2
    assert shift["mean_current"] > shift["mean_reference"]
    assert shift["n_reference"] == 3000 and shift["n_current"] == 3000


def test_prediction_drift_series_vacias_no_revienta():
    result = prediction_drift(np.array([]), np.array([]))
    assert result["mean_reference"] is None and result["mean_current"] is None


def test_drift_report_marca_drift_solo_por_predicciones():
    ref = generate_dataset(3000, seed=8)
    cur = generate_dataset(3000, seed=9)
    rng = np.random.default_rng(8)
    report = drift_report(
        ref,
        cur,
        NUM,
        CAT,
        reference_scores=rng.beta(2, 8, 3000),
        current_scores=rng.beta(8, 2, 3000),
    )
    assert report["drifted_features"] == []
    assert report["predictions"]["drift"] is True
    assert report["drift_detected"] is True


def test_drift_report_estructura_y_umbral():
    ref = generate_dataset(2000, seed=10)
    cur = generate_dataset(2000, seed=11)
    cur["monthly_charges"] = cur["monthly_charges"] * 2.0 + 40
    report = drift_report(ref, cur, NUM, CAT, psi_threshold=50.0)
    assert set(report) == {
        "n_reference",
        "n_current",
        "psi_threshold",
        "features",
        "drifted_features",
        "predictions",
        "drift_detected",
    }
    assert report["n_reference"] == 2000 and report["n_current"] == 2000
    assert report["psi_threshold"] == 50.0
    assert report["predictions"] is None
    assert report["drift_detected"] is False  # el shift existe pero el umbral es enorme
    numeric = report["features"]["monthly_charges"]
    assert {"type", "psi", "ks_statistic", "ks_pvalue", "drift"} <= set(numeric)
    assert set(report["features"]["contract_type"]) == {"type", "psi", "drift"}


@pytest.mark.parametrize("col", ["is_fiber", "num_products", "support_tickets_90d"])
def test_drift_report_features_enteras_estables_sin_falsos_positivos(col):
    ref = generate_dataset(3000, seed=12)
    cur = generate_dataset(3000, seed=13)
    report = drift_report(ref, cur, [col], [])
    assert report["features"][col]["drift"] is False
