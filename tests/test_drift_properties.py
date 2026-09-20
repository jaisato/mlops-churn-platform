"""Propiedades matematicas del PSI que deben cumplirse para cualquier entrada.

Complementan a los tests de ejemplo de test_drift.py: aqui Hypothesis busca
contraejemplos (listas vacias, constantes, valores extremos, un solo elemento...).
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from churn.monitoring.drift import bin_edges, prediction_drift, psi_categorical, psi_numeric

finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, allow_subnormal=False
)
samples = st.lists(finite_floats, min_size=1, max_size=200)
categories = st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=1, max_size=200)
scores = st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=200)


@given(samples, samples)
@settings(max_examples=150, deadline=None)
def test_psi_numeric_es_finito_y_no_negativo(ref, cur):
    psi = psi_numeric(pd.Series(ref), pd.Series(cur))
    assert np.isfinite(psi)
    assert psi >= -1e-12


@given(samples)
@settings(max_examples=150, deadline=None)
def test_psi_numeric_de_una_muestra_consigo_misma_es_cero(values):
    serie = pd.Series(values)
    assert psi_numeric(serie, serie) < 1e-9


@given(samples, st.integers(min_value=2, max_value=20))
@settings(max_examples=150, deadline=None)
def test_bin_edges_cubren_toda_la_recta_y_no_decrecen(values, bins):
    edges = bin_edges(np.array(values, dtype=float), bins=bins)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert np.all(np.diff(edges) >= 0)
    assert len(edges) <= max(bins, len(set(values))) + 2


@given(categories, categories)
@settings(max_examples=150, deadline=None)
def test_psi_categorical_es_simetrico_y_no_negativo(ref, cur):
    a, b = pd.Series(ref), pd.Series(cur)
    assert psi_categorical(a, b) >= -1e-12
    assert abs(psi_categorical(a, b) - psi_categorical(b, a)) < 1e-9


@given(categories)
@settings(max_examples=100, deadline=None)
def test_psi_categorical_identico_es_cero(values):
    serie = pd.Series(values)
    assert psi_categorical(serie, serie) < 1e-9


@given(scores, scores, st.floats(min_value=0.01, max_value=5.0))
@settings(max_examples=100, deadline=None)
def test_prediction_drift_es_coherente_con_su_umbral(ref, cur, threshold):
    result = prediction_drift(np.array(ref), np.array(cur), psi_threshold=threshold)
    assert result["n_reference"] == len(ref) and result["n_current"] == len(cur)
    assert 0.0 <= result["mean_reference"] <= 1.0 and 0.0 <= result["mean_current"] <= 1.0
    assert result["drift"] == (result["psi"] > threshold)
