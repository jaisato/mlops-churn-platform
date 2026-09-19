"""Deteccion de data drift entre la distribucion de entrenamiento y produccion.

Metricas implementadas (sin dependencias pesadas, faciles de auditar):
  - PSI (Population Stability Index) para variables numericas y categoricas.
  - Test de Kolmogorov-Smirnov (scipy) para numericas.
  - PSI sobre las probabilidades predichas ("prediction drift"): detecta que el
    modelo esta puntuando distinto aunque cada feature por separado parezca estable.

Interpretacion estandar del PSI:  < 0.1 sin cambio | 0.1-0.2 cambio moderado |
> 0.2 cambio significativo (alerta).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-6


def _numeric_values(values: pd.Series | np.ndarray | list) -> np.ndarray:
    """Array float sin NaN (los nulos no cuentan en la distribucion)."""
    arr = pd.Series(values).dropna().to_numpy(dtype=float)
    return arr


def _psi_from_distributions(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    ref_pct = ref_pct + EPS
    cur_pct = cur_pct + EPS
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def bin_edges(reference: np.ndarray, bins: int = 10) -> np.ndarray:
    """Bordes de bins (con -inf/+inf en los extremos) a partir de la referencia.

    - Muchos valores distintos: cuantiles de la referencia (bins equiprobables).
    - Pocos valores distintos (<= bins; enteros, binarias): un bin por valor, con los
      bordes en los puntos medios. Con cuantiles, una binaria desbalanceada o un
      cambio de 3 a 4 productos quedaban enmascarados en un unico bin (PSI = 0).
    - Referencia constante: un bin estrecho alrededor de la constante, de modo que
      cualquier otro valor en produccion cuenta como cambio.
    """
    uniques = np.unique(reference)
    if len(uniques) == 0:
        return np.array([-np.inf, np.inf])
    if len(uniques) == 1:
        constant = uniques[0]
        inner = np.array([constant, np.nextafter(constant, np.inf)])
    elif len(uniques) <= bins:
        inner = (uniques[:-1] + uniques[1:]) / 2
    else:
        inner = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))[1:-1]
        if len(inner) == 0:  # cuantiles degenerados (un valor domina): un bin por valor
            inner = (uniques[:-1] + uniques[1:]) / 2
    return np.concatenate(([-np.inf], inner, [np.inf]))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = _numeric_values(reference)
    cur = _numeric_values(current)
    edges = bin_edges(ref, bins)
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    return _psi_from_distributions(ref_pct, cur_pct)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    # astype(str) evita el TypeError al ordenar categorias de tipos mezclados (p. ej. 1 y "1").
    ref = pd.Series(reference).dropna().astype(str)
    cur = pd.Series(current).dropna().astype(str)
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    ref_pct = ref.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    cur_pct = cur.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    return _psi_from_distributions(ref_pct, cur_pct)


def prediction_drift(
    reference_scores: pd.Series | np.ndarray,
    current_scores: pd.Series | np.ndarray,
    psi_threshold: float = 0.2,
) -> dict[str, Any]:
    """PSI entre las probabilidades predichas en la referencia y en produccion."""
    ref = _numeric_values(reference_scores)
    cur = _numeric_values(current_scores)
    psi = psi_numeric(ref, cur)
    return {
        "psi": round(psi, 4),
        "n_reference": int(len(ref)),
        "n_current": int(len(cur)),
        "mean_reference": round(float(ref.mean()), 4) if len(ref) else None,
        "mean_current": round(float(cur.mean()), 4) if len(cur) else None,
        "drift": bool(psi > psi_threshold),
    }


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    psi_threshold: float = 0.2,
    reference_scores: pd.Series | np.ndarray | None = None,
    current_scores: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for col in numeric_features:
        psi = psi_numeric(reference[col], current[col])
        ks_stat, ks_pvalue = stats.ks_2samp(reference[col].dropna(), current[col].dropna())
        features[col] = {
            "type": "numeric",
            "psi": round(psi, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "drift": bool(psi > psi_threshold),
        }
    for col in categorical_features:
        psi = psi_categorical(reference[col], current[col])
        features[col] = {
            "type": "categorical",
            "psi": round(psi, 4),
            "drift": bool(psi > psi_threshold),
        }

    predictions = None
    if reference_scores is not None and current_scores is not None:
        predictions = prediction_drift(reference_scores, current_scores, psi_threshold)

    drifted = sorted(name for name, f in features.items() if f["drift"])
    return {
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "psi_threshold": psi_threshold,
        "features": features,
        "drifted_features": drifted,
        "predictions": predictions,
        "drift_detected": bool(drifted) or bool(predictions and predictions["drift"]),
    }
