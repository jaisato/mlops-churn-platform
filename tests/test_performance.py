"""Rendimiento real sobre predicciones etiquetadas (`churn.monitoring.performance`)."""

import pandas as pd
import pytest

from churn.monitoring.performance import (
    DECISION_THRESHOLD,
    classification_metrics,
    performance_report,
)


def test_classification_metrics_con_ambas_clases():
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    # pred = [0, 1, 0, 1]: un acierto por clase; AUC = 3 de 4 pares bien ordenados
    assert metrics == {
        "roc_auc": 0.75,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "brier": 0.185,
    }


def test_classification_metrics_con_una_sola_clase_no_tiene_auc():
    metrics = classification_metrics([1, 1, 1], [0.2, 0.3, 0.9])
    assert metrics["roc_auc"] is None
    assert metrics["accuracy"] == round(1 / 3, 4)
    assert (metrics["precision"], metrics["recall"], metrics["f1"]) == (1.0, round(1 / 3, 4), 0.5)

    sin_positivos = classification_metrics([0, 0], [0.1, 0.2])
    assert sin_positivos["roc_auc"] is None
    assert (sin_positivos["precision"], sin_positivos["recall"], sin_positivos["f1"]) == (0, 0, 0)


def test_classification_metrics_respeta_el_umbral():
    assert DECISION_THRESHOLD == 0.5
    assert classification_metrics([0, 1], [0.3, 0.4])["accuracy"] == 0.5
    assert classification_metrics([0, 1], [0.3, 0.4], threshold=0.35)["accuracy"] == 1.0


def test_classification_metrics_acepta_series_y_redondea_a_4_decimales():
    y = pd.Series([0, 1, 1, 0, 1, 0, 1])
    p = pd.Series([0.12345, 0.98765, 0.5, 0.49999, 0.33333, 0.1, 0.9])
    metrics = classification_metrics(y, p)
    assert all(v == round(v, 4) for v in metrics.values() if v is not None)
    assert 0 < metrics["brier"] < 1


def test_performance_report_global_y_por_version():
    labelled = pd.DataFrame(
        {
            "churn": [0, 1, 0, 1, 1],
            "churn_probability": [0.35, 0.9, 0.2, 0.8, 0.3],
            "model_version": ["v2", "v1", "v1", "v2", "v2"],
            "tenure_months": [1, 2, 3, 4, 5],  # las features sobran y no molestan
        }
    )
    report = performance_report(labelled)
    assert report["threshold"] == 0.5
    assert report["overall"]["n"] == 5
    assert report["overall"]["churn_rate"] == 0.6
    assert report["overall"]["roc_auc"] == round(5 / 6, 4)  # un par mal ordenado (0.3 < 0.35)
    assert report["overall"]["recall"] == round(2 / 3, 4)
    assert list(report["by_version"]) == ["v1", "v2"]  # ordenadas por version
    assert report["by_version"]["v1"] == {
        "n": 2,
        "churn_rate": 0.5,
        "roc_auc": 1.0,
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "brier": 0.025,
    }
    assert report["by_version"]["v2"]["n"] == 3
    assert report["by_version"]["v2"]["roc_auc"] == 0.5
    assert report["by_version"]["v2"]["recall"] == 0.5


def test_performance_report_con_umbral_propio():
    labelled = pd.DataFrame(
        {"churn": [0, 1], "churn_probability": [0.2, 0.4], "model_version": ["v1", "v1"]}
    )
    assert performance_report(labelled, threshold=0.3)["overall"]["accuracy"] == 1.0
    assert performance_report(labelled)["overall"]["accuracy"] == 0.5


def test_performance_report_sin_filas_falla():
    with pytest.raises(ValueError, match="No hay predicciones etiquetadas"):
        performance_report(pd.DataFrame())
