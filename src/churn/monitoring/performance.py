"""Rendimiento real del modelo: metricas de clasificacion sobre predicciones etiquetadas.

El drift dice que los datos han cambiado; esto dice si el modelo sigue acertando. Con las
etiquetas que llegan por POST /labels se calculan, sobre la ventana reciente, las mismas
metricas que el entrenamiento mide en su holdout (AUC, accuracy, precision, recall, F1,
Brier), en conjunto y por version del modelo (las etiquetas llegan semanas despues de la
prediccion, asi que una ventana suele mezclar la version en servicio con la anterior).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.monitoring.store import LABEL_COLUMN, SCORE_COLUMN, VERSION_COLUMN

#: Umbral con el que se binariza la probabilidad para accuracy/precision/recall/F1
#: (el mismo que usa el entrenamiento para sus metricas de holdout).
DECISION_THRESHOLD = 0.5


def classification_metrics(
    y_true: Any, proba: Any, threshold: float = DECISION_THRESHOLD
) -> dict[str, float | None]:
    """AUC, accuracy, precision, recall, F1 y Brier de unas probabilidades frente a 0/1.

    `roc_auc` es None si solo hay una clase (no esta definido); precision/recall/F1 valen
    0 cuando no hay positivos predichos o reales (en lugar de avisar o fallar).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) == 2 else None,
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
    }


def _block(rows: pd.DataFrame, threshold: float) -> dict[str, Any]:
    return {
        "n": int(len(rows)),
        "churn_rate": round(float(rows[LABEL_COLUMN].mean()), 4),
        **classification_metrics(rows[LABEL_COLUMN], rows[SCORE_COLUMN], threshold),
    }


def performance_report(
    labelled: pd.DataFrame, threshold: float = DECISION_THRESHOLD
) -> dict[str, Any]:
    """Metricas en conjunto (`overall`) y por version (`by_version`) de las filas etiquetadas.

    `labelled` trae al menos las columnas `churn` (0/1), `churn_probability` y
    `model_version` (lo que devuelve `PredictionStore.labelled()`). Debe tener filas.
    """
    if labelled.empty:
        raise ValueError("No hay predicciones etiquetadas sobre las que medir el rendimiento")
    return {
        "threshold": threshold,
        "overall": _block(labelled, threshold),
        "by_version": {
            str(version): _block(rows, threshold)
            for version, rows in labelled.groupby(VERSION_COLUMN, sort=True)
        },
    }
