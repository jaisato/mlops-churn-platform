"""Validacion de calidad de datos previa al entrenamiento (fail-fast).

Un pipeline serio nunca entrena con datos rotos: aqui se comprueban esquema,
nulos, rangos y balance de clases antes de tocar el modelo.

`RANGES` y `ALLOWED_CATEGORIES` son la fuente unica de verdad del dominio de cada
feature: el contrato Pydantic de la API (`churn.serving.schemas`) se construye a
partir de ellos, de modo que training y serving no puedan divergir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from churn.data.generator import (
    CATEGORICAL_FEATURES,
    CONTRACT_TYPES,
    NUMERIC_FEATURES,
    PAYMENT_METHODS,
    TARGET,
)

RANGES: dict[str, tuple[float, float]] = {
    "tenure_months": (0, 120),
    "monthly_charges": (0, 500),
    "total_charges": (0, 60_000),
    "num_products": (1, 10),
    "support_tickets_90d": (0, 50),
    "is_fiber": (0, 1),
}

ALLOWED_CATEGORIES: dict[str, frozenset[str]] = {
    "contract_type": frozenset(CONTRACT_TYPES),
    "payment_method": frozenset(PAYMENT_METHODS),
}

MIN_ROWS = 500
CHURN_RATE_BOUNDS = (0.02, 0.8)

assert set(RANGES) == set(NUMERIC_FEATURES), "RANGES debe cubrir exactamente NUMERIC_FEATURES"
assert set(ALLOWED_CATEGORIES) == set(CATEGORICAL_FEATURES), "Categorias fuera de sincronia"


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_training_data(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    if df.columns.duplicated().any():
        report.errors.append("Columnas duplicadas")
        return report
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        report.errors.append(f"Faltan columnas: {sorted(missing)}")
        return report

    if len(df) < MIN_ROWS:
        report.errors.append(f"Dataset demasiado pequeno: {len(df)} filas (minimo {MIN_ROWS})")

    null_cols = [c for c in expected if df[c].isna().any()]
    if null_cols:
        report.errors.append(f"Columnas con nulos: {sorted(null_cols)}")

    # Rangos y categorias se evaluan sin nulos: ya se han reportado arriba y asi no se
    # duplican como "fuera de rango" o "categoria desconocida: nan".
    for col, (lo, hi) in RANGES.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            report.errors.append(f"Columna no numerica: {col}")
        elif not df[col].dropna().between(lo, hi).all():
            report.errors.append(f"Valores fuera de rango en {col} (esperado [{lo}, {hi}])")

    for col, allowed in ALLOWED_CATEGORIES.items():
        extra = set(df[col].dropna().unique()) - allowed
        if extra:
            report.errors.append(f"Categorias desconocidas en {col}: {sorted(extra, key=str)}")

    target = df[TARGET].dropna()
    if not pd.api.types.is_numeric_dtype(df[TARGET]) or not target.isin([0, 1]).all():
        report.errors.append("El objetivo churn debe contener solo 0 o 1")
    elif len(target):
        rate = target.mean()
        lo, hi = CHURN_RATE_BOUNDS
        if not lo <= rate <= hi:
            report.errors.append(
                f"Tasa de churn sospechosa: {rate:.3f} (esperado {lo:.0%}-{hi:.0%})"
            )

    return report
