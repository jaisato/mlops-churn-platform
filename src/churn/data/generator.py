"""Generador de datos sinteticos de churn (sector telco/suscripciones).

Genera un dataset realista y REPRODUCIBLE (semilla fija) con relacion no trivial
entre variables y probabilidad de baja, para poder entrenar y testear el pipeline
sin depender de datos privados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACT_TYPES = ["mensual", "anual", "bianual"]
PAYMENT_METHODS = ["domiciliacion", "tarjeta", "transferencia"]

CATEGORICAL_FEATURES = ["contract_type", "payment_method"]
NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "support_tickets_90d",
    "is_fiber",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"


def generate_dataset(n_rows: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Devuelve un DataFrame con las features y la etiqueta `churn` (0/1)."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, n_rows)
    monthly = np.round(rng.uniform(15, 110, n_rows), 2)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.1, n_rows), 2)
    products = rng.integers(1, 5, n_rows)
    tickets = rng.poisson(1.2, n_rows).clip(0, 15)
    fiber = rng.integers(0, 2, n_rows)
    contract = rng.choice(CONTRACT_TYPES, n_rows, p=[0.55, 0.3, 0.15])
    payment = rng.choice(PAYMENT_METHODS, n_rows, p=[0.5, 0.3, 0.2])

    # Modelo latente: menos antiguedad, mas tickets, contrato mensual y cuota alta -> mas churn
    logit = (
        -1.1
        - 0.045 * tenure
        + 0.55 * tickets
        + 0.022 * (monthly - 60)
        - 0.35 * products
        + np.where(contract == "mensual", 1.1, np.where(contract == "anual", 0.1, -0.6))
        + np.where(payment == "transferencia", 0.4, 0.0)
        - 0.25 * fiber
        + rng.normal(0, 0.6, n_rows)
    )
    churn = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "num_products": products,
            "support_tickets_90d": tickets,
            "is_fiber": fiber,
            "contract_type": contract,
            "payment_method": payment,
            TARGET: churn,
        }
    )
