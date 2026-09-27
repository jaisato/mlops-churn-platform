"""Generador de datos sinteticos de churn (sector telco/suscripciones).

Genera un dataset realista y REPRODUCIBLE (semilla fija) con relacion no trivial
entre variables y probabilidad de baja, para poder entrenar y testear el pipeline
sin depender de datos privados.

`DriftSpec` desplaza ese "mundo" de forma determinista: cambia la distribucion de
las features (covariate shift) y la relacion entre features y etiqueta (concept
drift). Sirve para producir, en tests y demos, un dataset etiquetado en el que el
modelo anterior funciona peor y el reentrenamiento tiene algo que corregir. Con la
especificacion por defecto (todo a cero) el dataset es identico al de siempre.
"""

from __future__ import annotations

from dataclasses import dataclass

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

BASE_TICKETS_RATE = 1.2
BASE_MONTHLY_CONTRACT_SHARE = 0.55
BASE_FIBER_WEIGHT = -0.25
BASE_TICKETS_WEIGHT = 0.55
MAX_DRIFT_SHIFT = 2.0


@dataclass(frozen=True)
class DriftSpec:
    """Desplazamiento determinista del mundo sintetico. Todo a cero = sin drift.

    Covariate shift (cambia lo que ve el modelo):
      tenure_shift            meses que se restan a la antiguedad (ola de clientes nuevos)
      monthly_charges_shift   euros que se suman a la cuota mensual (subida de tarifas)
      tickets_rate_shift      incremento de la media de tickets de soporte
      monthly_contract_shift  puntos de proporcion que gana el contrato mensual
    Concept drift (cambia la relacion con la etiqueta, la parte que un modelo viejo
    no puede absorber):
      fiber_weight_shift      cambio del peso de la fibra en el logit (un competidor
                              de fibra hace que esos clientes se vayan mas)
      tickets_weight_shift    cambio del peso de los tickets (soporte mejor: un ticket
                              ya no anticipa la baja)
      logit_shift             desplazamiento del intercepto
    """

    tenure_shift: float = 0.0
    monthly_charges_shift: float = 0.0
    tickets_rate_shift: float = 0.0
    monthly_contract_shift: float = 0.0
    fiber_weight_shift: float = 0.0
    tickets_weight_shift: float = 0.0
    logit_shift: float = 0.0

    @classmethod
    def from_shift(cls, shift: float) -> DriftSpec:
        """Escenario de drift proporcional a `shift` (0 = ninguno, 1 = fuerte, maximo 2).

        Con `shift=1` varias features superan el umbral PSI de alerta (0.2) y el modelo
        entrenado en el mundo original pierde AUC frente a uno entrenado en el nuevo.
        La tasa de churn se mantiene dentro de los limites que exige la validacion.
        """
        if not 0.0 <= shift <= MAX_DRIFT_SHIFT:
            raise ValueError(f"drift shift debe estar entre 0 y {MAX_DRIFT_SHIFT}: {shift}")
        return cls(
            tenure_shift=18.0 * shift,
            monthly_charges_shift=20.0 * shift,
            tickets_rate_shift=1.0 * shift,
            monthly_contract_shift=0.2 * shift,
            fiber_weight_shift=1.4 * shift,
            tickets_weight_shift=-0.45 * shift,
            logit_shift=-1.3 * shift,
        )

    @property
    def is_null(self) -> bool:
        return self == DriftSpec()


def generate_dataset(
    n_rows: int = 20_000, seed: int = 42, drift: DriftSpec | None = None
) -> pd.DataFrame:
    """Devuelve un DataFrame con las features y la etiqueta `churn` (0/1).

    Con `drift=None` (o `DriftSpec()`) el resultado es bit a bit el de siempre para esa
    semilla; con un `DriftSpec` se aplica el desplazamiento descrito en la clase.
    """
    spec = drift or DriftSpec()
    rng = np.random.default_rng(seed)

    # El orden de las extracciones es el de siempre: con DriftSpec() el dataset es
    # identico al historico para cada semilla; los desplazamientos se aplican despues.
    tenure = rng.integers(1, 72, n_rows)
    monthly = np.round(rng.uniform(15, 110, n_rows), 2)
    total_factor = rng.uniform(0.9, 1.1, n_rows)
    products = rng.integers(1, 5, n_rows)
    tickets_rate = max(0.0, BASE_TICKETS_RATE + spec.tickets_rate_shift)
    tickets = rng.poisson(tickets_rate, n_rows).clip(0, 15)
    fiber = rng.integers(0, 2, n_rows)
    monthly_share = float(np.clip(BASE_MONTHLY_CONTRACT_SHARE + spec.monthly_contract_shift, 0, 1))
    other_share = 1.0 - monthly_share
    contract = rng.choice(
        CONTRACT_TYPES, n_rows, p=[monthly_share, other_share * 2 / 3, other_share / 3]
    )
    payment = rng.choice(PAYMENT_METHODS, n_rows, p=[0.5, 0.3, 0.2])

    if not spec.is_null:
        tenure = np.clip(tenure - int(round(spec.tenure_shift)), 1, 120)
        monthly = np.round(np.clip(monthly + spec.monthly_charges_shift, 0, 500), 2)
    total = np.round(monthly * tenure * total_factor, 2)

    # Modelo latente: menos antiguedad, mas tickets, contrato mensual y cuota alta -> mas churn
    logit = (
        -1.1
        + spec.logit_shift
        - 0.045 * tenure
        + (BASE_TICKETS_WEIGHT + spec.tickets_weight_shift) * tickets
        + 0.022 * (monthly - 60)
        - 0.35 * products
        + np.where(contract == "mensual", 1.1, np.where(contract == "anual", 0.1, -0.6))
        + np.where(payment == "transferencia", 0.4, 0.0)
        + (BASE_FIBER_WEIGHT + spec.fiber_weight_shift) * fiber
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
