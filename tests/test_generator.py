import pandas as pd
import pytest

from churn.data.generator import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    generate_dataset,
)
from churn.data.validation import ALLOWED_CATEGORIES, RANGES


def test_esquema_y_tamano():
    df = generate_dataset(n_rows=1000, seed=1)
    assert len(df) == 1000
    assert set(FEATURE_COLUMNS + [TARGET]).issubset(df.columns)
    assert df.isna().sum().sum() == 0


def test_reproducible_con_semilla():
    a = generate_dataset(500, seed=3)
    b = generate_dataset(500, seed=3)
    assert a.equals(b)


def test_semillas_distintas_datasets_distintos():
    a = generate_dataset(500, seed=3)
    b = generate_dataset(500, seed=4)
    assert not a.equals(b)


def test_tasa_churn_razonable():
    df = generate_dataset(5000, seed=2)
    assert 0.05 < df[TARGET].mean() < 0.6


def test_senal_predictiva_existe():
    """Los clientes con contrato mensual deben tener mas churn que los bianuales."""
    df = generate_dataset(8000, seed=5)
    mensual = df[df.contract_type == "mensual"][TARGET].mean()
    bianual = df[df.contract_type == "bianual"][TARGET].mean()
    assert mensual > bianual


def test_mas_tickets_mas_churn():
    df = generate_dataset(8000, seed=6)
    muchos = df[df.support_tickets_90d >= 3][TARGET].mean()
    pocos = df[df.support_tickets_90d == 0][TARGET].mean()
    assert muchos > pocos


@pytest.mark.parametrize("n_rows", [1, 10, 777])
def test_tamano_arbitrario(n_rows):
    df = generate_dataset(n_rows=n_rows, seed=1)
    assert len(df) == n_rows
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET]


def test_dtypes():
    df = generate_dataset(300, seed=1)
    for col in NUMERIC_FEATURES + [TARGET]:
        assert pd.api.types.is_numeric_dtype(df[col]), col
    for col in CATEGORICAL_FEATURES:
        assert df[col].map(type).eq(str).all(), col
    assert set(df[TARGET].unique()) <= {0, 1}


def test_valores_dentro_del_contrato_de_validacion():
    """Contrato generador <-> validador: todo lo generado debe caer en RANGES/categorias."""
    df = generate_dataset(20_000, seed=42)
    for col, (lo, hi) in RANGES.items():
        assert df[col].between(lo, hi).all(), col
    for col, allowed in ALLOWED_CATEGORIES.items():
        assert set(df[col].unique()) <= allowed, col


def test_total_charges_coherente_con_cuota_y_antiguedad():
    df = generate_dataset(2000, seed=9)
    ratio = df.total_charges / (df.monthly_charges * df.tenure_months)
    assert ratio.between(0.89, 1.11).all()
