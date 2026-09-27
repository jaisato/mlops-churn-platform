import pandas as pd
import pytest

from churn.data.generator import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    DriftSpec,
    generate_dataset,
)
from churn.data.validation import (
    ALLOWED_CATEGORIES,
    CHURN_RATE_BOUNDS,
    RANGES,
    validate_training_data,
)
from churn.monitoring.drift import drift_report


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


# ------------------------------------------------------------------ drift determinista


def test_sin_drift_spec_el_dataset_es_el_de_siempre():
    """Compatibilidad: DriftSpec() no toca ni una extraccion aleatoria."""
    assert generate_dataset(800, seed=42).equals(generate_dataset(800, seed=42, drift=DriftSpec()))
    assert DriftSpec.from_shift(0.0).is_null
    assert not DriftSpec.from_shift(0.1).is_null


def test_drift_es_reproducible_y_distinto_del_original():
    spec = DriftSpec.from_shift(1.0)
    a = generate_dataset(800, seed=5, drift=spec)
    assert a.equals(generate_dataset(800, seed=5, drift=spec))
    assert not a.equals(generate_dataset(800, seed=5))


@pytest.mark.parametrize("shift", [0.5, 1.0, 2.0])
def test_drift_respeta_el_contrato_de_validacion(shift):
    df = generate_dataset(5000, seed=42, drift=DriftSpec.from_shift(shift))
    for col, (lo, hi) in RANGES.items():
        assert df[col].between(lo, hi).all(), col
    for col, allowed in ALLOWED_CATEGORIES.items():
        assert set(df[col].unique()) <= allowed, col
    lo, hi = CHURN_RATE_BOUNDS
    assert lo < df[TARGET].mean() < hi
    assert validate_training_data(df).is_valid


def test_drift_desplaza_las_features_por_encima_del_umbral_psi():
    base = generate_dataset(4000, seed=42)
    drifted = generate_dataset(4000, seed=43, drift=DriftSpec.from_shift(1.0))
    report = drift_report(
        base[FEATURE_COLUMNS], drifted[FEATURE_COLUMNS], NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    assert {"tenure_months", "monthly_charges", "support_tickets_90d"} <= set(
        report["drifted_features"]
    )
    assert drifted.tenure_months.mean() < base.tenure_months.mean()
    assert drifted.monthly_charges.mean() > base.monthly_charges.mean()
    assert (drifted.contract_type == "mensual").mean() > (base.contract_type == "mensual").mean()


def test_drift_cambia_la_relacion_con_la_etiqueta():
    """Concept drift: la fibra pasa de proteger a empujar la baja; un modelo viejo no lo sabe."""
    base = generate_dataset(8000, seed=42)
    drifted = generate_dataset(8000, seed=42, drift=DriftSpec.from_shift(1.0))
    efecto = lambda df: df[df.is_fiber == 1][TARGET].mean() - df[df.is_fiber == 0][TARGET].mean()  # noqa: E731
    assert efecto(base) < 0 < efecto(drifted)


def test_drift_shift_fuera_de_rango():
    with pytest.raises(ValueError, match="entre 0 y 2"):
        DriftSpec.from_shift(2.5)
    with pytest.raises(ValueError):
        DriftSpec.from_shift(-0.1)


def test_drift_spec_explicito_recorta_a_los_rangos_validos():
    extremo = DriftSpec(tenure_shift=500, monthly_charges_shift=1000, tickets_rate_shift=-5)
    df = generate_dataset(500, seed=1, drift=extremo)
    assert (df.tenure_months == 1).all()
    assert (df.monthly_charges == 500).all()
    assert (df.support_tickets_90d == 0).all()
