import numpy as np

from churn.data.generator import (
    CATEGORICAL_FEATURES,
    CONTRACT_TYPES,
    NUMERIC_FEATURES,
    PAYMENT_METHODS,
    generate_dataset,
)
from churn.data.validation import ALLOWED_CATEGORIES, RANGES, validate_training_data


def test_dataset_valido_pasa():
    df = generate_dataset(1000, seed=1)
    report = validate_training_data(df)
    assert report.is_valid
    assert report.errors == []


def test_columnas_faltantes():
    df = generate_dataset(1000, seed=1).drop(columns=["tenure_months"])
    report = validate_training_data(df)
    assert not report.is_valid
    assert any("Faltan columnas" in e for e in report.errors)


def test_nulos_detectados():
    df = generate_dataset(1000, seed=1)
    df.loc[10, "monthly_charges"] = np.nan
    report = validate_training_data(df)
    assert any("nulos" in e for e in report.errors)


def test_nulos_no_duplican_errores():
    """Un NaN no debe aparecer ademas como 'fuera de rango' ni como 'categoria nan'."""
    df = generate_dataset(1000, seed=1)
    df.loc[0, "monthly_charges"] = np.nan
    df.loc[1, "contract_type"] = np.nan
    report = validate_training_data(df)
    assert len(report.errors) == 1
    assert "nulos" in report.errors[0]


def test_fuera_de_rango():
    df = generate_dataset(1000, seed=1)
    df.loc[5, "monthly_charges"] = 9999.0
    report = validate_training_data(df)
    assert any("fuera de rango" in e for e in report.errors)


def test_valores_negativos_fuera_de_rango():
    df = generate_dataset(1000, seed=1)
    df.loc[5, "tenure_months"] = -1
    assert any("fuera de rango en tenure_months" in e for e in validate_training_data(df).errors)


def test_categoria_desconocida():
    df = generate_dataset(1000, seed=1)
    df.loc[3, "contract_type"] = "semanal"
    report = validate_training_data(df)
    assert any("Categorias desconocidas" in e for e in report.errors)


def test_dataset_pequeno():
    df = generate_dataset(100, seed=1)
    report = validate_training_data(df)
    assert any("pequeno" in e for e in report.errors)


def test_tasa_churn_sospechosa():
    df = generate_dataset(1000, seed=1)
    df["churn"] = 0
    report = validate_training_data(df)
    assert any("Tasa de churn sospechosa" in e for e in report.errors)


def test_invalid_target_cannot_pass_using_plausible_mean():
    df = generate_dataset(1000, seed=1)
    df["churn"] = 0.3
    report = validate_training_data(df)
    assert not report.is_valid
    assert any("solo 0 o 1" in e for e in report.errors)


def test_target_todo_nulo_solo_reporta_nulos():
    df = generate_dataset(1000, seed=1)
    df["churn"] = np.nan
    assert validate_training_data(df).errors == ["Columnas con nulos: ['churn']"]


def test_target_texto_rechazado():
    df = generate_dataset(1000, seed=1)
    df["churn"] = df["churn"].map({0: "no", 1: "si"})
    assert any("solo 0 o 1" in e for e in validate_training_data(df).errors)


def test_text_numeric_column_returns_report_instead_of_type_error():
    df = generate_dataset(1000, seed=1)
    df["monthly_charges"] = df["monthly_charges"].astype(str)
    report = validate_training_data(df)
    assert not report.is_valid
    assert any("no numerica" in e for e in report.errors)


def test_mixed_unknown_categories_return_report():
    df = generate_dataset(1000, seed=1)
    df["contract_type"] = df["contract_type"].astype(object)
    df.loc[0, "contract_type"] = 123
    df.loc[1, "contract_type"] = "unknown"
    assert not validate_training_data(df).is_valid


def test_duplicate_columns_return_report():
    df = generate_dataset(1000, seed=1)
    df = df.loc[:, [*df.columns, "monthly_charges"]]
    assert not validate_training_data(df).is_valid


def test_acumula_varios_errores_en_un_solo_informe():
    df = generate_dataset(600, seed=1)
    df.loc[0, "monthly_charges"] = -5.0
    df.loc[1, "payment_method"] = "bitcoin"
    errors = validate_training_data(df).errors
    assert any("fuera de rango" in e for e in errors)
    assert any("Categorias desconocidas" in e for e in errors)


# ---------------------------------------------------------------- fuente unica de verdad


def test_categorias_permitidas_derivadas_del_generador():
    assert ALLOWED_CATEGORIES["contract_type"] == set(CONTRACT_TYPES)
    assert ALLOWED_CATEGORIES["payment_method"] == set(PAYMENT_METHODS)
    assert set(ALLOWED_CATEGORIES) == set(CATEGORICAL_FEATURES)


def test_rangos_cubren_todas_las_numericas():
    assert set(RANGES) == set(NUMERIC_FEATURES)
    assert all(lo < hi for lo, hi in RANGES.values())
