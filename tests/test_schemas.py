import pytest
from pydantic import ValidationError

from churn.data.generator import CONTRACT_TYPES, FEATURE_COLUMNS, PAYMENT_METHODS
from churn.data.validation import RANGES
from churn.serving.schemas import (
    EXAMPLE_CUSTOMER,
    CustomerFeatures,
    DriftResponse,
    ModelInfoResponse,
    PredictionResponse,
)


def test_rangos_de_la_api_coinciden_con_la_validacion_de_entrenamiento():
    props = CustomerFeatures.model_json_schema()["properties"]
    for feature, (lo, hi) in RANGES.items():
        assert props[feature]["minimum"] == lo, feature
        assert props[feature]["maximum"] == hi, feature


def test_categorias_de_la_api_coinciden_con_el_generador():
    props = CustomerFeatures.model_json_schema()["properties"]
    assert props["contract_type"]["enum"] == CONTRACT_TYPES
    assert props["payment_method"]["enum"] == PAYMENT_METHODS


def test_la_api_pide_exactamente_las_features_del_modelo():
    assert list(CustomerFeatures.model_fields) == FEATURE_COLUMNS


def test_ejemplo_documentado_es_valido():
    CustomerFeatures(**EXAMPLE_CUSTOMER)
    assert CustomerFeatures.model_json_schema()["examples"] == [EXAMPLE_CUSTOMER]


def test_rechaza_tipos_incorrectos():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**EXAMPLE_CUSTOMER, "tenure_months": "tres"})
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**EXAMPLE_CUSTOMER, "is_fiber": 0.5})


def test_prediction_response_acota_la_probabilidad():
    with pytest.raises(ValidationError):
        PredictionResponse(churn_probability=1.2, risk_level="alto", model_version="v")
    with pytest.raises(ValidationError):
        PredictionResponse(churn_probability=0.2, risk_level="extremo", model_version="v")


def test_model_info_tolera_metadata_antiguo_sin_campos_nuevos():
    """Modelos entrenados antes del gate/MLflow siguen siendo servibles."""
    info = ModelInfoResponse(
        model_version="v0",
        algorithm="X",
        trained_at="2026-01-01T00:00:00+00:00",
        metrics={"roc_auc": 0.9, "n_train": 100},
        numeric_features=["a"],
        categorical_features=[],
        extra_desconocido="ignorado",
    )
    assert info.quality_gate is None and info.mlflow_run_id is None
    assert info.metrics["n_train"] == 100


def test_drift_response_valida_estructura_de_features():
    with pytest.raises(ValidationError):
        DriftResponse(
            model_version="v",
            n_reference=1,
            n_current=1,
            psi_threshold=0.2,
            features={"a": {"type": "raro", "psi": 0.1, "drift": False}},
            drifted_features=[],
            drift_detected=False,
        )
