from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from churn.data.generator import CONTRACT_TYPES, FEATURE_COLUMNS, PAYMENT_METHODS
from churn.data.validation import RANGES
from churn.serving.schemas import (
    EXAMPLE_CUSTOMER,
    OBSERVED_AT_FUTURE_TOLERANCE,
    BatchPredictionRequest,
    CustomerFeatures,
    DataSource,
    DriftResponse,
    LabelIn,
    LabelsRequest,
    LabelsResponse,
    ModelInfoResponse,
    PerformanceResponse,
    PredictionRequest,
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
    ok = dict(prediction_id="a" * 32, risk_level="alto", model_version="v")
    PredictionResponse(churn_probability=0.2, **ok)
    with pytest.raises(ValidationError):
        PredictionResponse(churn_probability=1.2, **ok)
    with pytest.raises(ValidationError):
        PredictionResponse(churn_probability=0.2, **{**ok, "risk_level": "extremo"})
    with pytest.raises(ValidationError):  # sin prediction_id no se puede etiquetar despues
        PredictionResponse(churn_probability=0.2, risk_level="alto", model_version="v")


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


# --------------------------------------------------------------------------- etiquetas


def _labels(*ids: str, churn: int = 1) -> dict:
    return {"labels": [{"prediction_id": i, "churn": churn} for i in ids]}


def test_labels_request_valido_y_observed_at_opcional():
    body = LabelsRequest(**_labels("a", "b"))
    assert [label.observed_at for label in body.labels] == [None, None]
    con_fecha = LabelIn(prediction_id="a", churn=0, observed_at="2026-10-01T12:00:00+02:00")
    assert con_fecha.observed_at is not None and con_fecha.observed_at.utcoffset() is not None


def test_labels_request_rechaza_ids_repetidos():
    with pytest.raises(ValidationError, match="repetido.*\\['a'\\]"):
        LabelsRequest(**_labels("a", "b", "a"))


def test_labels_request_acota_el_lote():
    with pytest.raises(ValidationError):
        LabelsRequest(labels=[])
    with pytest.raises(ValidationError):
        LabelsRequest(**_labels(*(f"id-{i}" for i in range(1001))))
    assert len(LabelsRequest(**_labels(*(f"id-{i}" for i in range(1000)))).labels) == 1000


@pytest.mark.parametrize(
    "label",
    [
        {"prediction_id": "a", "churn": 2},
        {"prediction_id": "a", "churn": "si"},
        {"prediction_id": "a", "churn": 0.5},
        {"prediction_id": "", "churn": 1},
        {"prediction_id": "a" * 65, "churn": 1},
        {"churn": 1},
        {"prediction_id": "a", "churn": 1, "observed_at": "ayer"},
    ],
)
def test_label_invalida(label):
    with pytest.raises(ValidationError):
        LabelIn(**label)


def test_data_source_admite_la_procedencia_de_etiquetas():
    source = DataSource(
        kind="file",
        rows=600,
        fingerprint="f" * 64,
        path="/models/datasets/labels.parquet",
        labels={
            "predicted_from": "2026-09-01T00:00:00+00:00",
            "model_versions": ["v1"],
            "labels_total": 640,
            "subjects": 580,
        },
    )
    assert source.labels is not None
    assert source.labels.model_versions == ["v1"]
    assert (source.labels.labels_total, source.labels.subjects) == (640, 580)
    assert source.labels.observed_to is None
    assert DataSource(kind="file", rows=1, fingerprint="f").labels is None
    with pytest.raises(ValidationError):
        DataSource(kind="crm", rows=1, fingerprint="f")


def test_data_source_no_amplia_kind_para_no_romper_el_rollback_de_imagen():
    """La imagen anterior valida `kind` con Literal["synthetic", "file", "dataframe"]: un
    valor nuevo romperia su /model/info tras un rollback. El bloque `labels` es aditivo."""
    with pytest.raises(ValidationError):
        DataSource(kind="labels", rows=1, fingerprint="f")


def test_performance_response_admite_auc_indefinido():
    block = {
        "n": 3,
        "churn_rate": 1.0,
        "roc_auc": None,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "brier": 0.2,
    }
    resp = PerformanceResponse(
        labelled_total=3, threshold=0.5, overall=block, by_version={"v1": block}
    )
    assert resp.serving_version is None and resp.holdout_roc_auc is None
    assert resp.by_version["v1"].roc_auc is None


# --------------------------------------------------------------------------- sujeto y tiempos


def test_prediction_request_subject_ref_opcional_y_fuera_de_las_features():
    sin = PredictionRequest(**EXAMPLE_CUSTOMER)
    assert sin.subject_ref is None and sin.features() == EXAMPLE_CUSTOMER
    con = PredictionRequest(**EXAMPLE_CUSTOMER, subject_ref="c-1")
    assert con.features() == EXAMPLE_CUSTOMER  # no viaja al modelo
    for invalido in ("", "x" * 129):
        with pytest.raises(ValidationError):
            PredictionRequest(**EXAMPLE_CUSTOMER, subject_ref=invalido)
    batch = BatchPredictionRequest(customers=[{**EXAMPLE_CUSTOMER, "subject_ref": "c-2"}])
    assert batch.customers[0].subject_ref == "c-2"
    ejemplo = PredictionRequest.model_json_schema()["examples"][0]
    assert set(ejemplo) == set(EXAMPLE_CUSTOMER) | {"subject_ref"}


def test_label_subject_ref_opcional():
    assert LabelIn(prediction_id="a", churn=1).subject_ref is None
    assert LabelIn(prediction_id="a", churn=1, subject_ref="c-1").subject_ref == "c-1"
    with pytest.raises(ValidationError):
        LabelIn(prediction_id="a", churn=1, subject_ref="")


def test_label_observed_at_no_puede_ser_futura():
    ahora = datetime.now(UTC)
    with pytest.raises(ValidationError, match="observed_at futura"):
        LabelIn(prediction_id="a", churn=1, observed_at=ahora + timedelta(hours=1))
    # naive = UTC; hora local sin zona dos horas "por delante" se detecta
    naive = (ahora + timedelta(hours=2)).replace(tzinfo=None)
    with pytest.raises(ValidationError, match="zona horaria"):
        LabelIn(prediction_id="a", churn=1, observed_at=naive)
    holgura = ahora + OBSERVED_AT_FUTURE_TOLERANCE - timedelta(seconds=30)
    assert LabelIn(prediction_id="a", churn=0, observed_at=holgura).observed_at == holgura
    assert LabelIn(prediction_id="a", churn=0, observed_at=ahora - timedelta(days=30))
    assert LabelIn(prediction_id="a", churn=0, observed_at=None).observed_at is None  # JSON null


def test_labels_response_rejected_por_defecto_vacio():
    resp = LabelsResponse(received=1, created=1, updated=0, unknown=[], labelled_total=1)
    assert resp.rejected == []
