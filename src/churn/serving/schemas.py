"""Contratos Pydantic de la API de scoring.

Los rangos numericos se derivan de `churn.data.validation.RANGES` (misma fuente que
la validacion de entrenamiento). Las categorias son `Literal` para que aparezcan
como enumeraciones en OpenAPI; un test de contrato comprueba que coinciden con
`churn.data.generator`.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from churn.data.validation import RANGES

RiskLevel = Literal["bajo", "medio", "alto"]

EXAMPLE_CUSTOMER: dict[str, Any] = {
    "tenure_months": 3,
    "monthly_charges": 95.5,
    "total_charges": 280.0,
    "num_products": 1,
    "support_tickets_90d": 6,
    "is_fiber": 0,
    "contract_type": "mensual",
    "payment_method": "transferencia",
}


def _bounded(feature: str, description: str) -> Any:
    lo, hi = RANGES[feature]
    return Field(..., ge=lo, le=hi, description=description)


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    tenure_months: int = _bounded("tenure_months", "Antiguedad del cliente en meses")
    monthly_charges: float = _bounded("monthly_charges", "Cuota mensual")
    total_charges: float = _bounded("total_charges", "Facturacion acumulada")
    num_products: int = _bounded("num_products", "Productos contratados")
    support_tickets_90d: int = _bounded("support_tickets_90d", "Tickets de soporte (90 dias)")
    is_fiber: int = _bounded("is_fiber", "1 si tiene fibra, 0 si no")
    contract_type: Literal["mensual", "anual", "bianual"]
    payment_method: Literal["domiciliacion", "tarjeta", "transferencia"]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction_id: str = Field(
        ...,
        description="Identificador de esta prediccion: guardalo junto al cliente para "
        "devolver despues su churn real en POST /labels",
    )
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    model_version: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class LabelIn(BaseModel):
    """Churn real observado para una prediccion ya servida."""

    prediction_id: str = Field(..., min_length=1, max_length=64)
    churn: Literal[0, 1] = Field(..., description="1 si el cliente se dio de baja, 0 si no")
    observed_at: datetime | None = Field(
        None,
        description="Cuando se observo el resultado (ISO-8601; naive = UTC). "
        "Por defecto, el instante de recepcion",
    )


class LabelsRequest(BaseModel):
    labels: list[LabelIn] = Field(..., min_length=1, max_length=1000)

    @model_validator(mode="after")
    def _ids_unicos(self) -> "LabelsRequest":
        ids = [label.prediction_id for label in self.labels]
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        if duplicated:
            raise ValueError(f"prediction_id repetido en el lote: {duplicated}")
        return self


class LabelsResponse(BaseModel):
    received: int
    created: int = Field(..., description="Predicciones etiquetadas por primera vez")
    updated: int = Field(..., description="Etiquetas que sobrescriben una anterior")
    unknown: list[str] = Field(
        ..., description="prediction_id que no estan (o ya no estan) en el almacen"
    )
    labelled_total: int = Field(..., description="Predicciones etiquetadas en total")


class PerformanceBlock(BaseModel):
    n: int
    churn_rate: float
    roc_auc: float | None = Field(None, description="None si la ventana solo tiene una clase")
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier: float


class PerformanceResponse(BaseModel):
    serving_version: str | None = Field(None, description="Version cargada en este proceso")
    holdout_roc_auc: float | None = Field(
        None, description="AUC de la version en servicio sobre su holdout de entrenamiento"
    )
    labelled_total: int
    threshold: float
    overall: PerformanceBlock
    by_version: dict[str, PerformanceBlock]


class QualityGate(BaseModel):
    min_roc_auc: float
    passed: bool


PromotionDecision = Literal["promoted", "rejected", "no_champion"]


class Promotion(BaseModel):
    """Resultado de la comparacion campeon/retador al entrenar esta version."""

    decision: PromotionDecision
    margin: float
    challenger_roc_auc: float
    champion_version: str | None = None
    champion_roc_auc: float | None = None
    identical_training_data: bool = False
    reason: str


class LabelsWindow(BaseModel):
    """Ventana del dataset construido desde las etiquetas de produccion (sidecar)."""

    built_at: str | None = None
    predicted_from: str | None = None
    predicted_to: str | None = None
    observed_from: str | None = None
    observed_to: str | None = None
    model_versions: list[str] = []


class DataSource(BaseModel):
    kind: Literal["synthetic", "file", "labels", "dataframe"]
    rows: int
    fingerprint: str
    path: str | None = None
    seed: int | None = None
    drift_shift: float | None = None
    labels: LabelsWindow | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    algorithm: str
    trained_at: str
    metrics: dict[str, int | float]
    numeric_features: list[str]
    categorical_features: list[str]
    code_version: str | None = None
    seed: int | None = None
    quality_gate: QualityGate | None = None
    promotion: Promotion | None = None
    data_source: DataSource | None = None
    runtime: dict[str, str] | None = None
    mlflow_run_id: str | None = None
    mlflow_model_version: str | None = None


class ReloadResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    reloaded: bool
    model_version: str
    previous_version: str | None = None


class ModelVersionInfo(BaseModel):
    version: str
    current: bool
    complete: bool
    trained_at: str | None = None
    roc_auc: float | None = None
    promotion: PromotionDecision | None = Field(
        None, description="Decision campeon/retador al entrenarla (None en versiones antiguas)"
    )


class ModelVersionsResponse(BaseModel):
    current: str | None = Field(None, description="Version apuntada por `current` en el almacen")
    serving: str | None = Field(None, description="Version cargada en memoria en este proceso")
    versions: list[ModelVersionInfo]


class RollbackRequest(BaseModel):
    version: str | None = Field(
        None, description="Version a la que volver; por defecto la anterior a la actual"
    )


class RollbackResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    previous_version: str | None = None
    available_versions: list[str]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str | None = None


class LivenessResponse(BaseModel):
    status: Literal["alive"]
    code_version: str


class FeatureDrift(BaseModel):
    type: Literal["numeric", "categorical"]
    psi: float
    drift: bool
    ks_statistic: float | None = None
    ks_pvalue: float | None = None


class PredictionDrift(BaseModel):
    psi: float
    n_reference: int
    n_current: int
    mean_reference: float | None = None
    mean_current: float | None = None
    drift: bool


class DriftResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    n_reference: int
    n_current: int
    psi_threshold: float
    features: dict[str, FeatureDrift]
    drifted_features: list[str]
    predictions: PredictionDrift | None = None
    drift_detected: bool
