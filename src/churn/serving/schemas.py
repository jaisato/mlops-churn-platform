"""Contratos Pydantic de la API de scoring.

Los rangos numericos se derivan de `churn.data.validation.RANGES` (misma fuente que
la validacion de entrenamiento). Las categorias son `Literal` para que aparezcan
como enumeraciones en OpenAPI; un test de contrato comprueba que coinciden con
`churn.data.generator`.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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

    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    model_version: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class QualityGate(BaseModel):
    min_roc_auc: float
    passed: bool


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
