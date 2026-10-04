"""Contratos Pydantic de la API de scoring.

Los rangos numericos se derivan de `churn.data.validation.RANGES` (misma fuente que
la validacion de entrenamiento). Las categorias son `Literal` para que aparezcan
como enumeraciones en OpenAPI; un test de contrato comprueba que coinciden con
`churn.data.generator`.
"""

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from churn.data.validation import RANGES

RiskLevel = Literal["bajo", "medio", "alto"]

#: Holgura frente al reloj del servidor para aceptar un `observed_at` (desfase entre relojes).
OBSERVED_AT_FUTURE_TOLERANCE = timedelta(minutes=5)
SUBJECT_REF_MAX_LENGTH = 128
SUBJECT_REF_DESCRIPTION = (
    "Opcional. Clave OPACA y estable del sujeto (el cliente puntuado), p. ej. un hash de su id "
    "interno; nunca datos personales. Agrupa las predicciones del mismo cliente: el dataset de "
    "reentreno se queda con una etiqueta por sujeto y horizonte y el holdout del entrenamiento "
    "se separa por sujeto, para que un cliente puntuado varias veces no quede a ambos lados"
)

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


def _subject_ref_field() -> Any:
    return Field(
        None, min_length=1, max_length=SUBJECT_REF_MAX_LENGTH, description=SUBJECT_REF_DESCRIPTION
    )


class PredictionRequest(CustomerFeatures):
    """Features del cliente + `subject_ref` opcional (no es una feature: no llega al modelo)."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{**EXAMPLE_CUSTOMER, "subject_ref": "c-7f3a9c41"}]}
    )

    subject_ref: str | None = _subject_ref_field()

    def features(self) -> dict[str, Any]:
        return self.model_dump(exclude={"subject_ref"})


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
    customers: list[PredictionRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class LabelIn(BaseModel):
    """Churn real observado para una prediccion ya servida."""

    prediction_id: str = Field(..., min_length=1, max_length=64)
    churn: Literal[0, 1] = Field(
        ...,
        description="1 si el cliente se dio de baja dentro del horizonte de la prediccion, 0 si "
        "el horizonte ha vencido sin baja. Un 0 solo es firme cuando el horizonte ha vencido "
        "(etiqueta madura): enviarlo antes es un falso negativo",
    )
    observed_at: datetime | None = Field(
        None,
        description="Cuando se observo el resultado: la fecha de la baja (churn=1) o el cierre "
        "del horizonte (churn=0), en ISO-8601 (naive = UTC). Por defecto, el instante de "
        "recepcion. No puede ser anterior a la prediccion (la etiqueta se rechaza) ni futura "
        "(422; holgura de 5 minutos)",
    )
    subject_ref: str | None = _subject_ref_field()

    @field_validator("observed_at")
    @classmethod
    def _observed_at_no_futura(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        if aware > datetime.now(UTC) + OBSERVED_AT_FUTURE_TOLERANCE:
            raise ValueError(
                f"observed_at futura ({aware.isoformat()}): la etiqueta se envia cuando el "
                "resultado ya se ha observado (si es hora local, indica la zona horaria)"
            )
        return value


class LabelsRequest(BaseModel):
    labels: list[LabelIn] = Field(..., min_length=1, max_length=1000)

    @model_validator(mode="after")
    def _ids_unicos(self) -> "LabelsRequest":
        counts = Counter(label.prediction_id for label in self.labels)
        duplicated = sorted(i for i, n in counts.items() if n > 1)
        if duplicated:
            raise ValueError(f"prediction_id repetido en el lote: {duplicated}")
        return self


class LabelsResponse(BaseModel):
    received: int
    created: int = Field(..., description="Predicciones etiquetadas por primera vez")
    updated: int = Field(..., description="Etiquetas que sobrescriben una anterior")
    unknown: list[str] = Field(
        ...,
        description="prediction_id que no estan en el almacen (ni como prediccion ni como "
        "etiqueta: nunca existieron o se desalojaron sin llegar a etiquetarse)",
    )
    rejected: list[str] = Field(
        default_factory=list,
        description="prediction_id cuya observed_at es anterior a la prediccion: la etiqueta "
        "no se guarda",
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
    labels_total: int | None = Field(None, description="Etiquetas almacenadas antes de deduplicar")
    subjects: int | None = Field(None, description="Sujetos distintos (`subject_ref`)")


class DataSource(BaseModel):
    # Sin "labels": un dataset de etiquetas es un fichero (`kind = "file"`) con el bloque
    # `labels`. Ampliar el Literal romperia /model/info al hacer rollback a una imagen anterior
    # (su esquema no conoceria el valor); un campo nuevo, en cambio, simplemente se ignora.
    kind: Literal["synthetic", "file", "dataframe"]
    rows: int
    fingerprint: str
    path: str | None = None
    seed: int | None = None
    drift_shift: float | None = None
    labels: LabelsWindow | None = Field(
        None, description="Presente si el fichero se construyo desde las etiquetas de la API"
    )


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
