"""Configuracion 12-factor. Prefijo de entorno: CHURN_."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Valores de token que NUNCA deben llegar a produccion (defaults y placeholders de ejemplo).
INSECURE_ADMIN_TOKENS = frozenset(
    {"", "cambia-este-token", "genera-un-token-seguro", "token-local"}
)
MIN_ADMIN_TOKEN_LENGTH = 16
PRODUCTION_ENVIRONMENTS = frozenset({"prod", "production"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHURN_", env_file=".env", extra="ignore")

    app_name: str = "mlops-churn-platform"
    environment: str = "dev"
    log_level: str = "INFO"

    # Artefactos del modelo (volumen compartido entre trainer y api)
    model_dir: str = "models"
    # Versiones publicadas que se conservan en model_dir/versions (la actual nunca se borra)
    model_keep_versions: int = Field(default=5, ge=1)

    # MLflow (opcional): si esta definido, el entrenamiento registra runs y modelo
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "churn"
    mlflow_registered_model: str = "churn-classifier"
    # Alias que se asigna en el Model Registry a cada version que supera el gate ("" = no asignar)
    mlflow_champion_alias: str = "champion"

    # Gate de calidad: el entrenamiento NO publica artefactos si el AUC queda por debajo
    min_roc_auc: float = Field(default=0.75, ge=0.5, le=1.0)

    # Si es True, la API rechaza cargar un modelo entrenado con otra version menor de
    # scikit-learn (los pickles no son portables); las features incompatibles se rechazan siempre
    strict_artifact_compat: bool = True

    # Umbrales de negocio para clasificar el riesgo (bajo < medium <= medio < high <= alto)
    risk_medium: float = Field(default=0.35, ge=0.0, le=1.0)
    risk_high: float = Field(default=0.65, ge=0.0, le=1.0)

    # Monitorizacion de drift
    drift_min_rows: int = Field(default=200, ge=2)
    drift_buffer_size: int = Field(default=5000, ge=2)
    psi_alert_threshold: float = Field(default=0.2, gt=0.0)
    # Donde viven las predicciones recientes: "sqlite" (persistente, compartido entre
    # procesos; fichero drift_db_path o <model_dir>/drift.sqlite) o "memory" (por proceso)
    drift_store: Literal["sqlite", "memory"] = "sqlite"
    drift_db_path: str = ""

    # Token para operaciones administrativas (p. ej. /model/reload)
    admin_token: str = "cambia-este-token"

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in PRODUCTION_ENVIRONMENTS

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.risk_medium >= self.risk_high:
            raise ValueError(
                f"risk_medium ({self.risk_medium}) debe ser menor que risk_high ({self.risk_high})"
            )
        if self.drift_min_rows > self.drift_buffer_size:
            raise ValueError(
                f"drift_min_rows ({self.drift_min_rows}) no puede superar "
                f"drift_buffer_size ({self.drift_buffer_size}): el drift nunca se calcularia"
            )
        if self.is_production and not self.admin_token_is_secure:
            raise ValueError(
                "CHURN_ADMIN_TOKEN inseguro para produccion: define un token aleatorio de al "
                f"menos {MIN_ADMIN_TOKEN_LENGTH} caracteres (p. ej. `openssl rand -hex 32`)"
            )
        return self

    @property
    def admin_token_is_secure(self) -> bool:
        return (
            self.admin_token not in INSECURE_ADMIN_TOKENS
            and len(self.admin_token) >= MIN_ADMIN_TOKEN_LENGTH
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
