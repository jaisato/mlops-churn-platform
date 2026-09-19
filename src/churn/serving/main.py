"""API de scoring en tiempo real + monitorizacion de drift.

Ademas de servir predicciones, la API guarda un buffer rodante con las features
recibidas en produccion (y la probabilidad devuelta) y expone /monitoring/drift
para compararlas contra la muestra de referencia del entrenamiento (PSI + KS) y
contra la distribucion de puntuaciones del modelo (prediction drift).
"""

from __future__ import annotations

import logging
import secrets
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request

from churn import __version__
from churn.config import Settings, get_settings
from churn.logging_conf import configure_logging
from churn.monitoring.drift import drift_report
from churn.registry import LoadedModel, LocalModelStore
from churn.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    DriftResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReloadResponse,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServedModel:
    """Modelo cargado + puntuaciones de referencia precalculadas (prediction drift)."""

    loaded: LoadedModel
    reference_scores: np.ndarray

    @property
    def pipeline(self) -> Any:
        return self.loaded.pipeline

    @property
    def metadata(self) -> dict[str, Any]:
        return self.loaded.metadata

    @property
    def reference(self) -> pd.DataFrame:
        return self.loaded.reference

    @property
    def version(self) -> str:
        return self.loaded.version


def risk_level(probability: float, medium: float, high: float) -> str:
    """Nivel de riesgo de negocio: alto si p >= high, medio si p >= medium, bajo en otro caso."""
    if probability >= high:
        return "alto"
    if probability >= medium:
        return "medio"
    return "bajo"


def _load_model(app: FastAPI) -> ServedModel:
    """Carga los artefactos y precalcula las puntuaciones de la referencia."""
    store: LocalModelStore = app.state.store
    loaded = store.load()
    scores = loaded.pipeline.predict_proba(loaded.reference[loaded.feature_columns])[:, 1]
    served = ServedModel(loaded=loaded, reference_scores=np.asarray(scores, dtype=float))
    app.state.model = served  # asignacion atomica: las peticiones en vuelo ven el viejo o el nuevo
    return served


def _try_load_on_startup(app: FastAPI) -> None:
    store: LocalModelStore = app.state.store
    try:
        served = _load_model(app)
    except FileNotFoundError:
        app.state.model = None
        logger.warning("No hay modelo completo en %s; la API respondera 503", store.model_dir)
    except Exception:  # artefactos corruptos: arrancamos sin modelo para permitir /model/reload
        app.state.model = None
        logger.exception("Artefactos ilegibles en %s; la API respondera 503", store.model_dir)
    else:
        logger.info("Modelo cargado: %s", served.version)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.store = LocalModelStore(settings.model_dir)
        app.state.buffer = deque(maxlen=settings.drift_buffer_size)
        _try_load_on_startup(app)
        yield

    app = FastAPI(
        title="Churn Prediction API",
        description="Scoring de churn con modelo gestionado por pipeline MLOps.",
        version=__version__,
        lifespan=lifespan,
    )

    def _model(request: Request) -> ServedModel:
        model = request.app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Modelo no disponible todavia")
        return model

    def _risk(p: float) -> str:
        return risk_level(p, settings.risk_medium, settings.risk_high)

    def _score(model: ServedModel, customers: list[CustomerFeatures]) -> list[PredictionResponse]:
        rows = [c.model_dump() for c in customers]
        probas = model.pipeline.predict_proba(pd.DataFrame(rows))[:, 1]
        buffer = app.state.buffer
        predictions = []
        for row, p in zip(rows, probas, strict=True):
            p = float(p)
            buffer.append({**row, "churn_probability": p, "model_version": model.version})
            predictions.append(
                PredictionResponse(
                    churn_probability=round(p, 4), risk_level=_risk(p), model_version=model.version
                )
            )
        return predictions

    @app.get("/health", response_model=HealthResponse, tags=["sistema"])
    def health(request: Request):
        model = request.app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Modelo no cargado")
        return HealthResponse(status="ok", model_loaded=True, model_version=model.version)

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["modelo"])
    def model_info(request: Request):
        return ModelInfoResponse(**_model(request).metadata)

    @app.post("/model/reload", response_model=ReloadResponse, tags=["modelo"])
    def model_reload(request: Request, x_admin_token: str = Header(default="")):
        # compare_digest: comparacion en tiempo constante (no filtra el token por timing)
        if not secrets.compare_digest(x_admin_token.encode(), settings.admin_token.encode()):
            raise HTTPException(status_code=401, detail="Token de administracion invalido")
        previous = request.app.state.model
        previous_version = previous.version if previous else None
        try:
            served = _load_model(request.app)
        except FileNotFoundError as exc:
            # Nunca degradamos el servicio por un reload fallido: se conserva el modelo actual.
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # ModelArtifactError u otro fallo de carga
            logger.exception("Reload fallido; se mantiene la version %s", previous_version)
            raise HTTPException(
                status_code=500, detail=f"No se pudo cargar el modelo: {exc}"
            ) from exc
        logger.info("Modelo recargado: %s -> %s", previous_version, served.version)
        return ReloadResponse(
            reloaded=True, model_version=served.version, previous_version=previous_version
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
    def predict(request: Request, features: CustomerFeatures):
        return _score(_model(request), [features])[0]

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["scoring"])
    def predict_batch(request: Request, body: BatchPredictionRequest):
        return BatchPredictionResponse(predictions=_score(_model(request), body.customers))

    @app.get("/monitoring/drift", response_model=DriftResponse, tags=["monitorizacion"])
    def monitoring_drift(request: Request):
        model = _model(request)
        buffer = request.app.state.buffer
        if len(buffer) < settings.drift_min_rows:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Datos insuficientes para drift: {len(buffer)}/{settings.drift_min_rows} "
                    "predicciones acumuladas"
                ),
            )
        current = pd.DataFrame(list(buffer))
        # Solo comparamos puntuaciones producidas por la version en servicio: tras un
        # reload, las probabilidades del modelo anterior no son comparables.
        scores = current.loc[current["model_version"] == model.version, "churn_probability"]
        enough_scores = len(scores) >= settings.drift_min_rows
        report = drift_report(
            reference=model.reference,
            current=current,
            numeric_features=model.metadata["numeric_features"],
            categorical_features=model.metadata["categorical_features"],
            psi_threshold=settings.psi_alert_threshold,
            reference_scores=model.reference_scores if enough_scores else None,
            current_scores=scores if enough_scores else None,
        )
        return DriftResponse(model_version=model.version, **report)

    return app


app = create_app()
