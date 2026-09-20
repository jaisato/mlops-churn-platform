"""API de scoring en tiempo real + monitorizacion de drift.

Ademas de servir predicciones, la API guarda las features recibidas en produccion (y
la probabilidad devuelta) en un almacen rodante y expone /monitoring/drift para
compararlas contra la muestra de referencia del entrenamiento (PSI + KS) y contra
la distribucion de puntuaciones del modelo (prediction drift).

Operaciones de modelo (autenticadas con X-Admin-Token): /model/reload carga la
version en servicio del almacen, /model/rollback vuelve a una version anterior.

Observabilidad: cada peticion lleva X-Request-ID (propagado o generado), se registra
como una linea JSON y alimenta las metricas Prometheus de /metrics.
"""

from __future__ import annotations

import logging
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response

from churn import __version__
from churn.config import Settings, get_settings
from churn.logging_conf import configure_logging, request_id_var
from churn.monitoring.drift import drift_report
from churn.monitoring.store import PredictionStore, build_prediction_store
from churn.registry import LoadedModel, LocalModelStore, verify_compatibility
from churn.serving.metrics import CONTENT_TYPE_LATEST, Metrics
from churn.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    DriftResponse,
    HealthResponse,
    LivenessResponse,
    ModelInfoResponse,
    ModelVersionInfo,
    ModelVersionsResponse,
    PredictionResponse,
    ReloadResponse,
    RiskLevel,
    RollbackRequest,
    RollbackResponse,
)

logger = logging.getLogger(__name__)
access_logger = logging.getLogger("churn.access")


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


def risk_level(probability: float, medium: float, high: float) -> RiskLevel:
    """Nivel de riesgo de negocio: alto si p >= high, medio si p >= medium, bajo en otro caso."""
    if probability >= high:
        return "alto"
    if probability >= medium:
        return "medio"
    return "bajo"


def _load_model(app: FastAPI, version: str | None = None) -> ServedModel:
    """Carga una version (por defecto la de `current`), la valida y la deja en servicio."""
    store: LocalModelStore = app.state.store
    settings: Settings = app.state.settings
    loaded = store.load(version)
    for warning in verify_compatibility(
        loaded.metadata, strict_runtime=settings.strict_artifact_compat
    ):
        logger.warning("Compatibilidad del modelo %s: %s", loaded.version, warning)
    scores = loaded.pipeline.predict_proba(loaded.reference[loaded.feature_columns])[:, 1]
    served = ServedModel(loaded=loaded, reference_scores=np.asarray(scores, dtype=float))
    app.state.model = served  # asignacion atomica: las peticiones en vuelo ven el viejo o el nuevo
    app.state.metrics.set_model(served.version)
    return served


def _try_load_on_startup(app: FastAPI) -> None:
    store: LocalModelStore = app.state.store
    try:
        served = _load_model(app)
    except FileNotFoundError:
        app.state.model = None
        logger.warning("No hay modelo completo en %s; la API respondera 503", store.model_dir)
    except Exception:  # corruptos o incompatibles: arrancamos sin modelo y se podra hacer reload
        app.state.model = None
        logger.exception("No se pudo cargar el modelo de %s; respondera 503", store.model_dir)
    else:
        logger.info("Modelo cargado: %s", served.version)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.metrics = metrics
        app.state.store = LocalModelStore(
            settings.model_dir, keep_versions=settings.model_keep_versions
        )
        app.state.predictions = build_prediction_store(settings)
        _try_load_on_startup(app)
        yield

    app = FastAPI(
        title="Churn Prediction API",
        description="Scoring de churn con modelo gestionado por pipeline MLOps.",
        version=__version__,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ observabilidad

    def _record_request(request: Request, status: int, started: float, request_id: str) -> None:
        duration = time.perf_counter() - started
        route = request.scope.get("route")
        template = getattr(route, "path", "unmatched")  # plantilla, no la URL: cardinalidad acotada
        metrics.observe_request(request.method, template, status, duration)
        if settings.access_log and request.url.path != "/metrics":
            access_logger.info(
                "%s %s -> %s",
                request.method,
                request.url.path,
                status,
                extra={
                    "request_id": request_id,
                    "http": {
                        "method": request.method,
                        "path": request.url.path,
                        "route": template,
                        "status": status,
                        "duration_ms": round(duration * 1000, 2),
                    },
                },
            )

    @app.middleware("http")
    async def observability(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            _record_request(request, 500, started, request_id)
            raise
        else:
            response.headers["X-Request-ID"] = request_id
            _record_request(request, response.status_code, started, request_id)
            return response
        finally:
            request_id_var.reset(token)

    # ------------------------------------------------------------------ helpers

    def _model(request: Request) -> ServedModel:
        model = request.app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Modelo no disponible todavia")
        return model

    def _require_admin(token: str) -> None:
        # compare_digest: comparacion en tiempo constante (no filtra el token por timing)
        if not secrets.compare_digest(token.encode(), settings.admin_token.encode()):
            raise HTTPException(status_code=401, detail="Token de administracion invalido")

    def require_api_key(x_api_key: str = Header(default="")) -> None:
        """Protege scoring, informacion del modelo y drift cuando CHURN_API_KEY esta definida."""
        if not settings.api_key:
            return
        if not secrets.compare_digest(x_api_key.encode(), settings.api_key.encode()):
            raise HTTPException(
                status_code=401, detail="API key invalida o ausente (cabecera X-API-Key)"
            )

    protected = [Depends(require_api_key)]

    def _risk(p: float) -> RiskLevel:
        return risk_level(p, settings.risk_medium, settings.risk_high)

    def _score(model: ServedModel, customers: list[CustomerFeatures]) -> list[PredictionResponse]:
        rows = [c.model_dump() for c in customers]
        probas = [float(p) for p in model.pipeline.predict_proba(pd.DataFrame(rows))[:, 1]]
        store: PredictionStore = app.state.predictions
        store.append(
            {**row, "churn_probability": p, "model_version": model.version}
            for row, p in zip(rows, probas, strict=True)
        )
        predictions = []
        for p in probas:
            risk = _risk(p)
            metrics.observe_prediction(risk, model.version, p)
            predictions.append(
                PredictionResponse(
                    churn_probability=round(p, 4), risk_level=risk, model_version=model.version
                )
            )
        return predictions

    # ------------------------------------------------------------------ sistema

    @app.get("/health/live", response_model=LivenessResponse, tags=["sistema"])
    def health_live():
        """Liveness: el proceso responde. No depende de que haya modelo cargado."""
        return LivenessResponse(status="alive", code_version=__version__)

    @app.get("/health", response_model=HealthResponse, tags=["sistema"])
    def health(request: Request):
        """Readiness: 200 solo si hay un modelo cargado y listo para puntuar."""
        model = request.app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Modelo no cargado")
        return HealthResponse(status="ok", model_loaded=True, model_version=model.version)

    if settings.metrics_enabled:

        @app.get("/metrics", tags=["sistema"], include_in_schema=False)
        def prometheus_metrics():
            return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)

    # ------------------------------------------------------------------ modelo

    @app.get(
        "/model/info", response_model=ModelInfoResponse, tags=["modelo"], dependencies=protected
    )
    def model_info(request: Request):
        return ModelInfoResponse(**_model(request).metadata)

    @app.get(
        "/model/versions",
        response_model=ModelVersionsResponse,
        tags=["modelo"],
        dependencies=protected,
    )
    def model_versions(request: Request):
        store: LocalModelStore = request.app.state.store
        served = request.app.state.model
        return ModelVersionsResponse(
            current=store.current_version() or (served.version if served else None),
            serving=served.version if served else None,
            versions=[ModelVersionInfo(**entry) for entry in store.describe_versions()],
        )

    @app.post("/model/reload", response_model=ReloadResponse, tags=["modelo"])
    def model_reload(request: Request, x_admin_token: str = Header(default="")):
        _require_admin(x_admin_token)
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

    @app.post("/model/rollback", response_model=RollbackResponse, tags=["modelo"])
    def model_rollback(
        request: Request,
        body: RollbackRequest | None = None,
        x_admin_token: str = Header(default=""),
    ):
        """Vuelve a una version publicada (por defecto la anterior) y la deja como `current`.

        El puntero solo cambia si la version objetivo se carga bien: un rollback a una
        version corrupta deja el servicio y el puntero exactamente como estaban.
        """
        _require_admin(x_admin_token)
        store: LocalModelStore = request.app.state.store
        requested = body.version if body else None
        try:
            target = store.resolve_rollback_target(requested)
        except LookupError as exc:
            raise HTTPException(status_code=404 if requested else 409, detail=str(exc)) from exc
        except Exception as exc:  # version incompleta
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        previous = request.app.state.model
        previous_version = previous.version if previous else None
        try:
            served = _load_model(request.app, version=target)
        except Exception as exc:
            logger.exception("Rollback a %s fallido; se mantiene %s", target, previous_version)
            raise HTTPException(
                status_code=500, detail=f"No se pudo cargar la version {target}: {exc}"
            ) from exc
        store.set_current(target)
        logger.info("Rollback: %s -> %s", previous_version, served.version)
        return RollbackResponse(
            model_version=served.version,
            previous_version=previous_version,
            available_versions=store.list_versions(),
        )

    # ------------------------------------------------------------------ scoring

    @app.post(
        "/predict", response_model=PredictionResponse, tags=["scoring"], dependencies=protected
    )
    def predict(request: Request, features: CustomerFeatures):
        return _score(_model(request), [features])[0]

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        tags=["scoring"],
        dependencies=protected,
    )
    def predict_batch(request: Request, body: BatchPredictionRequest):
        return BatchPredictionResponse(predictions=_score(_model(request), body.customers))

    # ------------------------------------------------------------------ monitorizacion

    @app.get(
        "/monitoring/drift",
        response_model=DriftResponse,
        tags=["monitorizacion"],
        dependencies=protected,
    )
    def monitoring_drift(request: Request):
        model = _model(request)
        store: PredictionStore = request.app.state.predictions
        current = store.recent(settings.drift_buffer_size)
        if len(current) < settings.drift_min_rows:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Datos insuficientes para drift: {len(current)}/{settings.drift_min_rows} "
                    "predicciones acumuladas"
                ),
            )
        # Solo comparamos puntuaciones producidas por la version en servicio: tras un
        # reload o rollback, las probabilidades de otra version no son comparables.
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
        metrics.observe_drift(report)
        return DriftResponse(model_version=model.version, **report)

    return app


app = create_app()
