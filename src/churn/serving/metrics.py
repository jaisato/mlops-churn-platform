"""Metricas Prometheus de la API (registro propio por aplicacion, no el global).

Un registro por `create_app()` evita el error "Duplicated timeseries" cuando se crean
varias aplicaciones en el mismo proceso (tests) y hace explicito que las metricas son
del servicio, no del interprete.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

__all__ = ["CONTENT_TYPE_LATEST", "Metrics"]

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
PROBABILITY_BUCKETS = tuple(round(0.05 * i, 2) for i in range(1, 20))
PERFORMANCE_GAUGES = ("labelled", "roc_auc", "precision", "recall", "f1")


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "churn_http_requests_total",
            "Peticiones HTTP atendidas",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "churn_http_request_duration_seconds",
            "Latencia de las peticiones HTTP",
            ["method", "path"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.predictions = Counter(
            "churn_predictions_total",
            "Predicciones servidas por nivel de riesgo y version del modelo",
            ["risk_level", "model_version"],
            registry=self.registry,
        )
        self.probability = Histogram(
            "churn_prediction_probability",
            "Distribucion de las probabilidades de churn devueltas",
            buckets=PROBABILITY_BUCKETS,
            registry=self.registry,
        )
        self.model_info = Gauge(
            "churn_model_info",
            "Version del modelo en servicio (siempre 1)",
            ["model_version"],
            registry=self.registry,
        )
        self.drift_psi = Gauge(
            "churn_drift_psi",
            "PSI por feature en el ultimo informe de drift",
            ["feature"],
            registry=self.registry,
        )
        self.prediction_drift_psi = Gauge(
            "churn_prediction_drift_psi",
            "PSI de las probabilidades predichas en el ultimo informe de drift",
            registry=self.registry,
        )
        self.drift_detected = Gauge(
            "churn_drift_detected",
            "1 si el ultimo informe de drift detecto cambio, 0 si no",
            registry=self.registry,
        )
        self.labels_received = Counter(
            "churn_labels_total",
            "Etiquetas recibidas en POST /labels por resultado (created, updated, unknown, "
            "rejected)",
            ["result"],
            registry=self.registry,
        )
        self.labels_stored = Gauge(
            "churn_labels_stored",
            "Predicciones etiquetadas disponibles para reentrenar",
            registry=self.registry,
        )
        # Rendimiento real por version del modelo (ultimo informe de /monitoring/performance)
        self.performance = {
            name: Gauge(
                f"churn_performance_{name}",
                f"{name} del modelo sobre las predicciones etiquetadas recientes, por version",
                ["model_version"],
                registry=self.registry,
            )
            for name in PERFORMANCE_GAUGES
        }

    def observe_request(self, method: str, path: str, status: int, duration: float) -> None:
        self.requests.labels(method, path, str(status)).inc()
        self.latency.labels(method, path).observe(duration)

    def observe_prediction(self, risk: str, model_version: str, probability: float) -> None:
        self.predictions.labels(risk, model_version).inc()
        self.probability.observe(probability)

    def set_model(self, model_version: str) -> None:
        self.model_info.clear()
        self.model_info.labels(model_version).set(1)

    def observe_drift(self, report: dict[str, Any]) -> None:
        for feature, result in report["features"].items():
            self.drift_psi.labels(feature).set(result["psi"])
        predictions = report.get("predictions")
        if predictions:
            self.prediction_drift_psi.set(predictions["psi"])
        self.drift_detected.set(1 if report["drift_detected"] else 0)

    def observe_labels(
        self, *, created: int, updated: int, unknown: int, rejected: int, stored: int
    ) -> None:
        self.labels_received.labels("created").inc(created)
        self.labels_received.labels("updated").inc(updated)
        self.labels_received.labels("unknown").inc(unknown)
        self.labels_received.labels("rejected").inc(rejected)
        self.set_labels_stored(stored)

    def set_labels_stored(self, stored: int) -> None:
        self.labels_stored.set(stored)

    def observe_performance(self, report: dict[str, Any]) -> None:
        """Publica las metricas de `by_version`; las versiones ausentes del informe se retiran."""
        for gauge in self.performance.values():
            gauge.clear()
        for version, block in report["by_version"].items():
            self.performance["labelled"].labels(version).set(block["n"])
            for name in ("roc_auc", "precision", "recall", "f1"):
                if block[name] is not None:  # AUC indefinido con una sola clase
                    self.performance[name].labels(version).set(block[name])

    def render(self) -> bytes:
        return generate_latest(self.registry)
