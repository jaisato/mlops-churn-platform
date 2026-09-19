"""Entrenamiento del modelo de churn con tracking opcional en MLflow.

Uso:
    python -m churn.training.train --rows 20000 --model-dir models [--min-auc 0.8]
Variables:
    CHURN_MLFLOW_TRACKING_URI  si esta definida, se registran parametros,
    metricas, artefactos y version del modelo en MLflow Model Registry.
    CHURN_MIN_ROC_AUC          gate de calidad: por debajo de este AUC el
    entrenamiento falla y NO sobreescribe los artefactos del modelo en servicio.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn import __version__
from churn.config import get_settings
from churn.data.generator import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    generate_dataset,
)
from churn.data.validation import validate_training_data
from churn.logging_conf import configure_logging
from churn.registry import LocalModelStore

logger = logging.getLogger(__name__)

REFERENCE_SAMPLE_SIZE = 2000
DECISION_THRESHOLD = 0.5


class ModelQualityError(ValueError):
    """El modelo entrenado no supera el gate de calidad; no se publican artefactos."""


def build_pipeline(seed: int = 42) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.08, max_iter=250, random_state=seed
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def new_model_version(now: datetime | None = None) -> str:
    """Version legible por timestamp + sufijo aleatorio (evita colisiones en el mismo segundo)."""
    now = now or datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"


def train(
    df: pd.DataFrame, model_dir: str, seed: int = 42, min_roc_auc: float | None = None
) -> dict:
    """Entrena, evalua, aplica el gate de calidad y persiste artefactos.

    Devuelve el metadata generado. Lanza `ValueError` si los datos no son validos y
    `ModelQualityError` si el AUC no alcanza `min_roc_auc` (por defecto
    `CHURN_MIN_ROC_AUC`); en ambos casos no se toca el directorio del modelo.
    """
    if min_roc_auc is None:
        min_roc_auc = get_settings().min_roc_auc

    report = validate_training_data(df)
    if not report.is_valid:
        raise ValueError(f"Datos de entrenamiento invalidos: {report.errors}")

    x = df[FEATURE_COLUMNS]
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )

    pipeline = build_pipeline(seed=seed)
    pipeline.fit(x_train, y_train)

    proba = pipeline.predict_proba(x_test)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "brier": round(float(brier_score_loss(y_test, proba)), 4),
        "churn_rate_train": round(float(y_train.mean()), 4),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }

    quality_gate = {"min_roc_auc": min_roc_auc, "passed": metrics["roc_auc"] >= min_roc_auc}
    if not quality_gate["passed"]:
        raise ModelQualityError(
            f"AUC {metrics['roc_auc']} por debajo del minimo {min_roc_auc}: "
            "se conservan los artefactos anteriores"
        )

    now = datetime.now(timezone.utc)
    metadata = {
        "model_version": new_model_version(now),
        "code_version": __version__,
        "algorithm": "HistGradientBoostingClassifier",
        "trained_at": now.isoformat(),
        "seed": seed,
        "metrics": metrics,
        "quality_gate": quality_gate,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "mlflow_run_id": None,
    }

    store = LocalModelStore(model_dir)
    reference = x_train.sample(n=min(REFERENCE_SAMPLE_SIZE, len(x_train)), random_state=seed)
    store.save(pipeline, metadata, reference)
    logger.info("Modelo guardado en %s | metricas: %s", model_dir, metrics)

    run_id = _maybe_log_to_mlflow(pipeline, metadata, x_test.head(5))
    if run_id:
        metadata["mlflow_run_id"] = run_id
        store.save_metadata(metadata)
    return metadata


def _maybe_log_to_mlflow(pipeline, metadata: dict, input_example: pd.DataFrame) -> str | None:
    """Registro opcional en MLflow. Devuelve el run_id o None.

    El tracking es un plus de trazabilidad, no un punto unico de fallo: si MLflow no
    esta instalado o no responde, se registra el aviso y el entrenamiento (cuyos
    artefactos locales ya estan guardados) termina con exito.
    """
    settings = get_settings()
    if not settings.mlflow_tracking_uri:
        return None
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("MLflow no instalado; se omite el tracking")
        return None
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name=f"train-{metadata['model_version']}") as run:
            mlflow.log_params(
                {
                    "algorithm": metadata["algorithm"],
                    "n_train": metadata["metrics"]["n_train"],
                    "code_version": metadata["code_version"],
                    "model_version": metadata["model_version"],
                    "seed": metadata["seed"],
                    "min_roc_auc": metadata["quality_gate"]["min_roc_auc"],
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in metadata["metrics"].items() if isinstance(v, (int, float))}
            )
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=settings.mlflow_registered_model,
                input_example=input_example,
            )
            run_id = run.info.run_id
    except Exception:
        logger.exception(
            "Fallo al registrar en MLflow (%s); los artefactos locales ya estan guardados",
            settings.mlflow_tracking_uri,
        )
        return None
    logger.info("Run %s registrado en MLflow (%s)", run_id, settings.mlflow_tracking_uri)
    return run_id


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Entrena el modelo de churn")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", default=get_settings().model_dir)
    parser.add_argument(
        "--min-auc",
        type=float,
        default=None,
        help="Gate de calidad (por defecto CHURN_MIN_ROC_AUC)",
    )
    args = parser.parse_args(argv)

    df = generate_dataset(n_rows=args.rows, seed=args.seed)
    try:
        metadata = train(df, model_dir=args.model_dir, seed=args.seed, min_roc_auc=args.min_auc)
    except ValueError as exc:  # datos invalidos o gate de calidad no superado
        logger.error("Entrenamiento rechazado: %s", exc)
        return 2
    print(
        f"Entrenamiento OK. version={metadata['model_version']} "
        f"AUC={metadata['metrics']['roc_auc']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
