"""Entrenamiento del modelo de churn con tracking opcional en MLflow.

Uso:
    python -m churn.training.train --rows 20000 --model-dir models [--min-auc 0.8]
Variables:
    CHURN_MLFLOW_TRACKING_URI  si esta definida, se registran parametros,
    metricas, artefactos y version del modelo en MLflow Model Registry, y la
    version registrada recibe el alias CHURN_MLFLOW_CHAMPION_ALIAS.
    CHURN_MIN_ROC_AUC          gate de calidad: por debajo de este AUC el
    entrenamiento falla y NO sobreescribe los artefactos del modelo en servicio.
"""

from __future__ import annotations

import argparse
import logging
import platform
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn import __version__
from churn.config import Settings, get_settings
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
    now = now or datetime.now(UTC)
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"


def runtime_versions() -> dict[str, str]:
    """Versiones con las que se serializo el pipeline: la API las compara antes de servir."""
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def train(
    df: pd.DataFrame, model_dir: str, seed: int = 42, min_roc_auc: float | None = None
) -> dict:
    """Entrena, evalua, aplica el gate de calidad y persiste artefactos.

    Devuelve el metadata generado. Lanza `ValueError` si los datos no son validos y
    `ModelQualityError` si el AUC no alcanza `min_roc_auc` (por defecto
    `CHURN_MIN_ROC_AUC`); en ambos casos no se toca el directorio del modelo.
    """
    settings = get_settings()
    if min_roc_auc is None:
        min_roc_auc = settings.min_roc_auc

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

    now = datetime.now(UTC)
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
        "runtime": runtime_versions(),
        "mlflow_run_id": None,
        "mlflow_model_version": None,
    }

    store = LocalModelStore(model_dir, keep_versions=settings.model_keep_versions)
    reference = x_train.sample(n=min(REFERENCE_SAMPLE_SIZE, len(x_train)), random_state=seed)
    store.save(pipeline, metadata, reference)
    logger.info(
        "Version %s publicada en %s | metricas: %s", metadata["model_version"], model_dir, metrics
    )

    tracking = _maybe_log_to_mlflow(pipeline, metadata, x_test.head(5))
    if tracking:
        metadata.update(tracking)
        store.save_metadata(metadata)
    return metadata


def _maybe_log_to_mlflow(
    pipeline, metadata: dict, input_example: pd.DataFrame
) -> dict[str, str | None] | None:
    """Registro opcional en MLflow. Devuelve `{mlflow_run_id, mlflow_model_version}` o None.

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
                    "scikit_learn": metadata["runtime"]["scikit_learn"],
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in metadata["metrics"].items() if isinstance(v, (int, float))}
            )
            model_info = mlflow.sklearn.log_model(
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
    registry_version = _promote_champion(mlflow, settings, run_id, model_info)
    return {"mlflow_run_id": run_id, "mlflow_model_version": registry_version}


def _promote_champion(
    mlflow_module: Any, settings: Settings, run_id: str, model_info: Any
) -> str | None:
    """Asigna el alias de campeon a la version registrada del run. Devuelve la version o None.

    Solo llegan aqui modelos que han superado el gate, asi que el alias siempre apunta al
    ultimo modelo publicable: el registry deja de ser un log de escritura y pasa a ser la
    fuente de "que modelo esta en produccion".
    """
    alias = settings.mlflow_champion_alias
    if not alias:
        return None
    name = settings.mlflow_registered_model
    try:
        client = mlflow_module.MlflowClient()
        version = getattr(model_info, "registered_model_version", None)
        if version is None:
            found = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
            version = found[0].version if found else None
        if version is None:
            logger.warning("No se encontro la version registrada del run %s; sin alias", run_id)
            return None
        client.set_registered_model_alias(name, alias, str(version))
    except Exception:
        logger.exception("No se pudo asignar el alias %s en el registry %s", alias, name)
        return None
    logger.info("Alias %s -> %s v%s", alias, name, version)
    return str(version)


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
