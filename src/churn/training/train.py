"""Entrenamiento del modelo de churn con tracking opcional en MLflow.

Uso:
    python -m churn.training.train --rows 20000 --model-dir models [--min-auc 0.8]
    python -m churn.training.train --data data/churn_2026q3.parquet --model-dir models
    python -m churn.training.train --rows 20000 --drift-shift 1.0   (mundo sintetico desplazado)
Variables:
    CHURN_MLFLOW_TRACKING_URI  si esta definida, se registran parametros,
    metricas, artefactos y version del modelo en MLflow Model Registry, y la
    version registrada recibe el alias CHURN_MLFLOW_CHAMPION_ALIAS si se promociona.
    CHURN_MIN_ROC_AUC          gate de calidad: por debajo de este AUC el
    entrenamiento falla y NO sobreescribe los artefactos del modelo en servicio.
    CHURN_PROMOTION_MARGIN     campeon/retador: el modelo nuevo solo pasa a `current`
    si su AUC sobre el holdout de los datos nuevos es >= AUC del campeon - margen.
Codigos de salida: 0 promovido | 2 datos invalidos o gate no superado | 3 entrenado
pero no promovido (el campeon sigue en servicio).
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
from churn.data.generator import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET
from churn.data.sources import dataset_fingerprint, resolve_training_data
from churn.data.validation import validate_training_data
from churn.logging_conf import configure_logging
from churn.registry import LoadedModel, LocalModelStore

logger = logging.getLogger(__name__)

REFERENCE_SAMPLE_SIZE = 2000
DECISION_THRESHOLD = 0.5
EXIT_OK = 0
EXIT_REJECTED = 2
EXIT_NOT_PROMOTED = 3


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


def _load_champion(store: LocalModelStore) -> tuple[LoadedModel | None, str | None]:
    """Modelo en servicio, o `(None, motivo)` si no existe o no se puede evaluar."""
    if store.resolve_dir() is None:
        return None, "no hay modelo en servicio"
    try:
        return store.load(), None
    except Exception as exc:  # incompleto, corrupto o de otro runtime: no comparable
        return None, f"el modelo en servicio no se puede cargar ({exc})"


def compare_with_champion(
    store: LocalModelStore,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    challenger_roc_auc: float,
    margin: float,
    *,
    fingerprint: str,
    seed: int,
) -> dict[str, Any]:
    """Decide si el retador sustituye al campeon (`current`) sobre el MISMO holdout.

    El holdout sale de los datos nuevos: si el mundo ha cambiado, el campeon (entrenado en
    el mundo anterior) rinde peor ahi y el retador gana; si los datos nuevos no aportan
    nada, el campeon se mantiene. Devuelve el bloque `promotion` del metadata.
    """
    decision: dict[str, Any] = {
        "decision": "no_champion",
        "margin": margin,
        "challenger_roc_auc": challenger_roc_auc,
        "champion_version": None,
        "champion_roc_auc": None,
        "identical_training_data": False,
        "reason": "",
    }
    champion, why_not = _load_champion(store)
    if champion is None:
        decision["reason"] = f"se promueve: {why_not}"
        return decision

    decision["champion_version"] = champion.version
    champion_source = champion.metadata.get("data_source") or {}
    if champion_source.get("fingerprint") == fingerprint and champion.metadata.get("seed") == seed:
        decision["identical_training_data"] = True
        logger.warning(
            "Mismos datos (huella %s) y semilla que el campeon %s: el retador es identico; "
            "reentrenar asi no corrige ningun drift",
            fingerprint[:12],
            champion.version,
        )
    try:
        proba = champion.pipeline.predict_proba(x_test[champion.feature_columns])[:, 1]
        champion_auc = round(float(roc_auc_score(y_test, proba)), 4)
    except Exception as exc:  # features distintas, pickle de otra version...
        decision["reason"] = f"se promueve: el campeon no se puede evaluar ({exc})"
        return decision

    decision["champion_roc_auc"] = champion_auc
    if challenger_roc_auc >= champion_auc - margin:
        decision["decision"] = "promoted"
        decision["reason"] = (
            f"AUC retador {challenger_roc_auc} >= AUC campeon {champion_auc} - margen {margin}"
        )
    else:
        decision["decision"] = "rejected"
        decision["reason"] = (
            f"AUC retador {challenger_roc_auc} < AUC campeon {champion_auc} - margen {margin}: "
            "se mantiene el campeon"
        )
    return decision


def train(
    df: pd.DataFrame,
    model_dir: str,
    seed: int = 42,
    min_roc_auc: float | None = None,
    promotion_margin: float | None = None,
    data_source: dict[str, Any] | None = None,
) -> dict:
    """Entrena, evalua, aplica el gate de calidad, compara con el campeon y persiste.

    Devuelve el metadata generado; `metadata["promotion"]["decision"]` dice si la version
    nueva ha pasado a `current` (`promoted`/`no_champion`) o se ha guardado sin promover
    (`rejected`). Lanza `ValueError` si los datos no son validos y `ModelQualityError` si el
    AUC no alcanza `min_roc_auc` (por defecto `CHURN_MIN_ROC_AUC`); en ambos casos no se
    toca el directorio del modelo. `data_source` describe de donde salen las filas (lo
    rellena la CLI); la huella y el numero de filas se calculan siempre aqui.
    """
    settings = get_settings()
    if min_roc_auc is None:
        min_roc_auc = settings.min_roc_auc
    if promotion_margin is None:
        promotion_margin = settings.promotion_margin

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

    source = {"kind": "dataframe", **(data_source or {})}
    source["rows"] = int(len(df))
    source["fingerprint"] = dataset_fingerprint(df)

    store = LocalModelStore(model_dir, keep_versions=settings.model_keep_versions)
    promotion = compare_with_champion(
        store,
        x_test,
        y_test,
        metrics["roc_auc"],
        promotion_margin,
        fingerprint=source["fingerprint"],
        seed=seed,
    )
    promoted = promotion["decision"] != "rejected"

    now = datetime.now(UTC)
    metadata = {
        "model_version": new_model_version(now),
        "code_version": __version__,
        "algorithm": "HistGradientBoostingClassifier",
        "trained_at": now.isoformat(),
        "seed": seed,
        "metrics": metrics,
        "quality_gate": quality_gate,
        "promotion": promotion,
        "data_source": source,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "runtime": runtime_versions(),
        "mlflow_run_id": None,
        "mlflow_model_version": None,
    }

    # La referencia de drift sale de los datos con los que se ha entrenado ESTA version
    reference = x_train.sample(n=min(REFERENCE_SAMPLE_SIZE, len(x_train)), random_state=seed)
    store.save(pipeline, metadata, reference, promote=promoted)
    if promoted:
        logger.info(
            "Version %s publicada en %s | metricas: %s | %s",
            metadata["model_version"],
            model_dir,
            metrics,
            promotion["reason"],
        )
    else:
        logger.warning(
            "Version %s guardada SIN promover en %s | metricas: %s | %s",
            metadata["model_version"],
            model_dir,
            metrics,
            promotion["reason"],
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
    artefactos locales ya estan guardados) termina con exito. El alias de campeon solo
    se asigna a las versiones promovidas; las rechazadas quedan registradas sin alias.
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
    promotion = metadata["promotion"]
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
                    "data_kind": metadata["data_source"]["kind"],
                    "data_fingerprint": metadata["data_source"]["fingerprint"],
                    "promotion_decision": promotion["decision"],
                    "promotion_margin": promotion["margin"],
                    "champion_version": promotion["champion_version"],
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in metadata["metrics"].items() if isinstance(v, (int, float))}
            )
            if promotion["champion_roc_auc"] is not None:
                mlflow.log_metrics({"champion_roc_auc": promotion["champion_roc_auc"]})
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
    if promotion["decision"] == "rejected":
        logger.info("Version no promovida: el alias de campeon no cambia")
        return {"mlflow_run_id": run_id, "mlflow_model_version": None}
    registry_version = _promote_champion(mlflow, settings, run_id, model_info)
    return {"mlflow_run_id": run_id, "mlflow_model_version": registry_version}


def _promote_champion(
    mlflow_module: Any, settings: Settings, run_id: str, model_info: Any
) -> str | None:
    """Asigna el alias de campeon a la version registrada del run. Devuelve la version o None.

    Solo llegan aqui modelos que han superado el gate Y la comparacion con el campeon, asi
    que el alias siempre apunta al modelo en servicio: el registry deja de ser un log de
    escritura y pasa a ser la fuente de "que modelo esta en produccion".
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
    parser.add_argument(
        "--data",
        default=None,
        help="Fichero etiquetado (.csv/.parquet) con las features y la columna churn. "
        "Sin el, se usa el generador sintetico (--rows, --seed, --drift-shift).",
    )
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drift-shift",
        type=float,
        default=0.0,
        help="Solo sintetico: desplaza el mundo generado (0 = ninguno, 1 = fuerte, max 2) "
        "para simular drift en demos y tests",
    )
    parser.add_argument("--model-dir", default=get_settings().model_dir)
    parser.add_argument(
        "--min-auc",
        type=float,
        default=None,
        help="Gate de calidad (por defecto CHURN_MIN_ROC_AUC)",
    )
    parser.add_argument(
        "--promotion-margin",
        type=float,
        default=None,
        help="Margen de AUC frente al campeon (por defecto CHURN_PROMOTION_MARGIN; 1 = "
        "promover siempre que pase el gate)",
    )
    args = parser.parse_args(argv)

    try:
        df, source = resolve_training_data(
            data_path=args.data, rows=args.rows, seed=args.seed, drift_shift=args.drift_shift
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Datos de entrenamiento no disponibles: %s", exc)
        return EXIT_REJECTED
    logger.info("Datos de entrenamiento: %s", source)
    try:
        metadata = train(
            df,
            model_dir=args.model_dir,
            seed=args.seed,
            min_roc_auc=args.min_auc,
            promotion_margin=args.promotion_margin,
            data_source=source,
        )
    except ValueError as exc:  # datos invalidos o gate de calidad no superado
        logger.error("Entrenamiento rechazado: %s", exc)
        return EXIT_REJECTED
    promotion = metadata["promotion"]
    if promotion["decision"] == "rejected":
        print(
            f"Entrenamiento OK pero NO promovido. version={metadata['model_version']} "
            f"AUC={metadata['metrics']['roc_auc']} campeon={promotion['champion_version']} "
            f"AUC_campeon={promotion['champion_roc_auc']}"
        )
        return EXIT_NOT_PROMOTED
    print(
        f"Entrenamiento OK. version={metadata['model_version']} "
        f"AUC={metadata['metrics']['roc_auc']} ({promotion['decision']})"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
