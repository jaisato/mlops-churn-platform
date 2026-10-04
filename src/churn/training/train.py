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
Holdout: 20 % de los GRUPOS, elegidos por un hash estable (sal fija) de `subject_ref` (todas
las predicciones del mismo cliente juntas) o, sin el, del contenido de la fila (filas
identicas juntas). No depende de la semilla ni del tamano del dataset: entre reentrenos con
un dataset acumulado, una fila nunca cambia de lado. El campeon se evalua ademas solo con las
filas predichas despues de las etiquetas con las que se entreno (`predicted_to`).
Codigos de salida: 0 promovido | 2 datos invalidos o gate no superado | 3 entrenado
pero no promovido (el campeon sigue en servicio).
"""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn import __version__
from churn.config import Settings, get_settings
from churn.data.generator import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET
from churn.data.sources import dataset_fingerprint, resolve_training_data
from churn.data.validation import validate_training_data
from churn.logging_conf import configure_logging
from churn.monitoring.performance import DECISION_THRESHOLD, classification_metrics
from churn.monitoring.store import PREDICTED_AT_COLUMN, SUBJECT_COLUMN
from churn.registry import LoadedModel, LocalModelStore

logger = logging.getLogger(__name__)

REFERENCE_SAMPLE_SIZE = 2000
EXIT_OK = 0
EXIT_REJECTED = 2
EXIT_NOT_PROMOTED = 3

TEST_SIZE = 0.2
SPLIT_METHOD = "group_hash"
#: Sal del hash del holdout. Cambiarla reparte de nuevo todas las filas (y mezclaria el
#: holdout de un reentreno con lo que el campeon vio al entrenar): solo con un buen motivo.
SPLIT_SALT = "churn-holdout-v1"
#: Grupos independientes (sujetos, o filas sin sujeto) minimos, con ambas clases, para decidir
#: entre campeon y retador: 240 filas de 60 clientes son 60 observaciones, no 240. Es el
#: holdout esperado del dataset minimo (500 x 20 % = 100) con holgura para su variacion.
MIN_COMPARISON_GROUPS = 80


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


def split_groups(df: pd.DataFrame) -> pd.Series:
    """Grupo de cada fila para separar el holdout.

    `subject_ref` si viene (el mismo cliente puntuado varias veces queda entero de un lado) y,
    si no, el contenido de las features (las filas identicas quedan juntas).
    """
    content = pd.util.hash_pandas_object(df[FEATURE_COLUMNS], index=False)
    groups = "f:" + content.astype(str)
    if SUBJECT_COLUMN in df.columns:
        subjects = df[SUBJECT_COLUMN]
        has_subject = subjects.notna() & (subjects.astype(str).str.len() > 0)
        groups = groups.where(~has_subject, "s:" + subjects.astype(str))
    return groups


def holdout_mask(groups: pd.Series, test_size: float = TEST_SIZE) -> np.ndarray:
    """True en las filas del holdout: hash del grupo (con sal fija) por debajo de `test_size`.

    Determinista y estable: no depende de la semilla, del orden ni del tamano del dataset,
    asi que al acumular etiquetas una fila que fue de entrenamiento nunca pasa al holdout.
    """
    buckets = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(f"{SPLIT_SALT}:{group}".encode(), digest_size=8).digest(), "big"
            )
            for group in groups
        ),
        dtype=np.uint64,
        count=len(groups),
    )
    return buckets < np.uint64(int(test_size * 2**64))


def _auc(y_true: Any, proba: Any) -> float:
    return round(float(roc_auc_score(y_true, proba)), 4)


def _utc_times(values: Any) -> pd.Series:
    """Instantes ISO-8601 en UTC; lo ilegible queda como NaT (nunca cuenta como posterior)."""
    return pd.Series(pd.to_datetime(values, utc=True, errors="coerce", format="ISO8601"))


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
    challenger_proba: np.ndarray,
    margin: float,
    *,
    fingerprint: str,
    groups: pd.Series,
    predicted_at: pd.Series | None = None,
) -> dict[str, Any]:
    """Decide si el retador sustituye al campeon (`current`) sobre las MISMAS filas.

    Las filas salen del holdout de los datos nuevos: si el mundo ha cambiado, el campeon
    (entrenado en el mundo anterior) rinde peor ahi y el retador gana; si los datos nuevos no
    aportan nada, el campeon se mantiene. Si el campeon se entreno con etiquetas de la API
    (`data_source.labels.predicted_to`) y el holdout trae `predicted_at`, solo cuentan las
    filas predichas DESPUES: con un dataset acumulado, el resto puede incluir ejemplos con los
    que el campeon entreno (y que puntuaria con ventaja). Sin al menos `MIN_COMPARISON_GROUPS`
    grupos (sujetos) con ambas clases no hay evidencia para sustituirlo. Con los mismos datos que el
    campeon (misma huella) no se promueve: el retador no puede corregir nada. Devuelve el
    bloque `promotion` del metadata.
    """
    decision: dict[str, Any] = {
        "decision": "no_champion",
        "margin": margin,
        "challenger_roc_auc": _auc(y_test, challenger_proba),
        "champion_version": None,
        "champion_roc_auc": None,
        "identical_training_data": False,
        "evaluated_rows": int(len(y_test)),
        "evaluated_groups": int(groups.nunique()),
        "champion_cutoff": None,
        "reason": "",
    }
    champion, why_not = _load_champion(store)
    if champion is None:
        decision["reason"] = f"se promueve: {why_not}"
        return decision

    decision["champion_version"] = champion.version
    champion_source = champion.metadata.get("data_source") or {}
    if champion_source.get("fingerprint") == fingerprint:
        decision["identical_training_data"] = True
        decision["decision"] = "rejected"
        decision["reason"] = (
            f"mismos datos (huella {fingerprint[:12]}) que el campeon {champion.version}: "
            "reentrenar con ellos no corrige nada; se mantiene el campeon"
        )
        logger.warning(
            "Mismos datos (huella %s) que el campeon %s: el retador no aporta nada; no se promueve",
            fingerprint[:12],
            champion.version,
        )
        return decision

    fresh = np.ones(len(y_test), dtype=bool)
    cutoff = (champion_source.get("labels") or {}).get("predicted_to")
    if cutoff and predicted_at is None:
        logger.warning(
            "El campeon %s se entreno con etiquetas hasta %s, pero estos datos no traen "
            "predicted_at: se le evalua con todo el holdout",
            champion.version,
            cutoff,
        )
    elif cutoff:
        decision["champion_cutoff"] = cutoff
        fresh = (_utc_times(predicted_at) > _utc_times([cutoff])[0]).to_numpy()
    y_eval = y_test[fresh]
    n_groups = int(groups[fresh].nunique())
    decision["evaluated_rows"] = int(fresh.sum())
    decision["evaluated_groups"] = n_groups
    if n_groups < MIN_COMPARISON_GROUPS or y_eval.nunique() < 2:
        scope = f"posteriores a las etiquetas del campeon (hasta {cutoff})" if cutoff else "utiles"
        evidence = f"solo {n_groups} grupos ({len(y_eval)} filas) del holdout {scope}"
        if margin >= 1.0:  # margen 1 = promover siempre que pase el gate
            decision["decision"] = "promoted"
            decision["reason"] = f"{evidence}, pero el margen {margin} promueve sin comparar"
            return decision
        decision["decision"] = "rejected"
        decision["reason"] = (
            f"{evidence} (minimo {MIN_COMPARISON_GROUPS} grupos con ambas clases): no hay "
            "evidencia suficiente para sustituir al campeon; se mantiene el campeon"
        )
        return decision

    challenger_roc_auc = _auc(y_eval, challenger_proba[fresh])
    decision["challenger_roc_auc"] = challenger_roc_auc
    try:
        proba = champion.pipeline.predict_proba(x_test[fresh][champion.feature_columns])[:, 1]
        champion_auc = _auc(y_eval, proba)
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

    # Holdout por grupos y estable entre reentrenos (ver `holdout_mask`); las columnas de
    # trazabilidad del dataset de etiquetas (subject_ref, predicted_at...) no son features.
    groups = split_groups(df)
    in_test = holdout_mask(groups)
    x_train, y_train = df.loc[~in_test, FEATURE_COLUMNS], df.loc[~in_test, TARGET]
    x_test, y_test = df.loc[in_test, FEATURE_COLUMNS], df.loc[in_test, TARGET]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            f"El holdout ({len(y_test)} filas) o el entrenamiento ({len(y_train)} filas) no "
            "tiene ambas clases: hacen falta mas datos etiquetados"
        )
    split = {
        "method": SPLIT_METHOD,
        "test_size": TEST_SIZE,
        "groups_train": int(groups[~in_test].nunique()),
        "groups_test": int(groups[in_test].nunique()),
        "subject_rows": int(groups.str.startswith("s:").sum()),
    }

    pipeline = build_pipeline(seed=seed)
    pipeline.fit(x_train, y_train)

    proba = pipeline.predict_proba(x_test)[:, 1]
    # Las mismas metricas (y el mismo umbral) que /monitoring/performance mide en produccion
    # con las etiquetas reales: holdout y servicio son comparables cifra a cifra.
    holdout = classification_metrics(y_test, proba, DECISION_THRESHOLD)
    roc_auc = float(holdout["roc_auc"] or 0.0)  # ambas clases garantizadas arriba
    metrics: dict[str, Any] = {
        **holdout,
        "roc_auc": roc_auc,
        "churn_rate_train": round(float(y_train.mean()), 4),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }

    quality_gate = {"min_roc_auc": min_roc_auc, "passed": roc_auc >= min_roc_auc}
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
        proba,
        promotion_margin,
        fingerprint=source["fingerprint"],
        groups=groups[in_test],
        predicted_at=(
            df.loc[in_test, PREDICTED_AT_COLUMN] if PREDICTED_AT_COLUMN in df.columns else None
        ),
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
        "split": split,
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
