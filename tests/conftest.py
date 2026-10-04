import contextlib
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn.config import Settings
from churn.data.generator import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    generate_dataset,
)
from churn.monitoring.store import PredictionStore
from churn.serving.main import create_app
from churn.training.train import train

SMALL_ROWS = 1200  # suficiente para superar la validacion (>= 500) y el gate de calidad
ADMIN_HEADERS = {"X-Admin-Token": "token-test"}
STORE_CLOCK_START = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)


class TickingClock:
    """Reloj del almacen de predicciones en los tests: cada lectura avanza `step`.

    Las predicciones (y `labelled_at`) quedan fechadas en septiembre de 2026, en orden
    estricto, asi que una etiqueta con una fecha fija de octubre siempre es posterior a su
    prediccion (el almacen rechaza las anteriores) y los tests no dependen del reloj real.
    """

    def __init__(self, start: datetime = STORE_CLOCK_START, step: timedelta = timedelta(seconds=1)):
        self.now = start
        self.step = step

    def __call__(self) -> datetime:
        self.now += self.step
        return self.now


@pytest.fixture(autouse=True)
def store_clock(monkeypatch) -> TickingClock:
    clock = TickingClock()
    monkeypatch.setattr("churn.monitoring.store._utcnow", clock)
    return clock


def score_and_label(
    store: PredictionStore,
    df: pd.DataFrame,
    *,
    version: str = "v1",
    observed_at: datetime | str | None = None,
) -> list[str]:
    """Simula el bucle completo sobre un almacen: puntua las filas de `df` (con una
    probabilidad ficticia correlada con la etiqueta) y devuelve el `churn` real de cada
    una. Devuelve los `prediction_id` en el orden de `df`."""
    rows = df[FEATURE_COLUMNS].to_dict("records")
    probas = [0.8 if churn else 0.2 for churn in df[TARGET]]
    ids = store.append(
        {**row, "churn_probability": p, "model_version": version}
        for row, p in zip(rows, probas, strict=True)
    )
    store.label(
        {
            "prediction_id": prediction_id,
            "churn": int(churn),
            "observed_at": observed_at or datetime.now(UTC),
        }
        for prediction_id, churn in zip(ids, df[TARGET], strict=True)
    )
    return ids


def make_settings(model_dir: str, **overrides) -> Settings:
    """Settings hermeticos para tests: sin leer .env ni depender del entorno del runner.

    El almacen de predicciones por defecto es en memoria para que los clientes que
    comparten el modelo de sesion no se vean las predicciones unos a otros.
    """
    base = dict(
        environment="test",
        model_dir=model_dir,
        drift_min_rows=10,
        drift_store="memory",
        admin_token="token-test",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def weak_pipeline(seed: int = 42) -> Pipeline:
    """Sustituto de `build_pipeline`: un unico stump. Pasa un gate laxo (`min_roc_auc=0.5`)
    pero pierde con claridad frente a un campeon normal: sirve para forzar la decision
    "rejected" de la promocion de forma determinista."""
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = HistGradientBoostingClassifier(max_iter=1, max_depth=1, random_state=seed)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory) -> str:
    """Entrena una vez un modelo pequeno y reutilizalo en toda la sesion de tests."""
    model_dir = tmp_path_factory.mktemp("models")
    df = generate_dataset(n_rows=3000, seed=7)
    train(df, model_dir=str(model_dir), seed=7)
    return str(model_dir)


@pytest.fixture()
def client(trained_model_dir):
    app = create_app(make_settings(trained_model_dir))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client_sin_modelo(tmp_path):
    app = create_app(make_settings(str(tmp_path / "vacio")))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def train_small():
    """Entrena un modelo pequeno (rapido) en el directorio indicado y devuelve su metadata.

    Publica SIEMPRE (`promotion_margin=1.0` desactiva la comparacion con el campeon): con
    1200 filas el AUC del holdout es ruidoso y estos tests ejercitan el almacen y la API,
    no la promocion. La promocion tiene sus propios tests en `test_retraining.py`.
    """

    def _train(model_dir, seed: int = 21, **overrides) -> dict:
        df = generate_dataset(n_rows=SMALL_ROWS, seed=seed)
        options = {"promotion_margin": 1.0, **overrides}
        return train(df, model_dir=str(model_dir), seed=seed, **options)

    return _train


@pytest.fixture()
def client_factory(tmp_path, train_small):
    """Clientes con modelo y directorio propios (reload, rollback, artefactos corruptos...).

    Usan el almacen SQLite real dentro de su directorio. Devuelve `(client, model_dir)`;
    los clientes se cierran al terminar el test.
    """
    stack = contextlib.ExitStack()

    def _make(seed: int = 21, model_dir=None, train_model: bool = True, **overrides):
        model_dir = str(model_dir or tmp_path / f"model-{seed}")
        if train_model:
            train_small(model_dir, seed=seed)
        options = {"drift_min_rows": 5, "drift_store": "sqlite", **overrides}
        app = create_app(make_settings(model_dir, **options))
        client = stack.enter_context(TestClient(app, raise_server_exceptions=False))
        return client, model_dir

    yield _make
    stack.close()
