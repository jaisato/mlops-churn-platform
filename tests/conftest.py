import contextlib

import pytest
from fastapi.testclient import TestClient

from churn.config import Settings
from churn.data.generator import generate_dataset
from churn.serving.main import create_app
from churn.training.train import train

SMALL_ROWS = 1200  # suficiente para superar la validacion (>= 500) y el gate de calidad
ADMIN_HEADERS = {"X-Admin-Token": "token-test"}


def make_settings(model_dir: str, **overrides) -> Settings:
    """Settings hermeticos para tests: sin leer .env ni depender del entorno del runner."""
    base = dict(
        environment="test", model_dir=model_dir, drift_min_rows=10, admin_token="token-test"
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


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
    """Entrena un modelo pequeno (rapido) en el directorio indicado y devuelve su metadata."""

    def _train(model_dir, seed: int = 21) -> dict:
        df = generate_dataset(n_rows=SMALL_ROWS, seed=seed)
        return train(df, model_dir=str(model_dir), seed=seed)

    return _train


@pytest.fixture()
def client_factory(tmp_path, train_small):
    """Clientes con modelo y directorio propios (reload, artefactos corruptos, buffer...).

    Devuelve `(client, model_dir)`. Los clientes se cierran al terminar el test.
    """
    stack = contextlib.ExitStack()

    def _make(seed: int = 21, model_dir=None, train_model: bool = True, **overrides):
        model_dir = str(model_dir or tmp_path / f"model-{seed}")
        if train_model:
            train_small(model_dir, seed=seed)
        options = {"drift_min_rows": 5, **overrides}
        app = create_app(make_settings(model_dir, **options))
        client = stack.enter_context(TestClient(app, raise_server_exceptions=False))
        return client, model_dir

    yield _make
    stack.close()
