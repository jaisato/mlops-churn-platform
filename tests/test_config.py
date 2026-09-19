import pytest
from pydantic import ValidationError

from churn.config import Settings, get_settings

SECURE_TOKEN = "a" * 32


def test_valores_por_defecto():
    s = Settings(_env_file=None)
    assert s.model_dir == "models"
    assert s.mlflow_tracking_uri == ""
    assert (s.risk_medium, s.risk_high) == (0.35, 0.65)
    assert (s.drift_min_rows, s.drift_buffer_size, s.psi_alert_threshold) == (200, 5000, 0.2)
    assert s.min_roc_auc == 0.75
    assert s.is_production is False


def test_lee_variables_con_prefijo_churn(monkeypatch):
    monkeypatch.setenv("CHURN_RISK_HIGH", "0.9")
    monkeypatch.setenv("CHURN_DRIFT_MIN_ROWS", "20")
    monkeypatch.setenv("CHURN_DRIFT_BUFFER_SIZE", "50")
    monkeypatch.setenv("CHURN_MIN_ROC_AUC", "0.8")
    s = Settings(_env_file=None)
    assert s.risk_high == 0.9
    assert (s.drift_min_rows, s.drift_buffer_size) == (20, 50)
    assert s.min_roc_auc == 0.8


def test_variables_sin_prefijo_se_ignoran(monkeypatch):
    monkeypatch.setenv("RISK_HIGH", "0.9")
    assert Settings(_env_file=None).risk_high == 0.65


def test_umbrales_de_riesgo_deben_estar_ordenados():
    with pytest.raises(ValidationError, match="risk_medium"):
        Settings(_env_file=None, risk_medium=0.7, risk_high=0.6)
    with pytest.raises(ValidationError, match="risk_medium"):
        Settings(_env_file=None, risk_medium=0.5, risk_high=0.5)


def test_umbrales_fuera_de_0_1_rechazados():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, risk_high=1.5)


def test_drift_min_rows_no_puede_superar_el_buffer():
    with pytest.raises(ValidationError, match="drift_min_rows"):
        Settings(_env_file=None, drift_min_rows=100, drift_buffer_size=50)


def test_drift_min_rows_minimo_2():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, drift_min_rows=1)


@pytest.mark.parametrize("env", ["production", "prod", "PRODUCTION"])
@pytest.mark.parametrize("token", ["", "cambia-este-token", "token-local", "corto"])
def test_produccion_rechaza_tokens_inseguros(env, token):
    with pytest.raises(ValidationError, match="CHURN_ADMIN_TOKEN inseguro"):
        Settings(_env_file=None, environment=env, admin_token=token)


def test_produccion_acepta_token_seguro():
    s = Settings(_env_file=None, environment="production", admin_token=SECURE_TOKEN)
    assert s.is_production and s.admin_token_is_secure


@pytest.mark.parametrize("env", ["dev", "local", "test"])
def test_entornos_no_productivos_toleran_el_token_por_defecto(env):
    s = Settings(_env_file=None, environment=env)
    assert not s.admin_token_is_secure  # pero no bloquea el arranque


def test_get_settings_esta_cacheado():
    assert get_settings() is get_settings()
