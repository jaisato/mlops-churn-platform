import logging

from churn.data.generator import FEATURE_COLUMNS, generate_dataset
from tests.conftest import ADMIN_HEADERS
from tests.test_api import CUSTOMER, CUSTOMER_FIEL, _customers_from

# --------------------------------------------------------------------------- /metrics


def test_metrics_expone_formato_prometheus(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    body = resp.text
    assert "# HELP churn_http_requests_total" in body
    assert "churn_model_info{" in body and "} 1.0" in body


def test_metrics_cuentan_peticiones_por_plantilla_de_ruta(client):
    client.post("/predict", json=CUSTOMER)
    client.get("/no-existe")
    body = client.get("/metrics").text
    assert 'churn_http_requests_total{method="POST",path="/predict",status="200"} 1.0' in body
    assert 'churn_http_requests_total{method="GET",path="unmatched",status="404"} 1.0' in body
    assert "churn_http_request_duration_seconds_bucket" in body


def test_metrics_de_predicciones_y_probabilidades(client):
    version = client.post("/predict", json=CUSTOMER).json()["model_version"]
    client.post("/predict/batch", json={"customers": [CUSTOMER_FIEL, CUSTOMER_FIEL]})
    body = client.get("/metrics").text
    assert f'churn_predictions_total{{model_version="{version}",risk_level="bajo"}} 2.0' in body
    assert "churn_prediction_probability_count 3.0" in body


def test_metrics_de_drift_se_actualizan_con_cada_informe(client_factory):
    c, _ = client_factory(seed=101, drift_min_rows=50)
    df = generate_dataset(200, seed=102)
    df["monthly_charges"] = (df["monthly_charges"] * 2 + 40).round(2)
    c.post("/predict/batch", json={"customers": _customers_from(df)})
    assert c.get("/monitoring/drift").json()["drift_detected"] is True

    body = c.get("/metrics").text
    assert "churn_drift_detected 1.0" in body
    assert 'churn_drift_psi{feature="monthly_charges"}' in body
    assert all(f'churn_drift_psi{{feature="{f}"}}' in body for f in FEATURE_COLUMNS)
    assert "churn_prediction_drift_psi " in body


def test_metrics_model_info_cambia_con_el_reload(client_factory, train_small):
    c, model_dir = client_factory(seed=103)
    v1 = c.get("/health").json()["model_version"]
    train_small(model_dir, seed=104)
    v2 = c.post("/model/reload", headers=ADMIN_HEADERS).json()["model_version"]
    body = c.get("/metrics").text
    assert f'churn_model_info{{model_version="{v2}"}} 1.0' in body
    assert f'churn_model_info{{model_version="{v1}"}}' not in body  # la serie anterior se retira


def test_metrics_desactivables(client_factory):
    c, _ = client_factory(seed=105, metrics_enabled=False)
    assert c.get("/metrics").status_code == 404
    assert c.get("/health").status_code == 200


# --------------------------------------------------------------------------- request id y logs


def test_request_id_se_propaga_o_se_genera(client):
    propagado = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert propagado.headers["X-Request-ID"] == "abc-123"
    generado = client.get("/health")
    assert len(generado.headers["X-Request-ID"]) == 16


def test_access_log_es_una_linea_por_peticion_con_contexto(client, caplog):
    with caplog.at_level(logging.INFO, logger="churn.access"):
        client.post("/predict", json=CUSTOMER, headers={"X-Request-ID": "req-42"})
        client.get("/metrics")
    records = [r for r in caplog.records if r.name == "churn.access"]
    assert len(records) == 1  # /metrics no se registra
    record = records[0]
    assert record.request_id == "req-42"
    assert record.http["method"] == "POST"
    assert record.http["route"] == "/predict"
    assert record.http["status"] == 200
    assert record.http["duration_ms"] >= 0


def test_access_log_desactivable(client_factory, caplog):
    c, _ = client_factory(seed=106, access_log=False)
    with caplog.at_level(logging.INFO, logger="churn.access"):
        c.get("/health")
    assert not [r for r in caplog.records if r.name == "churn.access"]
    assert "X-Request-ID" in c.get("/health").headers


def test_error_no_controlado_se_registra_como_500(client, caplog, monkeypatch):
    def explota(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(client.app.state.model.pipeline, "predict_proba", explota)
    with caplog.at_level(logging.INFO, logger="churn.access"):
        try:
            client.post("/predict", json=CUSTOMER)
        except RuntimeError:
            pass  # el TestClient por defecto relanza las excepciones del servidor
    records = [r for r in caplog.records if r.name == "churn.access"]
    assert records and records[-1].http["status"] == 500
    assert 'status="500"' in client.get("/metrics").text


# --------------------------------------------------------------------------- API key


def test_api_key_protege_scoring_info_y_drift(client_factory):
    c, _ = client_factory(seed=107, api_key="clave-secreta")
    assert c.post("/predict", json=CUSTOMER).status_code == 401
    assert c.post("/predict", json=CUSTOMER, headers={"X-API-Key": "otra"}).status_code == 401
    assert c.post("/predict/batch", json={"customers": [CUSTOMER]}).status_code == 401
    assert c.get("/model/info").status_code == 401
    assert c.get("/model/versions").status_code == 401
    assert c.get("/monitoring/drift").status_code == 401

    ok = {"X-API-Key": "clave-secreta"}
    assert c.post("/predict", json=CUSTOMER, headers=ok).status_code == 200
    assert c.get("/model/info", headers=ok).status_code == 200
    assert c.get("/model/versions", headers=ok).status_code == 200
    assert c.get("/monitoring/drift", headers=ok).status_code == 409  # autenticado, sin datos


def test_api_key_no_afecta_a_salud_metricas_ni_admin(client_factory):
    c, _ = client_factory(seed=108, api_key="clave-secreta")
    assert c.get("/health").status_code == 200
    assert c.get("/health/live").status_code == 200
    assert c.get("/metrics").status_code == 200
    assert c.post("/model/reload", headers=ADMIN_HEADERS).status_code == 200


def test_sin_api_key_configurada_los_endpoints_quedan_abiertos(client):
    assert client.post("/predict", json=CUSTOMER).status_code == 200
    assert client.get("/model/info").status_code == 200
