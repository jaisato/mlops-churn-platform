import json

from churn.data.generator import FEATURE_COLUMNS, generate_dataset
from churn.serving.main import risk_level
from tests.conftest import ADMIN_HEADERS

CUSTOMER = {
    "tenure_months": 3,
    "monthly_charges": 95.5,
    "total_charges": 280.0,
    "num_products": 1,
    "support_tickets_90d": 6,
    "is_fiber": 0,
    "contract_type": "mensual",
    "payment_method": "transferencia",
}

CUSTOMER_FIEL = {
    "tenure_months": 60,
    "monthly_charges": 30.0,
    "total_charges": 1800.0,
    "num_products": 4,
    "support_tickets_90d": 0,
    "is_fiber": 1,
    "contract_type": "bianual",
    "payment_method": "domiciliacion",
}


def _customers_from(df) -> list[dict]:
    """Filas del generador como payloads JSON (tipos nativos, sin numpy)."""
    return json.loads(df[FEATURE_COLUMNS].to_json(orient="records"))


# --------------------------------------------------------------------------- sistema


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_health_sin_modelo_503(client_sin_modelo):
    assert client_sin_modelo.get("/health").status_code == 503


def test_openapi_expone_todos_los_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/predict", "/predict/batch", "/model/info", "/model/reload",
            "/monitoring/drift"} <= set(paths)
    assert client.get("/docs").status_code == 200


# --------------------------------------------------------------------------- scoring


def test_predict_riesgo_coherente(client):
    alto = client.post("/predict", json=CUSTOMER).json()
    bajo = client.post("/predict", json=CUSTOMER_FIEL).json()
    assert alto["churn_probability"] > bajo["churn_probability"]
    assert 0 <= bajo["churn_probability"] <= 1
    assert alto["risk_level"] in {"medio", "alto"}
    assert bajo["risk_level"] == "bajo"


def test_predict_validacion(client):
    invalido = dict(CUSTOMER, contract_type="semanal")
    assert client.post("/predict", json=invalido).status_code == 422


def test_predict_fuera_de_rango_422(client):
    assert client.post("/predict", json=dict(CUSTOMER, tenure_months=121)).status_code == 422
    assert client.post("/predict", json=dict(CUSTOMER, num_products=0)).status_code == 422
    assert client.post("/predict", json=dict(CUSTOMER, is_fiber=2)).status_code == 422


def test_predict_campo_faltante_422(client):
    incompleto = {k: v for k, v in CUSTOMER.items() if k != "monthly_charges"}
    assert client.post("/predict", json=incompleto).status_code == 422


def test_predict_sin_modelo_503(client_sin_modelo):
    assert client_sin_modelo.post("/predict", json=CUSTOMER).status_code == 503


def test_predict_batch(client):
    resp = client.post("/predict/batch", json={"customers": [CUSTOMER, CUSTOMER_FIEL]})
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_predict_batch_consistente_con_individual(client):
    individual = [
        client.post("/predict", json=c).json()["churn_probability"]
        for c in (CUSTOMER, CUSTOMER_FIEL)
    ]
    batch = client.post("/predict/batch", json={"customers": [CUSTOMER, CUSTOMER_FIEL]}).json()
    assert [p["churn_probability"] for p in batch["predictions"]] == individual


def test_predict_batch_limites(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
    demasiados = {"customers": [CUSTOMER] * 1001}
    assert client.post("/predict/batch", json=demasiados).status_code == 422


def test_predict_alimenta_el_buffer_con_version(client):
    version = client.post("/predict", json=CUSTOMER).json()["model_version"]
    ultimo = client.app.state.buffer[-1]
    assert ultimo["model_version"] == version
    assert 0 <= ultimo["churn_probability"] <= 1
    assert all(ultimo[k] == v for k, v in CUSTOMER.items())


def test_risk_level_umbrales_inclusivos():
    assert risk_level(0.0, 0.35, 0.65) == "bajo"
    assert risk_level(0.3499, 0.35, 0.65) == "bajo"
    assert risk_level(0.35, 0.35, 0.65) == "medio"
    assert risk_level(0.6499, 0.35, 0.65) == "medio"
    assert risk_level(0.65, 0.35, 0.65) == "alto"
    assert risk_level(1.0, 0.35, 0.65) == "alto"


def test_umbrales_de_riesgo_configurables(client_factory):
    # Con risk_medium=0 todo es al menos "medio"; con risk_high=1 nada llega a "alto"
    c, _ = client_factory(seed=71, risk_medium=0.0, risk_high=1.0)
    niveles = {c.post("/predict", json=x).json()["risk_level"] for x in (CUSTOMER, CUSTOMER_FIEL)}
    assert niveles == {"medio"}


# --------------------------------------------------------------------------- modelo


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["algorithm"] == "HistGradientBoostingClassifier"
    assert body["metrics"]["roc_auc"] > 0.5


def test_model_info_incluye_trazabilidad(client):
    body = client.get("/model/info").json()
    assert body["code_version"]
    assert body["seed"] == 7
    assert body["quality_gate"]["passed"] is True
    assert body["mlflow_run_id"] is None
    assert body["metrics"]["n_train"] == int(body["metrics"]["n_train"])  # enteros intactos


def test_model_info_sin_modelo_503(client_sin_modelo):
    assert client_sin_modelo.get("/model/info").status_code == 503


def test_model_reload_requiere_token(client):
    assert client.post("/model/reload").status_code == 401
    resp = client.post("/model/reload", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["reloaded"] is True


def test_model_reload_token_incorrecto_401(client):
    resp = client.post("/model/reload", headers={"X-Admin-Token": "token-tes"})
    assert resp.status_code == 401


def test_reload_carga_nueva_version(client_factory, train_small):
    c, model_dir = client_factory(seed=41)
    v1 = c.get("/health").json()["model_version"]

    train_small(model_dir, seed=42)  # reentreno en caliente sobre el mismo volumen
    resp = c.post("/model/reload", headers=ADMIN_HEADERS)

    assert resp.status_code == 200
    body = resp.json()
    assert body["previous_version"] == v1
    assert body["model_version"] != v1
    assert c.get("/health").json()["model_version"] == body["model_version"]
    assert c.post("/predict", json=CUSTOMER).json()["model_version"] == body["model_version"]


def test_reload_sin_artefactos_404_y_conserva_modelo(client_factory, tmp_path):
    from pathlib import Path

    c, model_dir = client_factory(seed=43)
    v1 = c.get("/health").json()["model_version"]
    (Path(model_dir) / "reference.csv").unlink()

    resp = c.post("/model/reload", headers=ADMIN_HEADERS)

    assert resp.status_code == 404
    assert "reference.csv" in resp.json()["detail"]
    assert c.get("/health").json()["model_version"] == v1
    assert c.post("/predict", json=CUSTOMER).status_code == 200


def test_reload_con_metadata_corrupta_500_y_conserva_modelo(client_factory):
    from pathlib import Path

    c, model_dir = client_factory(seed=44)
    v1 = c.get("/health").json()["model_version"]
    (Path(model_dir) / "metadata.json").write_text("{corrupto", encoding="utf-8")

    resp = c.post("/model/reload", headers=ADMIN_HEADERS)

    assert resp.status_code == 500
    assert "No se pudo cargar el modelo" in resp.json()["detail"]
    assert c.get("/health").json()["model_version"] == v1


def test_arranque_con_artefactos_corruptos_sirve_503_y_reload_recupera(
    client_factory, train_small, tmp_path
):
    model_dir = tmp_path / "corrupto"
    train_small(model_dir, seed=51)
    (model_dir / "metadata.json").write_text("{x", encoding="utf-8")

    c, _ = client_factory(model_dir=model_dir, train_model=False)
    assert c.get("/health").status_code == 503
    assert c.post("/predict", json=CUSTOMER).status_code == 503

    train_small(model_dir, seed=51)  # reparamos los artefactos
    assert c.post("/model/reload", headers=ADMIN_HEADERS).status_code == 200
    assert c.get("/health").status_code == 200


# --------------------------------------------------------------------------- drift


def test_drift_flujo(client):
    # Sin datos suficientes -> 409
    assert client.get("/monitoring/drift").status_code == 409
    # Alimentamos el buffer con predicciones
    for _ in range(12):
        client.post("/predict", json=CUSTOMER)
    resp = client.get("/monitoring/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert "features" in body and body["n_current"] >= 10
    assert body["model_version"] == client.get("/health").json()["model_version"]


def test_drift_sin_modelo_503(client_sin_modelo):
    assert client_sin_modelo.get("/monitoring/drift").status_code == 503


def test_drift_sin_cambio_de_distribucion(client_factory):
    c, _ = client_factory(seed=31, drift_min_rows=50)
    customers = _customers_from(generate_dataset(400, seed=98))
    assert c.post("/predict/batch", json={"customers": customers}).status_code == 200

    body = c.get("/monitoring/drift").json()
    assert body["drifted_features"] == []
    assert body["predictions"]["n_current"] == 400
    assert body["predictions"]["drift"] is False
    assert body["drift_detected"] is False


def test_drift_detecta_shift_via_api(client_factory):
    c, _ = client_factory(seed=32, drift_min_rows=50)
    df = generate_dataset(300, seed=99)
    df["monthly_charges"] = (df["monthly_charges"] * 2 + 40).round(2)  # fuerte, dentro del rango
    assert c.post("/predict/batch", json={"customers": _customers_from(df)}).status_code == 200

    body = c.get("/monitoring/drift").json()
    assert body["drift_detected"] is True
    assert "monthly_charges" in body["drifted_features"]
    assert body["features"]["monthly_charges"]["psi"] > 0.2
    assert body["features"]["monthly_charges"]["ks_pvalue"] < 0.01
    assert body["features"]["contract_type"]["type"] == "categorical"
    assert body["predictions"]["n_reference"] > 0


def test_drift_predicciones_solo_de_la_version_en_servicio(client_factory, train_small):
    c, model_dir = client_factory(seed=61)  # drift_min_rows=5
    for _ in range(6):
        c.post("/predict", json=CUSTOMER)
    assert c.get("/monitoring/drift").json()["predictions"] is not None

    train_small(model_dir, seed=62)
    assert c.post("/model/reload", headers=ADMIN_HEADERS).status_code == 200

    body = c.get("/monitoring/drift").json()
    assert body["n_current"] == 6  # las features del trafico anterior siguen valiendo
    assert body["predictions"] is None  # las puntuaciones del modelo anterior, no

    for _ in range(5):
        c.post("/predict", json=CUSTOMER)
    assert c.get("/monitoring/drift").json()["predictions"]["n_current"] == 5


def test_buffer_respeta_tamano_maximo(client_factory):
    c, _ = client_factory(seed=63, drift_buffer_size=20, drift_min_rows=5)
    for _ in range(30):
        c.post("/predict", json=CUSTOMER)
    assert c.get("/monitoring/drift").json()["n_current"] == 20
