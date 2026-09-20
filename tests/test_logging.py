import json
import logging

import pytest

from churn.logging_conf import UVICORN_LOGGERS, JsonFormatter, configure_logging, request_id_var


def _record(msg="hola %s", args=("mundo",), level=logging.INFO, exc_info=None):
    record = logging.LogRecord(
        name="churn.test", level=level, pathname=__file__, lineno=1, msg=msg, args=args,
        exc_info=exc_info,
    )
    record.created = 1_700_000_000.5  # 2023-11-14T22:13:20.5Z
    return record


def test_formato_json_con_campos_basicos():
    payload = json.loads(JsonFormatter().format(_record()))
    assert payload == {
        "ts": "2023-11-14T22:13:20.500000+00:00",
        "level": "INFO",
        "logger": "churn.test",
        "msg": "hola mundo",
    }


def test_timestamp_es_el_del_evento_no_el_del_formateo():
    """Dos formateos del mismo record deben dar el mismo ts (viene de record.created)."""
    record = _record()
    a = json.loads(JsonFormatter().format(record))["ts"]
    b = json.loads(JsonFormatter().format(record))["ts"]
    assert a == b == "2023-11-14T22:13:20.500000+00:00"


def test_incluye_traza_de_excepcion():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        record = _record(msg="fallo", args=(), level=logging.ERROR, exc_info=sys.exc_info())
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "ERROR"
    assert "RuntimeError: boom" in payload["exc"]


def test_mensajes_no_ascii_se_conservan():
    payload = json.loads(JsonFormatter().format(_record(msg="señal año ñ", args=())))
    assert payload["msg"] == "señal año ñ"


def test_incluye_campos_extra_y_request_id_del_record():
    record = _record(msg="acceso", args=())
    record.request_id = "req-1"
    record.http = {"method": "GET", "status": 200}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["request_id"] == "req-1"
    assert payload["http"] == {"method": "GET", "status": 200}
    assert "args" not in payload and "levelno" not in payload  # atributos estandar fuera


def test_request_id_del_contexto_cuando_el_record_no_lo_trae():
    token = request_id_var.set("ctx-9")
    try:
        payload = json.loads(JsonFormatter().format(_record()))
    finally:
        request_id_var.reset(token)
    assert payload["request_id"] == "ctx-9"
    assert "request_id" not in json.loads(JsonFormatter().format(_record()))


def test_valores_no_serializables_se_convierten_a_texto():
    record = _record(msg="x", args=())
    record.ruta = __import__("pathlib").Path("/tmp/a")
    assert json.loads(JsonFormatter().format(record))["ruta"] == "/tmp/a"


@pytest.fixture()
def root_logger_restaurado():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield root
    root.handlers, root.level = handlers, level
    root.setLevel(level)


def test_configure_logging_instala_un_unico_handler_json(root_logger_restaurado):
    configure_logging("debug")
    root = root_logger_restaurado
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)

    configure_logging("WARNING")  # idempotente: no acumula handlers
    assert len(root.handlers) == 1
    assert root.level == logging.WARNING


def test_configure_logging_redirige_uvicorn_al_formato_json(root_logger_restaurado):
    for name in UVICORN_LOGGERS:
        logging.getLogger(name).addHandler(logging.NullHandler())
        logging.getLogger(name).propagate = False
    configure_logging("INFO")
    for name in UVICORN_LOGGERS:
        assert logging.getLogger(name).handlers == []
        assert logging.getLogger(name).propagate is True
