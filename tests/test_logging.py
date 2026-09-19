import json
import logging

import pytest

from churn.logging_conf import JsonFormatter, configure_logging


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
