"""Logging estructurado en JSON (una linea por evento, apto para Loki/CloudWatch/ELK).

Cada linea lleva `ts`, `level`, `logger`, `msg`, el `request_id` de la peticion en curso
(si lo hay) y cualquier campo pasado en `extra=` al logger. Los loggers de uvicorn se
redirigen al mismo handler para que todo el proceso escriba el mismo formato.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")
_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("x", logging.INFO, __file__, 0, "", (), None))
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            # record.created es el instante real del evento (no el de formateo)
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None) or request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():  # campos pasados con extra=
            if key not in _STANDARD_ATTRS and key != "request_id" and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for name in UVICORN_LOGGERS:  # un unico formato para todo el proceso
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
