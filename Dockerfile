# syntax=docker/dockerfile:1
FROM python:3.14-slim AS builder
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# requirements.lock fija versiones exactas (generado con `make lock`): builds reproducibles
COPY requirements.lock .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements.lock

FROM python:3.14-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /models && chown -R appuser:appuser /models
COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
# UID numerico: el host puede resolverlo aunque no conozca el nombre (hadolint DL3066)
USER 10001
EXPOSE 8000
# Readiness: 200 solo con modelo cargado. Sin modelo el contenedor figura como "unhealthy",
# que es exactamente lo que un operador quiere ver (liveness pura: /health/live).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"]
# --no-access-log: la propia API escribe una linea JSON por peticion (con X-Request-ID)
CMD ["uvicorn", "churn.serving.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
