"""DSN opcional de GlitchTip/Sentry. Sin variable, la API no cambia."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def maybe_init_sentry(dsn: str | None = None) -> bool:
    value = os.environ.get("SENTRY_DSN", "") if dsn is None else dsn
    value = value.strip()
    if not value:
        return False
    try:
        import sentry_sdk
    except ImportError:
        logger.warning("SENTRY_DSN esta definido pero sentry-sdk no esta instalado")
        return False
    sentry_sdk.init(dsn=value, traces_sample_rate=0.0, send_default_pii=False)
    return True
