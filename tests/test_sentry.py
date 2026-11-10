import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from churn.sentry import maybe_init_sentry


def test_missing_dsn_does_nothing(monkeypatch) -> None:
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert maybe_init_sentry() is False


def test_dsn_without_sdk_warns_and_returns_false(monkeypatch, caplog) -> None:
    import builtins

    real_import = builtins.__import__

    def fail_sentry(name, *args, **kwargs):
        if name == "sentry_sdk":
            raise ImportError("sentry-sdk no esta instalado")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_sentry)
    monkeypatch.setenv("SENTRY_DSN", "http://public@glitchtip.local/1")
    with caplog.at_level(logging.WARNING):
        assert maybe_init_sentry() is False
    assert "sentry-sdk" in caplog.text


@pytest.mark.parametrize("explicit", [False, True], ids=["environment", "explicit-override"])
def test_dsn_initializes_sdk_with_privacy_defaults(monkeypatch, explicit) -> None:
    init = Mock()
    monkeypatch.setitem(sys.modules, "sentry_sdk", SimpleNamespace(init=init))
    dsn = "https://public@monitoring.example/1"
    monkeypatch.setenv(
        "SENTRY_DSN", "https://other@monitoring.example/2" if explicit else f" {dsn} "
    )

    assert maybe_init_sentry(f" {dsn} " if explicit else None) is True

    init.assert_called_once_with(dsn=dsn, traces_sample_rate=0.0, send_default_pii=False)


@pytest.mark.parametrize("dsn", ["", " ", "\t\n"])
def test_empty_explicit_dsn_disables_sdk_even_when_environment_is_configured(
    monkeypatch, dsn
) -> None:
    init = Mock()
    monkeypatch.setitem(sys.modules, "sentry_sdk", SimpleNamespace(init=init))
    monkeypatch.setenv("SENTRY_DSN", "https://public@monitoring.example/1")

    assert maybe_init_sentry(dsn) is False

    init.assert_not_called()


def test_sdk_initialization_error_is_not_reported_as_success(monkeypatch) -> None:
    init = Mock(side_effect=ValueError("invalid DSN"))
    monkeypatch.setitem(sys.modules, "sentry_sdk", SimpleNamespace(init=init))

    with pytest.raises(ValueError, match="invalid DSN"):
        maybe_init_sentry("invalid")
