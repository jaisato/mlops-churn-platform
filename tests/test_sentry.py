import logging

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
