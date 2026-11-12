import warnings
from pathlib import Path

from mutmut.configuration import ProcessIsolation, _load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mutmut_config_loads_without_deprecation_warnings_and_targets_pure_modules(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = _load_config()

    assert not caught, [str(w.message) for w in caught]
    assert cfg.source_paths == [Path("src")]
    assert cfg.process_isolation == ProcessIsolation.FORKSERVER
    assert cfg.log_to_file is True

    pure_modules = sorted(
        str(p.relative_to(REPO_ROOT))
        for sub in ("src/churn/monitoring", "src/churn/data")
        for p in (REPO_ROOT / sub).glob("*.py")
    )
    assert pure_modules
    for mod in pure_modules:
        assert cfg.should_mutate(mod), mod

    assert not cfg.should_mutate("src/churn/config.py")
    assert not cfg.should_mutate("src/churn/serving/main.py")
    assert not cfg.should_mutate("src/churn/training/train.py")

    assert cfg.pytest_add_cli_args_test_selection
    for test_file in cfg.pytest_add_cli_args_test_selection:
        assert (REPO_ROOT / test_file).is_file(), test_file
