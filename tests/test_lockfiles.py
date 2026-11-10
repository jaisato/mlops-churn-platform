"""Los lockfiles respetan los rangos que declaran requirements*.txt.

Dependabot sube los suelos de requirements*.txt pero no regenera los lockfiles, y la CI y la
imagen instalan desde estos: con `pydantic>=2.14.2` declarado, los dos locks seguian en
2.14.1 (y mutmut/mypy por debajo de su suelo en el de desarrollo), asi que el verde de la CI
no probaba lo declarado. Este test lo detecta sin red; `make lock` lo arregla.
"""

from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO = Path(__file__).resolve().parents[1]
PAIRS = [
    ("requirements.txt", "requirements.lock"),
    ("requirements-dev.txt", "requirements-dev.lock"),
]


def declared_requirements(path: Path) -> list[Requirement]:
    """Requisitos de un requirements*.txt, siguiendo los `-r` (comentarios y vacias fuera)."""
    requirements: list[Requirement] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r"):
            requirements += declared_requirements(path.parent / line[2:].strip())
            continue
        requirements.append(Requirement(line))
    return requirements


def locked_versions(path: Path) -> dict[str, list[str]]:
    """Versiones fijadas por paquete; `--universal` puede fijar varias con marcadores distintos."""
    pins: dict[str, list[str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith((" ", "#")):
            continue
        name, _, version = raw.split(";", 1)[0].strip().partition("==")
        pins.setdefault(canonicalize_name(name), []).append(version.strip())
    return pins


@pytest.mark.parametrize(("declared", "lock"), PAIRS)
def test_lockfile_cumple_los_rangos_declarados(declared, lock):
    pins = locked_versions(REPO / lock)
    problems = []
    for requirement in declared_requirements(REPO / declared):
        versions = pins.get(canonicalize_name(requirement.name))
        if not versions:
            problems.append(f"{requirement.name} no esta en {lock}")
            continue
        problems += [
            f"{requirement.name}=={version} no cumple '{requirement.specifier}'"
            for version in versions
            if not requirement.specifier.contains(version, prereleases=True)
        ]
    assert not problems, (
        f"{lock} desfasado respecto a {declared}; regeneralo con `make lock`: {problems}"
    )


def test_el_parser_detecta_un_pin_por_debajo_del_suelo(tmp_path):
    (tmp_path / "base.txt").write_text("pydantic>=2.14.2,<3.0  # comentario\n", encoding="utf-8")
    (tmp_path / "dev.txt").write_text("-r base.txt\npytest>=9.2.0\n", encoding="utf-8")
    (tmp_path / "dev.lock").write_text(
        "# cabecera\npydantic==2.14.1\n    # via -r base.txt\npytest==9.2.0\n"
        "numpy==2.4.6 ; python_full_version < '3.12'\n"
        "numpy==2.5.3 ; python_full_version >= '3.12'\n",
        encoding="utf-8",
    )
    requirements = {r.name: r for r in declared_requirements(tmp_path / "dev.txt")}
    pins = locked_versions(tmp_path / "dev.lock")
    assert pins["numpy"] == ["2.4.6", "2.5.3"]
    assert not requirements["pydantic"].specifier.contains(pins["pydantic"][0])
    assert requirements["pytest"].specifier.contains(pins["pytest"][0])
