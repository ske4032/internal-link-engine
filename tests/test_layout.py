"""Issue #1 step (f): the trivial test, made to earn its place.

`pytest` has to collect something or some CI configs treat the run as a failure. A test
that asserts `True` would satisfy that and catch nothing. The thing that actually breaks
is the package layout: import-linter's five contracts name modules by dotted path, mypy
and deptry walk the same tree, and a missing `__init__.py` turns a package into an
implicit namespace package that lint-imports cannot see.
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from configparser import ConfigParser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The layers of the `layers` contract, plus the two modules outside it.
SUBPACKAGES = (
    "api",
    "pipeline",
    "discovery",
    "anchor",
    "audit",
    "graph",
    "embedding",
    "models",
    "ingest",
    "ml",
)

# The split enforced by the `algorithms-are-pure` contract, not by convention.
GRAPH_MODULES = ("repo", "algorithms")

EXPECTED_CONTRACTS = {
    "layers",
    "domain-independence",
    "algorithms-are-pure",
    "api-excludes-ml",
    "models-are-leaves",
}


def _contract_sections(parser: ConfigParser) -> list[str]:
    return [name for name in parser.sections() if name.startswith("importlinter:contract:")]


def _read_importlinter() -> ConfigParser:
    path = REPO_ROOT / ".importlinter"
    assert path.is_file(), f"{path} is missing — issue #1 step (c)"
    parser = ConfigParser()
    parser.read_string(path.read_text(encoding="utf-8"))
    return parser


def test_root_package_is_importable() -> None:
    package = importlib.import_module("linking_engine")
    assert package.__name__ == "linking_engine"
    assert package.__file__ is not None, "linking_engine has no __init__.py"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_is_importable_and_is_a_real_package(name: str) -> None:
    module = importlib.import_module(f"linking_engine.{name}")
    assert module.__file__ is not None, (
        f"linking_engine.{name} resolved as a namespace package: add an __init__.py, "
        "or lint-imports and mypy will not see it"
    )
    assert Path(module.__file__).name == "__init__.py", (
        f"linking_engine.{name} is a module, not a package: {module.__file__}"
    )


@pytest.mark.parametrize("name", GRAPH_MODULES)
def test_graph_is_split_into_repo_and_algorithms(name: str) -> None:
    module = importlib.import_module(f"linking_engine.graph.{name}")
    assert module.__file__ is not None
    assert Path(module.__file__).name == f"{name}.py", (
        f"expected graph/{name}.py, got {module.__file__}"
    )


def test_importlinter_declares_the_five_contracts() -> None:
    parser = _read_importlinter()
    assert parser["importlinter"]["root_package"] == "linking_engine"
    names = {section.split(":")[-1] for section in _contract_sections(parser)}
    assert names == EXPECTED_CONTRACTS, (
        f"lint-imports must report 5 contracts kept; declared: {sorted(names)}"
    )


def test_every_module_named_in_the_contracts_exists() -> None:
    parser = _read_importlinter()
    targets: set[str] = set()
    for section in _contract_sections(parser):
        for key in ("modules", "source_modules"):
            targets.update(parser[section].get(key, "").split())
        layers = parser[section].get("layers", "").split()
        targets.update(f"linking_engine.{layer}" for layer in layers)

    # Guards against a parsing bug quietly emptying the loop below. `ingest` and `ml`
    # are deliberately absent: they sit outside the layered contract.
    assert len(targets) >= 8, f"parsed only {sorted(targets)} out of .importlinter"
    assert {"linking_engine.api", "linking_engine.models"} <= targets, "layers unparsed"
    assert "linking_engine.graph.algorithms" in targets, (
        "algorithms-are-pure is the load-bearing contract and names graph.algorithms"
    )

    missing = []
    for name in sorted(targets):
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    assert not missing, f"named in .importlinter but not importable: {missing}"


def test_interpreter_is_inside_the_supported_range() -> None:
    assert (3, 12) <= sys.version_info[:2] < (3, 14), (
        f"running on {sys.version_info[:2]}; the ML tree does not support 3.14 yet"
    )


def test_pyproject_pins_python_below_314() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requires = "".join(pyproject["project"]["requires-python"].split())
    assert requires == ">=3.12,<3.14", (
        f"requires-python is {requires!r}; 3.14 is not yet supported by the ML tree"
    )


def test_pyproject_pins_the_numpy_major() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = ["".join(dep.split()) for dep in pyproject["project"]["dependencies"]]
    numpy = [pin for pin in pins if pin.startswith("numpy")]
    assert numpy == ["numpy>=2.0,<3"], (
        f"numpy pin is {numpy}; hdbscan, umap-learn and lightgbm each lag NumPy majors"
    )
