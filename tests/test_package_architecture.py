# ruff: noqa: S101
"""Protect the Generation and Dataset package responsibility boundaries."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest

from src import datasets, domain, generation
from src.datasets import dataset_packages
from src.datasets.contracts import dataset_contracts_metadata as dataset_metadata
from src.datasets.contracts import dataset_contracts_views as dataset_views
from src.datasets.packages import dataset_packages_builder

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
_GENERATION_ROOT = _SOURCE_ROOT / "generation"
_DATASET_ROOT = _SOURCE_ROOT / "datasets"

_RESPONSIBILITY_PACKAGES = (
    "src.generation.cases",
    "src.generation.contracts",
    "src.generation.publication",
    "src.generation.runtime",
    "src.generation.validation",
    "src.datasets.contracts",
    "src.datasets.packages",
    "src.datasets.preprocessing",
    "src.datasets.runtime",
)


def _module_identity(path: Path) -> tuple[str, str]:
    """Return the import name and relative-import package for one source path."""
    parts = path.relative_to(_REPOSITORY_ROOT).with_suffix("").parts
    if path.name == "__init__.py":
        module_name = ".".join(parts[:-1])
        return module_name, module_name
    module_name = ".".join(parts)
    return module_name, module_name.rpartition(".")[0]


def _import_targets(path: Path) -> set[str]:
    """Resolve statically declared absolute and relative import targets."""
    _module_name, package_name = _module_identity(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                base_name = importlib.util.resolve_name(relative_name, package_name)
            else:
                base_name = node.module or ""
            if base_name:
                targets.add(base_name)
            targets.update(f"{base_name}.{alias.name}" if base_name else alias.name for alias in node.names if alias.name != "*")
    return targets


def _implementation_graph() -> dict[str, set[str]]:
    """Return concrete Generation and Dataset module dependencies."""
    paths = tuple(path for package_root in (_GENERATION_ROOT, _DATASET_ROOT) for path in package_root.rglob("*.py") if path.name != "__init__.py")
    modules = {_module_identity(path)[0]: path for path in paths}
    return {module_name: _import_targets(path).intersection(modules) for module_name, path in modules.items()}


def _cyclic_edges(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    """Return edges that cannot be removed by dependency-first traversal."""
    remaining = {module_name: set(dependencies) for module_name, dependencies in graph.items()}
    while leaves := {module_name for module_name, dependencies in remaining.items() if not dependencies}:
        remaining = {module_name: dependencies.difference(leaves) for module_name, dependencies in remaining.items() if module_name not in leaves}
    return remaining


def _facade_root_targets(path: Path) -> set[str]:
    """Return direct root-module targets declared by one lazy facade."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "_MODULES" for target in node.targets):
            mapping = ast.literal_eval(node.value)
            return {target for target in mapping.values() if "." not in target}
    msg = f"{path} does not declare its public _MODULES facade mapping"
    raise AssertionError(msg)


def _internal_imports(package_root: Path) -> list[tuple[Path, str]]:
    """Return imports declared anywhere below one responsibility package."""
    return [(path, target) for path in package_root.rglob("*.py") for target in _import_targets(path)]


@pytest.mark.parametrize("module_name", _RESPONSIBILITY_PACKAGES)
def test_responsibility_packages_import_their_public_facades(module_name: str) -> None:
    """Keep each declared Generation and Dataset responsibility package importable."""
    package = import_module(module_name)
    assert package.__all__
    assert all(getattr(package, name) is not None for name in package.__all__)


def test_root_public_facades_resolve_to_modules() -> None:
    """Resolve every declared root facade without depending on import order."""
    for root_package in (generation, datasets):
        assert root_package.__all__
        assert all(isinstance(getattr(root_package, name), ModuleType) for name in root_package.__all__)


@pytest.mark.parametrize(
    ("package_name", "public_name", "implementation_name"),
    [
        (
            "src.generation.contracts",
            "scalar_handoff",
            "src.generation.contracts.generation_contracts_scalar_handoff",
        ),
        (
            "src.experiments.config",
            "temporal",
            "src.experiments.config.experiments_config_temporal",
        ),
        (
            "src.learning",
            "temporal",
            "src.learning.learning_temporal",
        ),
    ],
)
def test_phase_three_public_modules_resolve_to_their_single_owners(
    package_name: str,
    public_name: str,
    implementation_name: str,
) -> None:
    """Keep the new lazy public names identical to their concrete owners."""
    package = import_module(package_name)
    implementation = import_module(implementation_name)
    assert getattr(package, public_name) is implementation


def test_canonical_cli_modules_are_discoverable() -> None:
    """Keep the two maintained ``python -m`` module paths discoverable."""
    for module_name in ("src.generation.cli.cli_generation", "src.datasets.dataset_packages"):
        specification = importlib.util.find_spec(module_name)
        assert specification is not None
        assert specification.origin is not None


def test_generation_cli_module_exposes_help() -> None:
    """Exercise Generation help locally; Dataset help remains install-check owned."""
    completed = subprocess.run(
        [sys.executable, "-m", "src.generation.cli.cli_generation", "--help"],
        cwd=_REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_concrete_implementation_import_graph_is_acyclic() -> None:
    """Reject cycles between concrete modules while excluding facade initializers."""
    cyclic = _cyclic_edges(_implementation_graph())
    detail = "\n".join(f"{module}: {sorted(edges)}" for module, edges in sorted(cyclic.items()))
    assert not cyclic, detail


def test_only_public_root_modules_and_the_persisted_dataset_facade_remain() -> None:
    """Reject restored flat implementations or compatibility-only module shims."""
    generation_targets = _facade_root_targets(_GENERATION_ROOT / "__init__.py")
    generation_flat = {path.stem for path in _GENERATION_ROOT.glob("generation_*.py")}
    assert generation_flat <= generation_targets

    dataset_targets = _facade_root_targets(_DATASET_ROOT / "__init__.py") | {"dataset_packages"}
    dataset_flat = {path.stem for path in _DATASET_ROOT.glob("dataset_*.py")}
    assert dataset_flat <= dataset_targets
    assert (_DATASET_ROOT / "dataset_packages.py").is_file()


def test_lower_generation_packages_do_not_import_workflow_or_cli() -> None:
    """Keep orchestration and command parsing above all responsibility packages."""
    forbidden = ("src.generation.generation_workflow", "src.generation.cli")
    violations = [
        (path.relative_to(_REPOSITORY_ROOT), target)
        for package_name in ("cases", "contracts", "publication", "runtime", "validation")
        for path, target in _internal_imports(_GENERATION_ROOT / package_name)
        if any(target == prefix or target.startswith(f"{prefix}.") for prefix in forbidden)
    ]
    assert not violations, violations


@pytest.mark.parametrize(
    ("contract_root", "package_name"),
    [
        (_GENERATION_ROOT / "contracts", "src.generation"),
        (_DATASET_ROOT / "contracts", "src.datasets"),
    ],
)
def test_contract_packages_do_not_reverse_import_upper_layers(
    contract_root: Path,
    package_name: str,
) -> None:
    """Keep contracts independent of sibling runtime, workflow, and publication layers."""
    contract_name = f"{package_name}.contracts"
    violations = [
        (path.relative_to(_REPOSITORY_ROOT), target)
        for path, target in _internal_imports(contract_root)
        if (target == package_name or target.startswith(f"{package_name}."))
        and not (target == contract_name or target.startswith(f"{contract_name}."))
    ]
    assert not violations, violations


def test_dataset_package_lifecycle_preserves_canonical_builder_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route the persisted facade callable to one package-builder implementation."""
    package_services = import_module("src.datasets.packages")
    build = dataset_packages.build_campaign_packages
    qualified_name = f"{build.__module__}.{build.__name__}"
    assert package_services.build_campaign_packages is build
    assert qualified_name == "src.datasets.dataset_packages.build_campaign_packages"
    assert dataset_packages.__name__ == dataset_metadata.BUILDER_MODULE

    calls: list[tuple[object, Path | str | None]] = []
    result = object()

    def build_implementation(
        campaign: object,
        *,
        storage_root: Path | str | None = None,
    ) -> object:
        calls.append((campaign, storage_root))
        return result

    monkeypatch.setattr(dataset_packages_builder, "build_campaign_packages", build_implementation)
    campaign = object()
    storage_root = Path("test-storage")
    assert build(campaign, storage_root=storage_root) is result  # type: ignore[arg-type]
    assert calls == [(campaign, storage_root)]


def test_steady_contract_inspection_is_deterministic_and_task_owned() -> None:
    """Project the registered TaskSpec directly without a copied steady contract."""
    task = domain.tasks.registry.get_task("steady_flow")
    first = dataset_views.inspect_contract("steady_flow")
    second = dataset_views.inspect_contract("steady_flow")
    assert first == second
    assert first.contract is second.contract is task
    assert first.contract_digest == task.data_contract_digest
